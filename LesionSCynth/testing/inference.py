import argparse
import importlib
import time

import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchio as tio
from pathlib import Path
import sys
import warnings
import logging
from datetime import datetime
import json
from skimage import measure
import pickle

from ..configs.config import Config
from ..im_utils import resample_to_ref, sitk_to_numpy, new_image_from_ref, resample_spacing


def import_config(config_path: Path):
    """ Function to import config while allowing for absolute paths to the config file. """
    if config_path.name.endswith('.pkl'):
        with open(config_path, 'rb') as f:
            return pickle.load(f)
    else:
        spec = importlib.util.spec_from_file_location('config', config_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['config'] = module
        spec.loader.exec_module(module)

        return module.config


def load_model(model_path, config, device):
    model = config.model_class.load_from_checkpoint(model_path, config=config, map_location=device)
    model.eval()
    # Check that model has indeed been loaded to the correct device
    if model.device.type != device.type:
        model.to(device)
    return model


def get_prediction(subject, model, pred_dir, config, device, channels_dimension, softmax=True
                   ) -> tio.ScalarImage:
    pmap_path = pred_dir / f'{subject["name"]}_pmap.nii.gz'
    if pmap_path.exists():
        warn_msg = f'Prediction already exists for {subject["name"]}. Skipping prediction generation.'
        warnings.warn(warn_msg)
        prediction = tio.ScalarImage(pmap_path)
        return prediction

    if hasattr(config, 'patch_overlap') and config.patch_overlap is not None:
        patch_overlap = config.patch_overlap
    else:
        patch_overlap = tuple(p // 2 for p in config.patch_size)

    grid_sampler = tio.inference.GridSampler(subject, config.patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=config.validation_batch_size)  # type: ignore
    aggregator = tio.inference.GridAggregator(grid_sampler)

    with torch.no_grad():
        for i, patches_batch in enumerate(patch_loader):
            inputs = patches_batch[config.modalities[0]][tio.DATA]
            inputs = inputs.to(dtype=torch.float32, device=device)
            locations = patches_batch[tio.LOCATION]

            logits = model(inputs)

            # Only want the final output and not the intermediate outputs for deep supervision
            probabilities = logits[-1] if config.deep_supervision_levels > 1 else logits
            if softmax:
                probabilities = probabilities.softmax(dim=channels_dimension)
            if isinstance(probabilities, dict) and len(probabilities) == 1:
                probabilities = probabilities[config.modalities[0]]
            aggregator.add_batch(probabilities, locations)

    foreground = aggregator.get_output_tensor()
    affine = subject.get_first_image()[tio.AFFINE]
    if foreground.shape[0] == 2:
        prediction = tio.ScalarImage(tensor=foreground[1, ...].unsqueeze(dim=0), affine=affine)
    else:
        prediction = tio.ScalarImage(tensor=foreground, affine=affine)

    return prediction


def uncrop_image(ref_im, crop_im, crop_start_coords):
    """
    Adapted from spinalcordtoolbox https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/deepseg_/sc.py#LL379C1-L392C1
    Paste the cropped segmentation image into the original image space.
    """
    x_crop_lst, y_crop_lst, z_crop_lst = crop_start_coords

    data_crop = sitk_to_numpy(crop_im)
    data_uncrop = np.zeros_like(sitk_to_numpy(ref_im), dtype=np.float32)

    crop_size_x, crop_size_y = data_crop.shape[:2]

    for i_z, zz in enumerate(z_crop_lst):
        z_slice = data_crop[:, :, i_z]
        x_start, y_start = int(x_crop_lst[i_z]), int(y_crop_lst[i_z])
        x_end = x_start + crop_size_x if x_start + crop_size_x < data_uncrop.shape[0] else data_uncrop.shape[0]
        y_end = y_start + crop_size_y if y_start + crop_size_y < data_uncrop.shape[1] else data_uncrop.shape[1]
        try:
            data_uncrop[x_start:x_end, y_start:y_end, zz] = z_slice[0:(x_end - x_start), 0:(y_end - y_start)]
        except:
            continue

    return new_image_from_ref(data_uncrop, ref_im, numpy_to_sitk=True)


def process_cropped_shifted(subject, model, input_parent_dir: Path, orig_dir: Path, preds_dir: Path, config: Config,
                            path_to_bounds: Path, device: torch.device, CHANNELS_DIMENSION: int):
    volume_id = subject['name']
    intermediate_dir = input_parent_dir / volume_id / 'intermediate_files'

    preds_dir = preds_dir / volume_id
    preds_dir.mkdir(exist_ok=True, parents=True)

    with open(path_to_bounds, 'r') as f:
        coords_dict = json.load(f)

    seg_im = get_prediction(subject, model, preds_dir, config, device, CHANNELS_DIMENSION)
    seg_im = seg_im.as_sitk()  # convert to SITK image from PyTorch Tensor

    # Re-sample to preprocessed space, in case there was padding applied before input to model
    gt_im = sitk.ReadImage(intermediate_dir / 'seg_cropped.nii.gz')
    seg_im = resample_to_ref(seg_im, gt_im, interpolator=sitk.sitkNearestNeighbor)

    if not (preds_dir / f'{volume_id}_pmap.nii.gz').exists():
        sitk.WriteImage(seg_im, preds_dir / f'{volume_id}_pmap.nii.gz')

    # Load the reference images
    # Use GT images, if they exist, as they are quicker to load
    def read_if_exists(path1, path2):
        return sitk.ReadImage(path1) if path1.exists() else sitk.ReadImage(path2)

    ref_im_orig = read_if_exists(orig_dir / volume_id / 'seg.nii.gz',
                                 orig_dir / volume_id / f'{config.modalities[0]}.nii.gz')
    ref_im_unpadded = read_if_exists(intermediate_dir / 'seg_cropped.nii.gz',
                                     intermediate_dir / f'{config.modalities[0]}_cropped.nii.gz')
    if (intermediate_dir / f'{config.modalities[0]}_sc_cropped.nii.gz').exists():
        ref_im_resampled = sitk.ReadImage(intermediate_dir / f'{config.modalities[0]}_sc_cropped.nii.gz')
    else:
        ref_im_resampled = sitk.DICOMOrient(ref_im_orig, 'LAS')
        ref_im_resampled = resample_spacing(ref_im_resampled, ref_im_unpadded.GetSpacing())

    # Uncrop the cropped and shifted image back to the original space
    ref_coords_uncrop = coords_dict[volume_id]
    seg_uncropped = uncrop_image(ref_im_resampled, seg_im, ref_coords_uncrop)
    seg_to_orig = resample_to_ref(seg_uncropped, ref_im_orig)
    sitk.WriteImage(seg_to_orig, preds_dir / f'{volume_id}_pmap_orig_space.nii.gz')

    # Binarise, if threshold given
    for threshold in args.thresholds:
        segmentation_path = preds_dir / f'seg_thresh_{threshold}.nii.gz'
        if not segmentation_path.exists():
            # binarise the pmap based on the threshold
            label = sitk.BinaryThreshold(seg_to_orig, lowerThreshold=threshold, upperThreshold=1e6, insideValue=1,
                                         outsideValue=0)
            sitk.WriteImage(label, segmentation_path)

    # Create symlinks to the original images
    for modality in config.modalities + ['seg']:
        symlink_path = preds_dir / f'{modality}.nii.gz'
        path_orig = Path(args.orig_dir) / volume_id / f'{modality}.nii.gz'
        if not (symlink_path.is_symlink() or symlink_path.exists()):
            symlink_path.symlink_to(path_orig.resolve())


def main(args):
    args.preds_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=args.preds_dir / 'prediction.log', level=logging.INFO)
    # Prepend the log with the time and date
    logging.info(f'Prediction started at {datetime.now()}')
    start_time = time.time()  # get current time

    config = import_config(args.config)
    config.update_params(mode='test')
    config.create_validation_transform(patch_size=config.patch_size)

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            warnings.warn('CUDA device requested but not available. Using CPU instead.')
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)

    CHANNELS_DIMENSION = 1

    data = config.data_module(args.input_dir, config)
    data.setup('test')  # Set up in 'test' mode
    print('Test set:', len(data.test_set), 'images')

    for mod_name, mod_path in args.model_file.items():
        model = load_model(mod_path, config, device)
        pred_dir = args.preds_dir / mod_name

        if pred_dir.exists():
            logging.warning(f'Prediction directory {pred_dir} already exists.')
            if args.overwrite:
                logging.warning('Overwriting existing predictions.')
            else:
                logging.warning('Skipping model. If you want to overwrite, use the --overwrite flag.')
                continue
        else:
            logging.info(f'Processing model {mod_name}')
            print(f'Processing model {mod_name}')

        # Loop through each "subject" (volume) in the test set.
        for subject in tqdm(data.test_set):
            if (args.volumes_subset is not None) and (subject['name'] not in args.volumes_subset):
                continue

            if args.safe_mode:
                try:
                    process_cropped_shifted(subject, model, args.input_dir, args.orig_dir, pred_dir, config,
                                            args.path_to_bounds, device, CHANNELS_DIMENSION)
                except Exception as e:
                    logging.exception(f'Error processing {subject["name"]}: {e}')
                    print(f'Error processing {subject["name"]}. Skipping... \nError message: {e}')
                    continue
            else:
                process_cropped_shifted(subject, model, args.input_dir, args.orig_dir, pred_dir, config,
                                        args.path_to_bounds, device, CHANNELS_DIMENSION)

    # Log the time taken
    end_time = time.time()
    logging.info(f'Prediction finished at {datetime.now()}')
    logging.info(f'Time taken: {end_time - start_time} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on the model.")
    parser.add_argument('--input_dir', '-i', type=Path, required=True,
                        help='Directory containing the subject/volume directories with the processed images')
    parser.add_argument('--orig_dir', '-org', type=Path, required=True, help=
                        'Directory containing the subject/volume directories with the original images, '
                        'whose names match the modalities argument e.g. t2.nii.gz/stir.nii.gz')
    parser.add_argument('--preds_dir', '-p', type=Path, default=None, help='Directory to save the output predictions')
    parser.add_argument('--model_dir', '-md', type=Path, default=None, help=
                        'If supplied, the model files (best_f1_*.ckpt and best_loss_*.ckpt) and config (multimodal.py) '
                        'will be derived from this. If those args are also supplied, they will override the model_dir.')
    parser.add_argument('--model_file', '-mf', type=Path, nargs='+', default=None,
                        help='Path to the saved model checkpoint(s), if --model_dir is not supplied. '
                             'Can supply multiple models but they should share the same config file.')
    parser.add_argument('--path_to_bounds', '-b', type=Path, default=None,
                        help='Path to coordinates for uncropping data preprocessed with cropping/shifting.')
    parser.add_argument('--config', '-c', type=Path, default=None,
                        help='Path to the config file, if --model_dir is not supplied')
    parser.add_argument('--thresholds', '-thr', type=float, nargs='+', default=[0.5],
                        help='Thresholds to apply to the pmap.')
    parser.add_argument('--volumes_subset', '-sub', default=None, nargs='+', help='Subset of volumes to process.')
    parser.add_argument('--min_size', '-m', type=int, default=None,
                        help='Minimum height of the images. Images will be padded above & below in the '
                             'superior-inferior axis if shorter than this length.')
    parser.add_argument('--device', '-d', type=str, default=None,
                        help='Torch device to use for inference. Default: cuda, if available. '
                             'Other options: cpu, or specify a specific GPU, e.g. cuda:0 or cuda:1.')
    parser.add_argument('--safe_mode', '-s', action='store_true',
                        help='If supplied, will catch exceptions and log them.')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='If supplied, will overwrite existing predictions.')
    args = parser.parse_args()

    # Process the args for model files and configuration
    # Ensure either model_dir is supplied or model_file and configuration are supplied
    if not (args.model_dir or (args.model_file and args.config)):
        raise argparse.ArgumentError(argument=None,
                                     message='Either model_dir or model_file & configuration must be supplied')
    # Construct the path to the model if it's not supplied
    if not args.model_file:
        model_files_best_loss = list(args.model_dir.glob('best_loss*.ckpt'))
        other_model_files = list(args.model_dir.glob('*.ckpt'))
        other_model_files = [f for f in other_model_files if f not in model_files_best_loss]

        args.model_file = {}
        if len(model_files_best_loss) == 1:
            args.model_file['best_loss'] = model_files_best_loss.pop()

        for f in (other_model_files + model_files_best_loss):
            args.model_file[f.stem] = f
    else:
        args.model_file = {f.name.replace('.ckpt', ''): f for f in args.model_file}

    if not args.config:
        # 1. Use pickle file if available.
        pkl_files = list(args.model_dir.glob('*.pkl'))
        if len(pkl_files) == 1:
            args.config = pkl_files[0]
            logging.warning(f'Found pickled config. Using {args.config}')
        elif len(pkl_files) > 1:
            raise FileNotFoundError(f'Multiple pickle files found in {args.model_dir}. Please specify the config file.')
        else:
            # 2. Otherwise, use the python file config in the directory, if there is only one
            config_files = list(args.model_dir.glob('*.py'))
            config_files = [f for f in config_files if f.name != 'config.py']  # Exclude the base config
            if len(config_files) == 1:
                args.config = config_files[0]
                logging.warning(f'No config file supplied. Using {args.config}')
            elif len(config_files) > 1:
                raise FileNotFoundError(f'Multiple config files found in {args.model_dir}. Please specify the config file.')
            else:
                raise FileNotFoundError(f'No config file found in {args.model_dir}')

    main(args)


