import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage.measure import label
import skimage.morphology
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
import argparse

from ..im_utils import (new_image_from_ref, sitk_to_numpy, dilate_slicewise, erode_slicewise,
                        check_matching_geom, resample_to_ref, read_orient_extract)


def check_and_write(im: sitk.Image, path: Path, overwrite: bool = False) -> None:
    """
    Saves a SimpleITK image to disk, raising an error if the file exists and overwrite is False.
    Args:
        im (sitk.Image): The image to write.
        path (Path): Destination path for the image.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    Returns:
        None
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"File {path} already exists. Add --overwrite to the args to overwrite.")
    else:
        sitk.WriteImage(im, path)


def setup_logging(out_dir: Path) -> str:
    """
    Sets up a log file with a timestamp in the specified output directory.
    Args:
        out_dir (Path): Directory in which to store the log file.
    Returns:
        str: Timestamp string used in the log filename.
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(filename=out_dir / f'intensity_analysis_{timestamp}.log', level=logging.INFO)
    return timestamp


def make_outdir(out_dir: Path) -> Path:
    """
    Ensures that the specified output directory exists.
    Args:
        out_dir (Path): Path to the output directory.
    Returns:
        Path: The same path that was passed in, after ensuring it exists.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def summary_stats(arr: np.ndarray, key_prefix: str = '') -> dict[str, float]:
    """
    Computes summary statistics for an array of intensity values.
    Args:
        arr (np.ndarray): Array of intensity values.
        key_prefix (str, optional): Prefix to prepend to each dictionary key. Defaults to ''.
    Returns:
        dict[str, float]: Dictionary containing statistical summaries of the input array.
    """
    return {
        f'{key_prefix}mean': np.mean(arr),
        f'{key_prefix}median': np.median(arr),
        f'{key_prefix}std': np.std(arr),
        f'{key_prefix}min': np.min(arr),
        f'{key_prefix}max': np.max(arr),
        f'{key_prefix}5th_percentile': np.percentile(arr, 5),
        f'{key_prefix}q1': np.percentile(arr, 25),
        f'{key_prefix}q3': np.percentile(arr, 75),
        f'{key_prefix}95th_percentile': np.percentile(arr, 95),
        f'{key_prefix}iqr': np.percentile(arr, 75) - np.percentile(arr, 25)
    }


def process_subject(args: argparse.Namespace, subj: Path, out_file_prefix: str) -> list[dict]:
    """
    Processes a single subject by loading images, resampling if needed, and extracting lesion statistics.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        subj (Path): Path to the subject folder.
        out_file_prefix (str): Prefix to use for output file names.
    Returns:
        list[dict]: A list of dictionaries containing lesion and spinal cord statistics.
    """
    try:
        anat_arr, anat_im = read_orient_extract(subj / args.anat_name)
        seg_arr, seg_im = read_orient_extract(args.seg_dir / subj.name / args.seg_name)
        sc_arr, sc_seg_im = read_orient_extract(args.sc_seg_dir / subj.name / args.sc_seg_name)

        if args.resample:
            seg_im, seg_arr = maybe_resample(seg_im, anat_im, seg_arr)
            sc_seg_im, sc_arr = maybe_resample(sc_seg_im, anat_im, sc_arr)

        voxel_size_mm3 = np.prod(seg_im.GetSpacing())

        if args.save_cc:
            # Compute and save the connected components of the lesion seg
            seg_arr = save_connected_components(seg_arr, seg_im, args, subj.name, out_file_prefix)
        else:
            seg_arr = seg_arr.astype(np.uint8)

        return collect_lesion_data(subj.name, anat_arr, seg_arr, sc_arr, sc_seg_im,
                                   seg_im, voxel_size_mm3, args, out_file_prefix)

    except Exception as e:
        logging.error(f"Error processing {subj.name}.\n{e}")
        return []


def maybe_resample(im: sitk.Image, ref_im: sitk.Image, arr: np.ndarray,
                   interpolator: int = sitk.sitkNearestNeighbor) -> tuple[sitk.Image, np.ndarray]:
    """
    Resamples the input image and array to match the reference image if their geometries differ.
    Args:
        im (sitk.Image): The input image to resample.
        ref_im (sitk.Image): The reference image to match.
        arr (np.ndarray): Numpy array corresponding to the input image.
        interpolator (int, optional): SimpleITK interpolation method. Defaults to sitk.sitkNearestNeighbor.
    Returns:
        tuple[sitk.Image, np.ndarray]: The resampled image and updated numpy array.
    """
    if not check_matching_geom(ref_im, im):
        im = resample_to_ref(im, ref_im, interpolator=interpolator)
        arr = sitk_to_numpy(im)
    return im, arr


def save_connected_components(seg_arr: np.ndarray, seg_im: sitk.Image, args: argparse.Namespace,
                              subj_name: str, out_file_prefix: str) -> np.ndarray:
    """
    Computes connected components from the segmentation, saves them as a new image, and returns the labeled array.
    Args:
        seg_arr (np.ndarray): Binary segmentation array.
        seg_im (sitk.Image): Reference image used for spatial metadata.
        args (argparse.Namespace): Parsed command-line arguments, including output directory and overwrite flag.
        subj_name (str): Subject identifier used in file naming.
        out_file_prefix (str): Prefix for the output file name.
    Returns:
        np.ndarray: Array with labeled connected components.
    """
    seg_arr = (seg_arr > 0).astype(np.uint8)
    seg_cc = label(seg_arr).astype(np.uint8)
    seg_cc_im = new_image_from_ref(seg_cc, seg_im)
    cc_out_dir = make_outdir(args.out_dir / 'connected_components' / subj_name)
    check_and_write(seg_cc_im, cc_out_dir / f'{out_file_prefix}cc.nii.gz', args.overwrite)
    return seg_cc


def collect_lesion_data(subj_name: str, anat_arr: np.ndarray, seg_arr: np.ndarray, sc_arr: np.ndarray,
                        sc_seg_im: sitk.Image, seg_im: sitk.Image, voxel_size_mm3: float,
                        args: argparse.Namespace, out_file_prefix: str) -> list[dict]:
    """
    Extracts statistics for lesions and spinal cord from anatomical and segmentation images.

    Args:
        subj_name (str): Subject identifier.
        anat_arr (np.ndarray): Anatomical image array.
        seg_arr (np.ndarray): Labeled lesion segmentation array.
        sc_arr (np.ndarray): Spinal cord segmentation array.
        sc_seg_im (sitk.Image): Spinal cord image with spatial metadata.
        seg_im (sitk.Image): Lesion segmentation image with spatial metadata.
        voxel_size_mm3 (float): Volume of a voxel in cubic millimeters.
        args (argparse.Namespace): Parsed command-line arguments including output directory and save flags.
        out_file_prefix (str): Prefix for output filenames.

    Returns:
        list[dict]: List of dictionaries with lesion and spinal cord statistics.
    """
    data = []
    sc_stats = summary_stats(anat_arr[sc_arr == 1], key_prefix='sc_')
    sc_no_les_stats = summary_stats(anat_arr[(sc_arr == 1) & (seg_arr == 0)], key_prefix='sc_no_les_')
    sc_volume = (sc_arr > 0).sum() * voxel_size_mm3

    # Dilate the lesion masks on each sagittal slice
    dilated_segs = {}
    for dilation_factor in [3, 5, 7]:
        dilated_segs[dilation_factor] = dilate_slicewise(seg_arr, skimage.morphology.disk(dilation_factor),
                                                         slice_axis=0, multi_values=True)
        if args.save_dilations:
            dilated_im = new_image_from_ref(dilated_segs[dilation_factor], seg_im, numpy_to_sitk=True)
            dil_out_dir = make_outdir(args.out_dir / 'dilated_lesions')
            check_and_write(dilated_im,
                            dil_out_dir / f'{out_file_prefix}{subj_name}_seg_dilated{dilation_factor}.nii.gz',
                            args.overwrite)

    # The spinal cord mask sometimes has higher intensities because of a partial volume effect with the CSF.
    # Erode the mask slightly to reduce this effect
    erosion_factor = 1
    sc_arr_eroded = erode_slicewise(sc_arr, skimage.morphology.disk(erosion_factor), slice_axis=0, multi_values=False)
    if args.save_dilations:
        sc_eroded_im = new_image_from_ref(sc_arr_eroded, sc_seg_im, numpy_to_sitk=True)
        eroded_out_dir = make_outdir(args.out_dir / 'eroded_spinal_cord')
        check_and_write(sc_eroded_im,
                        eroded_out_dir / f'{out_file_prefix}{subj_name}_sc_eroded{erosion_factor}.nii.gz',
                        args.overwrite)

    for les_id in np.unique(seg_arr)[1:]:
        lesion_volume = (seg_arr == les_id).sum() * voxel_size_mm3
        lesion_stats = summary_stats(anat_arr[seg_arr == les_id], key_prefix='les_')

        lesion_data_dict = {
            'subject': subj_name,
            'lesion_id': les_id,
            'lesion_volume': lesion_volume,
            'sc_volume': sc_volume,
            **sc_stats,
            **sc_no_les_stats,
            **lesion_stats
        }

        for dilation_factor in [3, 5, 7]:
            # Get the surrounding region of this lesion, and mask by the spinal cord segmentation
            surround = (dilated_segs[dilation_factor] == les_id) & (seg_arr == 0) & (sc_arr == 1)
            if surround.max() == 0:
                logging.warning(f"No valid surrounding region for lesion {les_id} in {subj_name} with "
                                f"dilation factor {dilation_factor}.")
                continue
            lesion_data_dict.update(summary_stats(anat_arr[surround], key_prefix=f'surround_{dilation_factor}_'))

        data.append(lesion_data_dict)
    return data


def compute_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes contrast and signal-to-noise metrics for lesions from existing statistics in a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with lesion and spinal cord statistics.
    Returns:
        pd.DataFrame: Updated DataFrame with additional computed metrics.
    """
    df['snr'] = df['les_mean'] / df['sc_std']
    df['snr_no_les'] = df['les_mean'] / df['sc_no_les_std']
    df['les_cord_contrast'] = (df['les_mean'] - df['sc_mean']) / df['sc_mean']
    df['les_cord-no-les_contrast'] = (df['les_mean'] - df['sc_no_les_mean']) / df['sc_no_les_mean']
    for dilation_factor in [3, 5, 7]:
        df[f'surround_{dilation_factor}_contrast'] = (
                (df['les_mean'] - df[f'surround_{dilation_factor}_mean']) / df[f'surround_{dilation_factor}_mean'])
    return df


def save_contrast_summary(df, metadata_path, out_path):
    """
    Summarise the lesion contrast stats for the different training sets.
    Args:
        df (pd.DataFrame): DataFrame containing the lesion contrast stats for each subject. Should contain the columns:
                           'subject', 'surround_3_contrast', 'surround_5_contrast', 'surround_7_contrast'.
        metadata_path (str): Path to the metadata file. Should contain the columns:
                            'id' (matching 'subject' in df), 'fold', 'incl_exp_17', 'incl_exp_36', 'incl_exp_72',
                            'incl_exp_145' -> these are binary flags for whether an image is included at that scale.
        out_path (str): Path to save the contrast summary file.
    """
    metadata_df = pd.read_csv(metadata_path, usecols=['id', 'fold', 'incl_exp_17', 'incl_exp_36',
                                                      'incl_exp_72', 'incl_exp_145'],
                              dtype={'id': int, 'fold': str})
    merged = pd.merge(df, metadata_df, left_on='subject', right_on='id', how='inner')

    def percentile(n):
        def pct(x): return x.quantile(n)

        pct.__name__ = f'pct_{int(n * 100):02d}'
        return pct

    df_list = []
    for size in ['all', '17', '36', '72', '145']:
        for fold in range(1, 6):
            tmp = merged if size == 'all' else merged[merged[f'incl_exp_{size}'] == 1]
            tmp = tmp[tmp['fold'].isin([str(fold), str(fold % 5 + 1), str((fold + 1) % 5 + 1)])]
            cols = ['surround_3_contrast', 'surround_5_contrast', 'surround_7_contrast']
            summary = tmp[cols].agg(['mean', 'std', 'min', 'max', 'median',
                                     percentile(0.05), percentile(0.1), percentile(0.2),
                                     percentile(0.25), percentile(0.75), percentile(0.95)])
            summary['subset'] = size
            summary['first_fold'] = fold
            df_list.append(summary)

    contrast_summary = pd.concat(df_list).reset_index(names=['statistic'])
    contrast_summary.to_csv(out_path, index=False)


def main(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)
    out_file_prefix = f'{args.out_file}_' if args.out_file else ''
    timestamp = setup_logging(args.out_dir)

    total = len(list(args.anat_dir.iterdir())) if not args.subset else len(args.subset)
    data = []

    with tqdm(total=total) as pbar:
        for subj in args.anat_dir.iterdir():
            if args.subset and subj.name not in args.subset:
                continue
            data.extend(process_subject(args, subj, out_file_prefix))
            pbar.update(1)

    df = pd.DataFrame(data)
    df = compute_additional_metrics(df)
    df.to_csv(args.out_dir / f'{args.out_file}_lesion_intensity_stats_{timestamp}.csv', index=False)

    if args.metadata_path:
        save_contrast_summary(df, metadata_path=args.metadata_path, out_path=args.out_dir / 'contrast_summary.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anat_dir', '-ad', type=Path, required=True,
                        help="Directory containing the subject or volume folders. Each folder should contain the anat "
                             "image, and optionally the lesion seg and/or spinal cord seg images.")
    parser.add_argument('--seg_dir', '-sd', type=Path, default=None,
                        help="Optional, if different from the anat_dir. Path to directory containing subject or volume "
                             "folders that contain the seg images.")
    parser.add_argument('--sc_seg_dir', '-ssd', type=Path, default=None,
                        help="Optional, if different from the anat_dir. Path to directory containing subject or volume "
                             "folders that contain the spinal cord seg images.")
    parser.add_argument('--anat_name', '-an', type=str, default='t2.nii.gz', help="Name of the anat image file.")
    parser.add_argument('--seg_name', '-sn', type=str, default='seg.nii.gz', help="Name of the lesion seg image file.")
    parser.add_argument('--sc_seg_name', '-ssn', type=str, default='t2_sc_seg.nii.gz',
                        help="Name of the spinal cord seg image file.")
    parser.add_argument('--resample', '-r', action='store_true',
                        help="Resample the seg images to the anat image space, if they are in different spaces.")
    parser.add_argument('--out_dir', '-o', type=Path, required=True, help="Output directory for the results.")
    parser.add_argument('--out_file', '-of', type=str, default=None, help="Output file prefix for every saved file.")
    parser.add_argument('--subset', '-s', type=str, nargs='+', default=None, help="Subset of subjects to process.")
    parser.add_argument('--save_cc', '-cc', action='store_true', help="Save connected components of the lesion seg.")
    parser.add_argument('--save_dilations', '-ss', action='store_true', help="Save dilated regions of the lesions.")
    parser.add_argument('--metadata_path', '-mp', type=str, default=None,
                        help="Path to the metadata file for the training sets. Used to summarise the lesion contrast "
                             "stats.")
    parser.add_argument('--overwrite', '-ow', action='store_true', help="Overwrite existing files.")

    args = parser.parse_args()

    # Check that the directories exist
    if not args.anat_dir.exists():
        raise FileNotFoundError(f"Invalid --anat_dir. Directory {args.anat_dir} does not exist.")
    if args.seg_dir and not args.seg_dir.exists():
        raise FileNotFoundError(f"Invalid --seg_dir. Directory {args.seg_dir} does not exist.")
    if args.sc_seg_dir and not args.sc_seg_dir.exists():
        raise FileNotFoundError(f"Invalid --sc_seg_dir. Directory {args.sc_seg_dir} does not exist.")

    if args.seg_dir is None:
        args.seg_dir = args.anat_dir
    if args.sc_seg_dir is None:
        args.sc_seg_dir = args.anat_dir

    main(args)
