
import argparse
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from skimage.measure import label
from tqdm import tqdm
import subprocess
import pandas as pd
import json
from copy import deepcopy
from scipy.spatial.distance import dice

from ..im_utils import sitk_to_numpy, new_image_from_ref


def get_iou(gt, pred):
    """ Compute the IoU between two binary masks. If both masks are empty, return 1.0."""
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / union if union != 0 else 1.0


def get_dice(gt, pred):
    """ Compute the Dice coefficient between two binary masks. If both masks are empty, return 1.0."""
    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return 1.0
    return 1 - dice(gt.flatten(), pred.flatten())


def generate_ccs(input_dir, output_dir, thresh, agg='max', save_scores=False, gt_dir=None):
    """
    Binarise and generate connected components from pmap inputs.
    Args:
        input_dir: input directory containing the pmap images.
        output_dir: output directory to save the connected components.
        thresh: threshold to binarise the input pmap.
        agg: how to aggregate the softmax scores for each connected component. ('max' or 'mean')
        save_scores: If True, save the aggregated softmax scores for each connected component in a CSV file.
        gt_dir: If provided, the ground truth directory to orient the input images to match the GT orientation.
    Returns:
        None: The connected components are saved in the output directory.
    """
    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. Skipping.")
        return
    print(f"Generating connected components at {output_dir}")

    output_dir.mkdir(exist_ok=True)

    total = len(list(input_dir.iterdir()))
    for impath in tqdm(input_dir.iterdir(), total=total):
        fname = impath.name
        im = sitk.ReadImage(impath)

        if gt_dir:
            # Load the target orientation
            gt_im = sitk.ReadImage(gt_dir / fname)
            target_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(gt_im.GetDirection())
            im = sitk.DICOMOrient(im, target_orientation)

        arr = sitk_to_numpy(im)

        # Binarise and get connected components
        arr_bin = arr > thresh
        cc = label(arr_bin)
        cc_im = new_image_from_ref(cc, ref_im=im)
        sitk.WriteImage(cc_im, output_dir / fname)

        if save_scores:
            # For each connected component get its max and mean softmax score & save as csv
            csv_path = output_dir / fname.replace('.nii.gz', '.csv')

            with open(csv_path, 'w') as file:
                file.write("label,p\n")

            for cc_id in range(1, cc.max() + 1):
                cc_pred = arr[cc == cc_id]
                if agg == 'max':
                    cc_val = cc_pred.max()
                elif agg == 'mean':
                    cc_val = cc_pred.mean()
                else:
                    raise ValueError(f"Unsupported aggregation method: {agg}")

                with open(csv_path, 'a') as file:
                    file.write(f"{cc_id},{cc_val}\n")


def filter_lesions(gt_df, prob_thresh, pred_df=None, pred_iou_thresh=None, gt_iou_thresh=None):
    """ Filter GT and predicted lesions based on probability and IoU thresholds.
    Args:
        gt_df (pd.DataFrame): GT table with one row per lesion, containing at least the following columns:
                                 predicted_proba, IoU
        prob_thresh (float): Probability threshold for predictions, any lesions below this threshold are discarded.
        pred_df (pd.DataFrame): predictions table - required only if pred_iou_thresh is not None:
                                Should contain at least the following column:  predicted_proba
        pred_iou_thresh (float): IoU threshold for predicted lesions.
        gt_iou_thresh (float): IoU threshold for GT lesions.
    Returns:
        gt_df_filtered (pd.DataFrame): Filtered GT table.
        pred_df_filtered (pd.DataFrame): Filtered predictions table.
    """
    # Filter the predictions to those with a minimum IoU, if relevant
    if pred_iou_thresh is None or pred_iou_thresh < 0:
        if pred_df is None:
            raise ValueError("pred_df must be provided if pred_iou_thresh is not None")
        # Filter predicted lesions to those above the evaluation threshold - use pred_df because we want to keep the
        #  predicted lesions that have no overlap with GT
        preds_filtered = pred_df[pred_df['predicted_proba'] >= prob_thresh]
    else:
        # IoU not in preds table, so need to use the GT table - if IoU
        preds_filtered = gt_df[(gt_df['IoU'] > pred_iou_thresh) & (gt_df['predicted_proba'] >= prob_thresh)]

    if gt_iou_thresh is None or gt_iou_thresh < 0:
        # If we don't filter GT by IoU threshold, then we don't care if it was detected or not, and therefore don't care
        #  about the predicted probability -> just return the full GT table
        gt_df_filtered = gt_df
    else:
        # Otherwise, only take GT lesions detected by a prediction above the probability threshold and IoU threshold
        gt_df_filtered = gt_df[(gt_df['IoU'] > gt_iou_thresh) & (gt_df['predicted_proba'] >= prob_thresh)]

    return gt_df_filtered, preds_filtered


def add_overlap_metrics(results, gt_df, pred_df, cc_thresh):
    """Add metrics for overlap (IoU and Dice) between GT and predicted lesions to the results dictionary.
    Args:
        results (dict): Dictionary containing the results of the FROC analysis.
        gt_df (pd.DataFrame): GT table containing at least the following columns:
                    image_name, reference_instance_id, predicted_proba, is_true_positive
        pred_df (pd.DataFrame): predictions table containing at least the following columns:
                    image_name, predicted_instance_id, predicted_proba, is_false_positive
        cc_thresh (float): Threshold used to binarise preds before computing connected components.
    Returns:
        results (dict): Updated results dictionary with overlap metrics.
        scores_df (pd.DataFrame): DataFrame containing the overlap metrics for each image.
    """
    new_results = deepcopy(results)
    scores_list = []

    for k, v in results.items():
        if isinstance(v, dict) and 'threshold' in v.keys():
            prob_thresh = v['threshold']
            for iou_thresh in [None, 0.0]:
                tag = '' if iou_thresh is None else f'_iou-gt-{iou_thresh}'
                gt_df_filtered, pred_df_filtered = filter_lesions(gt_df, prob_thresh, pred_df, iou_thresh, iou_thresh)
                scores = []
                # Loop over each of the CC pairs and calculate the IoU and Dice
                for im_id in tqdm(gt_df_filtered['image_name'].unique()):
                    gt_ids = gt_df_filtered[gt_df_filtered['image_name'] == im_id].reference_instance_id.unique()
                    pred_ids = pred_df_filtered[pred_df_filtered['image_name'] == im_id].predicted_instance_id.unique()

                    # Load the GT and predicted CCs - Nibabel 3x faster than SimpleITK here
                    gt_cc = nib.load(args.eval_dir / 'gt_ccs' / f'{im_id}.nii.gz').get_fdata()
                    pred_cc = nib.load(args.eval_dir / f'ccs_{cc_thresh}_max' / f'{im_id}.nii.gz').get_fdata()
                    # gt_cc = sitk_to_numpy(sitk.ReadImage(args.eval_dir / 'gt_ccs' / f'{im_id}.nii.gz'))
                    # pred_cc = sitk_to_numpy(sitk.ReadImage(args.eval_dir / f'ccs_{cc_thresh}_max' / f'{im_id}.nii.gz'))

                    # Take only the relevant ids in the gt and pred connected components
                    gt_bin = np.isin(gt_cc, gt_ids).astype(np.uint8)
                    pred_bin = np.isin(pred_cc, pred_ids).astype(np.uint8)

                    iou_val = get_iou(gt_bin, pred_bin)
                    dice_coef = get_dice(gt_bin, pred_bin)

                    # If no GT lesions, there is still a row in table with ref ID NaN
                    n_gt_lesions = 0 if len(gt_ids) == 1 and np.isnan(gt_ids[0]) else len(gt_ids)
                    n_pred_lesions = len(pred_ids)

                    # Calculate IoU and Dice
                    scores.append([im_id, k, prob_thresh, iou_thresh, cc_thresh, n_gt_lesions, n_pred_lesions,
                                   iou_val, dice_coef])

                scores_df = pd.DataFrame(scores, columns=[
                    'im_id',  'eval_name', 'prob_thresh', 'iou_thresh', 'cc_thresh', 'n_gt_lesions', 'n_pred_lesions',
                    'iou', 'dice'])
                scores_list.append(scores_df)
                mean_iou, mean_dice = scores_df[scores_df['n_gt_lesions'] > 0][['iou', 'dice']].mean()
                new_results[k][f'iou{tag}'] = mean_iou
                new_results[k][f'dice{tag}'] = mean_dice

    scores_df = pd.concat(scores_list, ignore_index=True)
    return new_results, scores_df


def main(args):
    results_dir = args.eval_dir / args.results_subdir
    # Generate GT connected components
    generate_ccs(args.eval_dir / 'gt', args.eval_dir / 'gt_ccs', 0.5, 'max', save_scores=False)
    for thresh in args.initial_thresholds:
        for agg in ['max']:
            output_dir = args.eval_dir / f'ccs_{thresh}_{agg}'
            generate_ccs(args.eval_dir / 'pred', output_dir, thresh, agg, save_scores=True, gt_dir=args.eval_dir / 'gt')

            results_subdir = results_dir / f'{thresh}_{agg}'
            script_args = ['python', args.froc_script_path, '-g', args.eval_dir / 'gt_ccs', '-p', output_dir,
                           '-c', output_dir, '-o', results_subdir, '-t', str(args.iou_thresh), '--debug']
            if (results_subdir / 'results.json').exists() and not args.overwrite:
                if (results_subdir / 'overlap_scores.csv').exists():
                    print(f'results.json and overlap_scores.csv already exist for {thresh}_{agg}. Skipping.')
                    continue
            else:
                print(f'Running FROC calculation for {thresh}_{agg}')
                subprocess.run(script_args)

            # Remove unnecessary files but keep final image_level files (for stat tests later): image_level_mean...tsv
            for f in results_subdir.glob('image_level_*_iteration_*.tsv'):
                if f.name == 'image_level_0.5_iteration_0.tsv':
                    continue
                f.unlink()

            # Calculate voxel-wise overlaps (IoU and Dice)
            if args.get_overlaps:
                # Calculate area under PR curve and average precision
                gt_df = pd.read_csv(results_subdir / 'table_gt.tsv', sep='\t')
                pred_df = pd.read_csv(results_subdir / 'table_pred.tsv', sep='\t')

                with open(results_subdir / 'results.json', 'r') as f:
                    results = json.load(f)
                print(f'Calculating overlap metrics for {thresh}_{agg}')
                results, scores_df = add_overlap_metrics(results, gt_df, pred_df, thresh)
                scores_df.to_csv(results_subdir / 'overlap_scores.csv', index=False)

                with open(results_subdir / 'results.json', 'w') as f:
                    json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', '-e', type=Path, required=True,
                        help='Path to the evaluation directory containing "gt" and "pred" subdirectories.')
    parser.add_argument('--results_subdir', '-r', type=str, default='results_froc',
                        help='Subdirectory to store FROC results in the evaluation directory.')
    parser.add_argument('--froc_script_path', '-s', type=Path, required=True,
                        help='Path to the FROC script from MS-Multi-Spine Challenge (msmultispineevaluation/main.py)')
    parser.add_argument('--iou_thresh', '-t', type=float, default=0.2, help='IoU threshold to determine a TP or FP')
    parser.add_argument('--initial_thresholds', '-it', type=float, nargs='+', default=[0.5, 0.01],
                        help='Initial thresholds for binarisation & connected components generation.')
    parser.add_argument('--get_overlaps', action='store_true', help='If flag is set, calculate IoU and Dice at each '
                                                                    'threshold used in the FROC evaluation. Increases '
                                                                    'processing time significantly.')
    parser.add_argument('--overwrite', action='store_true',
                        help='If set, overwrite existing FROC results if they exist.')
    args = parser.parse_args()

    if not (args.eval_dir / 'gt').exists():
        raise FileNotFoundError(f"Could not find 'gt' directory in {args.eval_dir}")
    if not (args.eval_dir / 'pred').exists():
        raise FileNotFoundError(f"Could not find 'pred' directory in {args.eval_dir}")

    main(args)
