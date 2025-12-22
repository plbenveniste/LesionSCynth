""""
From my understanding this script extracts the intensity statistics of lesions and spinal cord from MS MRI images.
"""
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
import json
import os
import sys
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from im_utils import (new_image_from_ref, sitk_to_numpy, dilate_slicewise, erode_slicewise,
                        check_matching_geom, resample_to_ref, read_orient_extract)


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


def process_subject(args: argparse.Namespace, subj_dict: dict) -> list[dict]:
    """
    Processes a single subject by loading images, resampling if needed, and extracting lesion statistics.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        subj_dict (dict): Dictionary containing subject information.
    Returns:
        list[dict]: A list of dictionaries containing lesion and spinal cord statistics.
    """
    anat_arr, anat_im = read_orient_extract(subj_dict['scan'])
    seg_arr, seg_im = read_orient_extract(subj_dict['lesion_seg'])
    sc_arr, sc_seg_im = read_orient_extract(subj_dict['sc_seg'])

    # Compute voxel size in mm^3
    voxel_size_mm3 = np.prod(seg_im.GetSpacing())
    # convert to int
    seg_arr = seg_arr.astype(np.uint8)

    return collect_lesion_data(subj_dict['subject_id'], anat_arr, seg_arr, sc_arr, sc_seg_im,
                                seg_im, voxel_size_mm3, args)


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


def collect_lesion_data(subj_name: str, anat_arr: np.ndarray, seg_arr: np.ndarray, sc_arr: np.ndarray,
                        sc_seg_im: sitk.Image, seg_im: sitk.Image, voxel_size_mm3: float,
                        args: argparse.Namespace) -> list[dict]:
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

    # The spinal cord mask sometimes has higher intensities because of a partial volume effect with the CSF.
    # Erode the mask slightly to reduce this effect
    erosion_factor = 1
    sc_arr_eroded = erode_slicewise(sc_arr, skimage.morphology.disk(erosion_factor), slice_axis=0, multi_values=False)

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


def save_contrast_summary(df, out_path):
    """
    Summarise the lesion contrast stats for the different training sets.
    Args:
        df (pd.DataFrame): DataFrame containing the lesion contrast stats for each subject. Should contain the columns:
                           'subject', 'surround_3_contrast', 'surround_5_contrast', 'surround_7_contrast'.
        out_path (str): Path to save the contrast summary file.
    """
    def percentile(n):
        def pct(x): return x.quantile(n)

        pct.__name__ = f'pct_{int(n * 100):02d}'
        return pct

    cols = ['surround_3_contrast', 'surround_5_contrast', 'surround_7_contrast']
    summary = df[cols].agg(['mean', 'std', 'min', 'max', 'median',
                             percentile(0.05), percentile(0.1), percentile(0.2),
                             percentile(0.25), percentile(0.75), percentile(0.95)])

    summary = summary.reset_index(names=['statistic'])
    summary.to_csv(out_path, index=False)


def main(args):
    # Create output directory
    output_path = args.out_dir
    os.makedirs(output_path, exist_ok=True)

    # Open the msd dataset json file to get lists of images
    with open(args.msd, 'r') as f:
        msd_json = json.load(f)
    images = msd_json['data']

    # initialize output data
    data = []

    for image in tqdm(images):
        print(f"Processing subject: {image}")
        data.extend(process_subject(args, images[image]))
    
    # Convert to dataframe and compute additional metrics
    df = pd.DataFrame(data)
    df = compute_additional_metrics(df)

    output_path = os.path.join(args.out_dir, 'lesion_intensity_stats.csv')
    df.to_csv(output_path, index=False)
    save_contrast_summary(df, out_path=os.path.join(args.out_dir, 'contrast_summary.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--msd', type=Path, required=True, help="Path to the msd dataset json file.")
    parser.add_argument('--out-dir', '-o', type=Path, required=True, help="Output directory for the results.")
    args = parser.parse_args()

    main(args)
