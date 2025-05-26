from typing import Tuple, Dict
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import center_of_mass, uniform_filter1d
from scipy.spatial.distance import dice
import subprocess
import logging
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
import math
import json

from ..im_utils import read_extract, resample_spacing, new_image_from_ref, sitk_to_numpy, get_bbox_bounds, \
    bbox, bbox_sitk, resample_to_ref


def largest_cc_mask(mask: np.ndarray) -> np.ndarray:
    """Get the largest connected component of a binary mask.
    Args:
        mask: 3D numpy array of the binary mask.
    Returns:
        mask_largest: 3D numpy array of the largest connected component.
    """
    cc, num_cc = label(mask, return_num=True)
    if num_cc > 1:
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        largest_cc = np.argmax(sizes)
        mask_largest = (cc == largest_cc).astype(np.uint8)
    else:
        mask_largest = mask

    return mask_largest


def close_and_open(mask: np.ndarray) -> np.ndarray:
    """Apply binary closing and then slice-wise opening to a binary mask.
    Args:
        mask: 3D numpy array of the binary mask.
    Returns:
        mask_processed: 3D numpy array of the processed mask.
    """
    mask_closed = binary_closing(mask, footprint=np.ones((3, 3, 3)))
    mask_processed = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        mask_processed[i] = binary_opening(mask_closed[i], footprint=np.ones((3, 3)))

    return mask_processed


def centre_of_mass_per_slice(input_arr: np.ndarray, extrapolate=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Given a numpy array of binary data, find the centre of mass for each slice along z-axis,
    and interpolate if any slice has no non-zero value.
    If extrapolate=False, then the first/last number of entries in each returned vector may be nan
    Args:
        input_arr: 3D numpy array of binary data.
        extrapolate: If True, will extrapolate the centre of mass data along the S-I axis.
    Returns:
        x_ctr: 1D numpy array of the x-coordinate of the centre of mass for each axial slice.
        y_ctr: 1D numpy array of the y-coordinate of the centre of mass for each axial slice.
        z_ctr: 1D numpy array of the z-coordinates corresponding to the axial slices.
    """
    # Get a unique label per axial slice, so we can find the centre of mass for each slice at once
    # Multiply the binary ctr_data with the slice number, creating a unique label for each slice
    z_index_1 = np.arange(1, input_arr.shape[2]+1)  # Get index along z-axis (starting at 1)
    # The ctr or sc_seg is given a unique label for each slice
    labelled_ctr = input_arr * z_index_1.reshape((1, 1, input_arr.shape[2]))
    # Find the centre of mass for all slices at once (getting CoM for each "label", i.e. each slice, separately)
    CoMs = center_of_mass(input_arr, labels=labelled_ctr, index=z_index_1)
    # Extract the axis-specific indexes of the centre of mass in each axial slice
    x_ctr, y_ctr, z_ctr = np.array(CoMs).T  # vectors with length input_arr.shape[2]

    # If any of the slices have no centre of mass (nan) because of missing centreline, interpolate
    left = None if extrapolate else np.nan
    right = None if extrapolate else np.nan
    x_ctr[np.isnan(x_ctr)] = np.interp(np.flatnonzero(np.isnan(x_ctr)),
                                       np.flatnonzero(~np.isnan(x_ctr)),
                                       x_ctr[~np.isnan(x_ctr)],
                                       left=left, right=right)
    y_ctr[np.isnan(y_ctr)] = np.interp(np.flatnonzero(np.isnan(y_ctr)),
                                       np.flatnonzero(~np.isnan(y_ctr)),
                                       y_ctr[~np.isnan(y_ctr)],
                                       left=left, right=right)
    z_ctr[np.isnan(z_ctr)] = np.arange(input_arr.shape[2])[np.isnan(z_ctr)]

    return x_ctr, y_ctr, z_ctr


def extrapolate_ctr_data(ctr_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         n_slices_up: int = 0, n_slices_down: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Extrapolate the centre of mass data along the S-I axis for a certain number of axial slices.
    In effect, just repeats the last or first value for the number of slices to extrapolate.
    The shape of the returned arrays is the same as the input arrays (possibly with some nans remaining).
    Args:
        ctr_data: Tuple of numpy arrays of x, y, z coordinates of centre of mass for each axial slice.
        n_slices_up: Number of slices to extrapolate upwards.
        n_slices_down: Number of slices to extrapolate downwards.
    Returns:
        ctr_data_extrap: Tuple of numpy arrays of extrapolated x, y, z coordinates of centre of mass.
    """
    x_ctr, y_ctr, z_ctr = ctr_data
    x_ctr_extrap = x_ctr.copy()
    y_ctr_extrap = y_ctr.copy()
    z_ctr_extrap = z_ctr.copy()
    # Get the first and last non-nan values
    non_nans = np.argwhere(~np.isnan(x_ctr))[0]
    if non_nans[0] == 0:
        n_slices_down = 0
    elif non_nans[0] < n_slices_down:
        n_slices_down = non_nans[0]
        start_down = 0
    else:
        start_down = non_nans[0] - n_slices_down
    if non_nans[-1] == len(z_ctr) - 1:
        n_slices_up = 0
    elif len(z_ctr) - non_nans[-1] - 1 < n_slices_up:
        n_slices_up = len(z_ctr) - non_nans[-1] - 1
        end_up = len(z_ctr)
    else:
        end_up = non_nans[-1] + n_slices_up

    if n_slices_down:
        x_ctr_extrap[start_down:non_nans[0]] = x_ctr[non_nans[0]]
        y_ctr_extrap[start_down:non_nans[0]] = y_ctr[non_nans[0]]
        z_ctr_extrap[start_down:non_nans[0]] = np.arange(start_down, non_nans[0])
    if n_slices_up:
        x_ctr_extrap[non_nans[-1]+1:end_up] = x_ctr[non_nans[-1]]
        y_ctr_extrap[non_nans[-1]+1:end_up] = y_ctr[non_nans[-1]]
        z_ctr_extrap[non_nans[-1]+1:end_up] = np.arange(non_nans[-1]+1, end_up)

    return x_ctr_extrap, y_ctr_extrap, z_ctr_extrap


def get_sc_seg(image_path: Path, output_path: Path, contrast: str, upper_or_lower: str) -> None:
    """Get the spinal cord segmentation from the input image.
    Args:
        image_path: Path to the image file.
        output_path: Path to save the output segmentation (including file name).
        upper_or_lower: either 'upper' or 'lower', to indicate whether the acquisition is an upper or lower one.
    Returns:
        sc_seg: 3D numpy array of the spinal cord segmentation.
    """
    # Determine which model to use based on the inputs
    task, c = None, None
    if contrast == 'T2starw':
        task = 'seg_sc_t2star'
    elif contrast == 'MP2RAGE':
        task = 'seg_ms_sc_mp2rage'
    elif contrast in ['STIR', 'PSIR']:
        task = 'seg_sc_ms_lesion_stir_psir'
        c = contrast.lower()
    elif contrast == 'T2w':
        c = 't2'
        if upper_or_lower == 'lower':
            task = 'seg_lumbar_sc_t2w'
    elif contrast == 'T1w':
        c = 't1'
    else:
        task = 'seg_sc_contrast_agnostic'
        c = 't2'

    if task is not None:
        call_args = ['sct_deepseg', '-i', str(image_path), '-c', c, '-o', str(output_path), '-task', task, '-v', '0']
    else:
        call_args = ['sct_deepseg_sc', '-i', str(image_path), '-c', c, '-o', str(output_path), '-v', '0', '-thr', '0.1']

    print(call_args)
    subprocess.call(call_args)


def combine_sc_seg(sc_seg_arr_lumbar: np.ndarray, sc_seg_arr_other: np.ndarray) -> np.ndarray:
    """ Combine the seg from the lumbar model with a seg from another model above the lumbar level.
    The lumbar model generally stops well at conus medullaris but fails to segment the lower thoracic cord, whereas
    the T2 model generally has OK results on the thoracic cord but fails to stop at the conus medullaris.

    This will first check that the two segmentations overlap. If not, it raises an error and exit.
    If they overlap, then it will take the lumbar segmentation for the lower part and starting from the midpoint of the
    lumbar seg along S-I axis, it will take union of the two segmentations.

    Args:
        sc_seg_arr_lumbar: 3D numpy array of the binary spinal cord segmentation from the lumbar model.
                            Positive index in third axis should be superior direction.
        sc_seg_arr_other: 3D numpy array of the binary spinal cord segmentation from another model.
    Returns:
        sc_seg_combined: 3D numpy array of the combined spinal cord segmentation.
    """
    # Check that the two segmentations overlap
    if not np.any(sc_seg_arr_lumbar * sc_seg_arr_other):
        raise ValueError('The two segmentations do not overlap.')

    # Find the top of the lumbar segmentation.
    _, _, (z_min_lumbar, z_max_lumbar) = get_bbox_bounds(sc_seg_arr_lumbar)

    z_midpoint = (z_max_lumbar + z_min_lumbar) // 2

    new_seg = deepcopy(sc_seg_arr_lumbar)
    new_seg[..., z_midpoint:] = np.logical_or(sc_seg_arr_lumbar[..., z_midpoint:], sc_seg_arr_other[..., z_midpoint:])

    return new_seg


def extrapolate_sc_seg(sc_seg_arr: np.ndarray, upper_or_lower: str):
    """Extrapolate the spinal cord segmentation along the S-I axis. For upper acquisitions, we extrapolate downwards,
    and for lower acquisitions, we extrapolate upwards.
    Args:
        sc_seg_arr: 3D numpy array of the spinal cord segmentation. In RPI+ orientation (LAS- in SITK).
        upper_or_lower: either 'upper' or 'lower', to indicate whether the acquisition is an upper or lower one.
    Returns:
        sc_seg_extrap: 3D numpy array of the extrapolated spinal cord segmentation, with same shape as input.
    """
    assert upper_or_lower in ['upper', 'lower'], \
        f'upper_or_lower must be either "upper" or "lower", not {upper_or_lower}.'
    shp = sc_seg_arr.shape
    # Get maximum extents of spinal cord seg along S-I axis
    _, _, (z_min, z_max) = get_bbox_bounds(sc_seg_arr)

    if (upper_or_lower == 'upper') and (z_min == 0):
        logging.info('No extrapolation needed for upper acquisition.')
        return sc_seg_arr
    elif (upper_or_lower == 'lower') and (z_max == shp[2] - 1):
        logging.info('No extrapolation needed for lower acquisition.')
        return sc_seg_arr

    # Create the grid points for the RegularGridInterpolator
    x_pts = np.linspace(0, shp[0]-1, shp[0])
    y_pts = np.linspace(0, shp[1]-1, shp[1])
    z_pts_sc = np.linspace(z_min, z_max, z_max-z_min+1)
    pts = (x_pts, y_pts, z_pts_sc)
    # Fit interpolator to the region with a spinal cord segmentation
    interp = RegularGridInterpolator(pts, sc_seg_arr[..., z_min:z_max+1],
                                     bounds_error=False, fill_value=None, method='nearest')

    if upper_or_lower == 'lower':
        new_z_pts = np.linspace(z_max, shp[2]-1, shp[2]-z_max)
    else:
        new_z_pts = np.linspace(0, z_min-1, z_min)

    new_pts = np.meshgrid(x_pts, y_pts, new_z_pts, indexing='ij')
    new_pts = np.stack(new_pts, axis=-1)
    # Interpolate (extrapolate) the spinal cord segmentation to the new points
    new_vals = interp(new_pts)
    # Combine the original seg with the extrapolated seg
    if upper_or_lower == 'lower':
        sc_seg_extrap = np.concatenate((sc_seg_arr[..., :z_max+1], new_vals), axis=-1)
    else:
        sc_seg_extrap = np.concatenate((new_vals, sc_seg_arr[..., z_min:]), axis=-1)

    return sc_seg_extrap


def process_sc_segs(vol_dir: Path, out_dir: Path, metadata_df: pd.DataFrame) -> None:
    """Process the spinal cord segmentations for all volumes in the data directory.
    Args:
        vol_dir: Path to the volume directory.
        out_dir: Path to the specific output directory (e.g. {vol_id} / 'intermediate_files').
        metadata_df: DataFrame containing the metadata for the volumes.
    """
    if not vol_dir.is_dir():
        return

    vol_id = vol_dir.name

    metadata_df.id = metadata_df.id.astype(str)
    metadata = metadata_df[metadata_df['id'] == vol_id]
    if len(metadata) == 0:
        logging.error(f'Metadata not found for {vol_id}')
        raise ValueError(f'Metadata not found for {vol_id}')
    is_lower = metadata['section'].values[0] == 'thor'

    for file in ['t2.nii.gz', 'stir.nii.gz']:
        if (out_dir / file.replace('.nii.gz', '_sc_seg_processed.nii.gz')).exists():
            logging.info(f'Skipping sc_seg generation for {vol_id} {file.replace(".nii.gz", "")} '
                         f'as it already exists.')
            continue

        anat_path = vol_dir / file
        if not anat_path.exists():
            if file == 'stir.nii.gz':
                continue
            else:
                raise FileNotFoundError(f'File {anat_path} not found.')

        # Get the spinal cord segmentation
        get_sc_seg(anat_path, out_dir / file.replace('.nii.gz', '_sc_seg.nii.gz'),
                   contrast='T2w', upper_or_lower='upper')

        sc_seg_im = sitk.ReadImage(out_dir / file.replace('.nii.gz', '_sc_seg.nii.gz'))
        sc_seg_im = sitk.DICOMOrient(sc_seg_im, 'LAS')
        sc_seg_arr = sitk_to_numpy(sc_seg_im)
        # Lumbar seg
        if is_lower:
            get_sc_seg(anat_path, out_dir / file.replace('.nii.gz', '_sc_seg_lumbar.nii.gz'),
                        contrast='T2w', upper_or_lower='lower')

            sc_seg_lumbar = sitk.ReadImage(out_dir / file.replace('.nii.gz', '_sc_seg_lumbar.nii.gz'))
            sc_seg_lumbar = sitk.DICOMOrient(sc_seg_lumbar, 'LAS')
            sc_seg_lumbar_arr = sitk_to_numpy(sc_seg_lumbar)
            try:
                sc_seg_arr = combine_sc_seg(sc_seg_lumbar_arr, sc_seg_arr)
            except ValueError as e:
                raise ValueError(f'Error combining lumbar spinal cord segmentations for {vol_id}: {e}')

        # Apply closing
        sc_seg_arr = binary_closing(sc_seg_arr, footprint=np.ones((3, 3, 3)))

        # Take the largest connected component
        largest_cc = largest_cc_mask(sc_seg_arr)

        # Apply opening slicewise to remove small isolated regions on each slice
        sc_seg_arr = np.zeros(sc_seg_arr.shape, dtype=np.uint8)
        for i in range(sc_seg_arr.shape[0]):
            sc_seg_arr[i] = binary_opening(largest_cc[i], footprint=np.ones((3, 3)))

        if not is_lower:
            # Extrapolate downwards for upper acquisitions
            sc_seg_arr = extrapolate_sc_seg(sc_seg_arr, upper_or_lower='upper')

        # Write to file
        sc_seg_im = new_image_from_ref(sc_seg_arr, sc_seg_im)
        sitk.WriteImage(sc_seg_im, out_dir / file.replace('.nii.gz', '_sc_seg_processed.nii.gz'))

        # Symlink the original images
        if not (out_dir / file).exists():
            (out_dir / file).symlink_to(anat_path)

    if not (out_dir / 'seg.nii.gz').exists():
        (out_dir / 'seg.nii.gz').symlink_to(vol_dir / 'seg.nii.gz')


def crop_around_sc(im: sitk.Image, sc_seg: sitk.Image, crop_size: Tuple[int, int]) -> Tuple[sitk.Image, sitk.Image]:
    """Crop the image around the spinal cord segmentation in the axial plane."""
    shp = im.GetSize()
    # Get the bounding box of the spinal cord segmentation
    bbox_sc = bbox_sitk(sc_seg)
    x_crop_start = max(0, bbox_sc[0] - crop_size[0])
    x_crop_end = min(shp[0], bbox_sc[1] + crop_size[0] + 1)
    y_crop_start = max(0, bbox_sc[2] - crop_size[1])
    y_crop_end = min(shp[1], bbox_sc[3] + crop_size[1] + 1)

    im_cropped = im[x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]
    sc_seg_cropped = sc_seg[x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]
    return im_cropped, sc_seg_cropped


def register_stir_to_t2(dirpath: Path, t2_im: sitk.Image, stir_im: sitk.Image, t2_sc_seg: sitk.Image,
                        stir_sc_seg: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
    """Register the STIR image to the T2 image. Then warp the spinal cord segmentation using this transform, and
    check the dice overlap between the two spinal cord segmentations. A new sub directory will be created within this
    called reg_results.
    Args:
        dirpath: Path to the directory to contain the registration results and intermediate files.
        t2_im: sitk.Image of the T2 image.
        stir_im: sitk.Image of the STIR image.
        t2_sc_seg: sitk.Image of the spinal cord segmentation from the T2 image.
        stir_sc_seg: sitk.Image of the spinal cord segmentation from the STIR image.
    Returns:
        stir_im_reg: sitk.Image of the registered STIR image.
        stir_sc_seg_reg: sitk.Image of the registered spinal cord segmentation from the STIR image.
    """
    sitk.WriteImage(t2_im, dirpath / 't2_sc_cropped.nii.gz')
    sitk.WriteImage(stir_im, dirpath / 'stir_sc_cropped.nii.gz')
    sitk.WriteImage(t2_sc_seg, dirpath / 't2_sc_seg_sc_cropped.nii.gz')
    sitk.WriteImage(stir_sc_seg, dirpath / 'stir_sc_seg_sc_cropped.nii.gz')

    # 7. Register STIR to T2.
    if not (dirpath / 'stir_reg.nii.gz').exists():
        call_args = ['sct_register_multimodal', '-i', str(dirpath / 'stir_sc_cropped.nii.gz'),
                     '-d', str(dirpath / 't2_sc_cropped.nii.gz'), '-param', 'step=1,type=im,algo=dl',
                     '-o', str(dirpath / 'stir_sc_cropped_reg.nii.gz'), '-v', '0',
                     '-ofolder', str(dirpath / 'reg_results')]

        with open(args.output_dir / 'registration.log', 'w') as f:
            subprocess.run(call_args, stdout=f, stderr=f)

        # Symlink to the main processing directory
        (dirpath / 'stir_reg.nii.gz').symlink_to('reg_results/stir_sc_cropped_reg.nii.gz')
    else:
        logging.info('Registration already performed, skipping.')

    # 7.a Warp the SC seg to the T2 image
    call_args = ['sct_apply_transfo', '-x', 'nn', '-i', str(dirpath / 'stir_sc_seg_sc_cropped.nii.gz'),
                 '-d', str(dirpath / 't2_sc_cropped.nii.gz'),
                 '-w', str(dirpath / 'reg_results' / 'warp_stir_sc_cropped2t2_sc_cropped.nii.gz'),
                 '-o', str(dirpath / 'stir_sc_seg_reg.nii.gz'), '-v', '0']

    with open(args.output_dir / 'registration.log', 'a') as f:
        subprocess.run(call_args, stdout=f, stderr=f)

    # Read the transformed images
    stir_im_reg = sitk.ReadImage(dirpath / 'stir_reg.nii.gz')
    stir_sc_seg_reg = sitk.ReadImage(dirpath / 'stir_sc_seg_reg.nii.gz')

    # Get Dice between the two SC segmentations
    stir_sc_seg_reg_arr = sitk_to_numpy(stir_sc_seg_reg)
    t2_sc_seg_arr = sitk_to_numpy(t2_sc_seg)
    stir_sc_seg_orig_arr = sitk_to_numpy(stir_sc_seg)

    dice_before = 1 - dice(stir_sc_seg_orig_arr.flatten(), t2_sc_seg_arr.flatten())
    dice_after = 1 - dice(stir_sc_seg_reg_arr.flatten(), t2_sc_seg_arr.flatten())

    # Write to file inside registration directory
    with open(dirpath / 'reg_results' / 'dice_scores.txt', 'w') as f:
        f.write(f'Dice before registration: {round(dice_before, 4)}\n')
        f.write(f'Dice after registration: {round(dice_after, 4)}\n')

    return stir_im_reg, stir_sc_seg_reg


def _find_crop_start_end(coord_ctr: np.ndarray, crop_size: int, im_dim: int, shift=True):
    """Adapted from SCT. Util function to find the coordinates to crop the image around the centerline (coord_ctr).
    Args: coord_ctr: Array with single coordinates per slice, representing the centerline of the spinal cord in a particular axis.
                     Length should be equal to the number of axial slices in the image.
                     We operate on a single axis at a time, e.g. Right-Left axis OR Anterior-Posterior axis.
          crop_size: The size of the crop window in the relevant axis.
          im_dim: The size of the image in the relevant axis.
          shift: Whether to shift the crop window so that the central voxel is the centerline voxel.
    """
    if coord_ctr.dtype != int:
        coord_ctr = coord_ctr.astype(int)

    half_size = crop_size // 2
    if shift:
        # For even crop sizes, there will be half_size-1 voxels on one side of the centerline and half_size on the other.
        # For odd crop sizes, there will be half_size voxels on each side of the centerline.
        coord_start = coord_ctr - half_size + int(crop_size % 2 == 0)
        # (end index will not be included in numpy slice, so incremented by +1 here)
        coord_end = coord_ctr + half_size + 1

        # If we met the limit of the image at the start or end, shift the crop window to the other side if possible
        if im_dim >= crop_size:
            coord_start[coord_end > im_dim] = im_dim - crop_size
            coord_end[coord_start < 0] = crop_size
        else:
            # Otherwise, just set the crop window to the whole axis
            coord_start[coord_end > im_dim] = 0
            coord_end[coord_start < 0] = im_dim
        # Limit the crop to the edges of the image
        coord_end[coord_end > im_dim] = im_dim
        coord_start[coord_start < 0] = 0
    else:
        # No shift -> we want to take a bounding box around the centerline with size crop_size, but without shifting the window so that the centreline is in the centre
        # Find maximum extent of centreline
        ctr_start, ctr_end = np.min(coord_ctr), np.max(coord_ctr)
        n_ctr_slices = ctr_end - ctr_start + 1

        # For even crop & even ctr slices, or odd crop & odd ctr slices, we can take the same number of slices on each side of the centreline
        if (n_ctr_slices % 2 == 0) == (crop_size % 2 == 0):
            # Take the same number of slices on each side of the centreline (end index will not be included in numpy slice, so incremented by +1 here)
            coord_start, coord_end = ctr_start - half_size + n_ctr_slices//2, ctr_end + half_size - n_ctr_slices//2 + 1
        else:
            # Otherwise, we must choose whether to take more slices on left or right - based on which side has more centreline across the whole image
            _, counts = np.unique(coord_ctr, return_counts=True)
            more_slices_left = np.sum(counts[:len(counts)//2]) > np.sum(counts[math.ceil(len(counts)/2):])

            coord_start = ctr_start - half_size + n_ctr_slices//2 + int(not more_slices_left)
            coord_end = ctr_end + half_size - n_ctr_slices//2 - int(more_slices_left) + 1

        # Limit the crop to the edges of the image
        coord_start = max(coord_start, 0)
        coord_end = min(coord_end, im_dim)

        # Convert the int coordinates to one coordinate per slice for consistency with the shift==True case
        coord_start, coord_end = np.repeat(coord_start, len(coord_ctr)), np.repeat(coord_end, len(coord_ctr))

    return coord_start, coord_end


def crop_image_around_centerline(im_data: np.ndarray, ctr_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 crop_size: list[int, int], aniso_sag=False, keep_full=False):
    """Crop the input image around the input centerline.
    Args:
        im_data: 3D numpy array of the image to be cropped (in RPI+ orientation).
        ctr_data: tuple of 3 numpy arrays of the centerline coords to be used for cropping.
        crop_size: Tuple (R-L axis, A-P axis) of crop size in voxels,
                        so for anisotropic sagittal image, might want (10, 48) for example.
        aniso_sag: Boolean, whether the image is anisotropic or not. If True, the SC won't be centred in the R-L axis.
        keep_full: Boolean, whether to keep the full image size in the S-I axis or only where there is valid ctr data.
    Returns:
        cropped_data: 3D numpy array of the cropped image, of shape (crop_size[0], crop_size[1], z)
                        where z=im_data.shape[2] if keep_full=True, otherwise z=number of axial slices with ctr data.
        [x_lst, y_lst, z_lst]: Lists of the starting coordinates of the cropped image in the original image space.
    """
    im_data = im_data.astype(np.float32)
    # Extract the centrline coordinates from the input tuple
    x_ctr, y_ctr, z_ctr = ctr_data
    # Get the maximum extents of the centreline along the S-I axis
    ctr_start_z, ctr_end_z = int(z_ctr[0]), int(z_ctr[-1])

    if keep_full:
        cropped_data = np.zeros((crop_size[0], crop_size[1], im_data.shape[2]), dtype=np.float32)
    else:
        cropped_data = np.zeros((crop_size[0], crop_size[1], ctr_end_z - ctr_start_z + 1), dtype=np.float32)

    x_start_all, x_end_all = _find_crop_start_end(x_ctr, crop_size[0], im_data.shape[0], shift=not aniso_sag)
    y_start_all, y_end_all = _find_crop_start_end(y_ctr, crop_size[1], im_data.shape[1])

    # Loop over the axial slices and crop each slice around the centreline
    for i, zz in enumerate(range(ctr_start_z, ctr_end_z+1)):
        x_start, x_end = int(x_start_all[z_ctr == zz].item()), int(x_end_all[z_ctr == zz].item())
        y_start, y_end = int(y_start_all[z_ctr == zz].item()), int(y_end_all[z_ctr == zz].item())

        x_shape, y_shape = im_data[x_start:x_end, y_start:y_end, zz].shape
        cropped_data[:x_shape, :y_shape, i] = im_data[x_start:x_end, y_start:y_end, zz]

    x_lst = x_start_all.reshape(-1).tolist()
    y_lst = y_start_all.reshape(-1).tolist()
    z_lst = np.arange(ctr_start_z, ctr_end_z+1).tolist()

    return cropped_data, [x_lst, y_lst, z_lst]


def process_single_volume(volume_id: str, metadata_df: pd.DataFrame, args: Dict) -> Dict:
    """ Processes a single volume, i.e. a single folder containing t2.nii.gz, seg.nii.gz and maybe stir.nii.gz
    1. Get the spinal cord segmentations from T2 and STIR, if available
    2. Convert to LAS orientation
    3. Resample T2 to 0.5mm isotropic resolution
       If STIR available:
           4. Crop T2 around SC mask.
           5. Resample STIR to T2 space.
           6. Save cropped T2 and STIR to file (for SCT registration function).
           7. Register STIR to T2.
    8. Extract the centreline/centre of mass from the T2 SC mask.
    9. Crop and shift all images around the centreline.

    Args:
        volume_id: ID of the volume to process.
        metadata_df: DataFrame containing the metadata for the volumes, including the section (upper/lower).
        args: dict containing the arguments from argparse.
    Returns:
        Dictionary with 'bbox_bounds' -> bounding box used to crop the image, if applicable;
                        'coords_uncrop' -> coordinates used to crop each axial slice.
    """
    orig_vol_dir = args.data_dir / volume_id
    stir_path = orig_vol_dir / 'stir.nii.gz'
    seg_path = orig_vol_dir / 'seg.nii.gz'
    intermediate_dir = args.output_dir / volume_id / 'intermediate_files'
    intermediate_dir.mkdir(exist_ok=True, parents=True)

    # 1. Get the spinal cord segmentations from T2 and STIR, if available
    try:
        process_sc_segs(orig_vol_dir, intermediate_dir, metadata_df)
    except Exception as e:
        raise ValueError(f'Error processing spinal cord segmentations for {volume_id}: {e}')

    # 2. Convert to LAS orientation
    t2_im = sitk.ReadImage(orig_vol_dir / 't2.nii.gz')
    t2_im = sitk.DICOMOrient(t2_im, 'LAS')

    # 3. Resample T2 and SC seg to new spacing
    t2_resampled = resample_spacing(t2_im, args.spacing)
    t2_sc_seg_im = sitk.ReadImage(intermediate_dir / 't2_sc_seg_processed.nii.gz')
    t2_sc_seg_im = resample_to_ref(t2_sc_seg_im, t2_resampled, interpolator=sitk.sitkNearestNeighbor)

    # Check if STIR exists
    process_stir = False
    if stir_path.exists():
        if not (intermediate_dir / 'stir_sc_seg_processed.nii.gz').exists():
            logging.warning(f'STIR file exists but SC seg not found for {volume_id}. Skipping STIR processing.')
        else:
            process_stir = True

    if process_stir:
        # 4. Crop T2 around SC mask (rectangular bounding box + 24 voxels each side)
        t2_resampled, t2_sc_seg_im = crop_around_sc(t2_resampled, t2_sc_seg_im, (24, 24))

        # 5. Resample STIR to T2 space.
        stir_im = sitk.ReadImage(stir_path)
        stir_im = resample_to_ref(stir_im, t2_resampled)
        stir_sc_seg_im = sitk.ReadImage(intermediate_dir / 'stir_sc_seg_processed.nii.gz')
        stir_sc_seg_im = resample_to_ref(stir_sc_seg_im, t2_resampled, interpolator=sitk.sitkNearestNeighbor)

        # 6. Save cropped T2 and STIR to file (for SCT registration function).
        stir_reg, stir_sc_seg_reg = register_stir_to_t2(intermediate_dir, t2_resampled, stir_im, t2_sc_seg_im, stir_sc_seg_im)
        stir_reg_arr = sitk_to_numpy(stir_reg)
        stir_sc_seg_arr = sitk_to_numpy(stir_sc_seg_reg)

    # Extract the numpy arrays
    t2_resampled_arr = sitk_to_numpy(t2_resampled)
    t2_sc_seg_arr = sitk_to_numpy(t2_sc_seg_im)

    # 9. Resample lesion seg if it exists. --------------------------------------------------------------
    if process_stir:
        seg_im = sitk.ReadImage(seg_path)
        seg_im = resample_to_ref(seg_im, t2_resampled, interpolator=sitk.sitkNearestNeighbor)
        seg_arr = sitk_to_numpy(seg_im)
        seg_arr = (seg_arr > 0).astype(np.uint8)
    else:
        seg_arr = np.zeros(t2_sc_seg_arr.shape, dtype=np.uint8)  # Dummy array.

    # 10. Crop to bounding box of STIR, if it exists. (STIR already resampled to T2, so is cropped to T2 bbox) ------
    if process_stir:
        bbox_bounds = get_bbox_bounds(stir_reg_arr)
        stir_reg_arr, [t2_resampled_arr, seg_arr, t2_sc_seg_arr, stir_sc_seg_arr] = \
            bbox(stir_reg_arr, [t2_resampled_arr, seg_arr, t2_sc_seg_arr, stir_sc_seg_arr])
    else:
        bbox_bounds = []

    # 11. Extract the centreline/centre of mass from the T2 spinal cord segmentation -------------------------------
    ctr_data = centre_of_mass_per_slice(t2_sc_seg_arr)
    # Extrapolate the centre of mass data along the S-I axis for a certain number of axial slices.
    ctr_data = extrapolate_ctr_data(ctr_data, n_slices_up=10, n_slices_down=10)
    # Remove remaining slices with no centre of mass data
    ctr_data = (ctr_data[0][~np.isnan(ctr_data[0])],
                ctr_data[1][~np.isnan(ctr_data[0])],
                ctr_data[2][~np.isnan(ctr_data[0])])
    # Apply smoothing
    ctr_data = (uniform_filter1d(ctr_data[0], size=9), uniform_filter1d(ctr_data[1], size=9), ctr_data[2])

    # 12. Crop and shift all images around the centreline including SC seg ------------------------------------------

    # T2 anat image
    t2_cropped_arr, coords_uncrop = crop_image_around_centerline(t2_resampled_arr, ctr_data, args.crop_size)
    t2_cropped = new_image_from_ref(t2_cropped_arr, t2_resampled)
    sitk.WriteImage(t2_cropped, intermediate_dir / 't2_cropped.nii.gz')
    # T2 spinal cord seg
    t2_sc_seg_cropped_arr, _ = crop_image_around_centerline(t2_sc_seg_arr, ctr_data, args.crop_size)
    # Apply closing for smoother edges and slice-wise opening to remove small isolated regions on outer slices
    t2_sc_seg_cropped_arr = close_and_open(t2_sc_seg_cropped_arr)
    # Save to file
    t2_sc_seg_cropped = new_image_from_ref(t2_sc_seg_cropped_arr, t2_resampled)
    sitk.WriteImage(t2_sc_seg_cropped, intermediate_dir / 't2_sc_seg_cropped.nii.gz')

    # STIR anat image and spinal cord seg
    if process_stir:
        stir_cropped, _ = crop_image_around_centerline(stir_reg_arr, ctr_data, args.crop_size)
        stir_cropped = new_image_from_ref(stir_cropped, t2_resampled)
        sitk.WriteImage(stir_cropped, intermediate_dir / 'stir_cropped.nii.gz')

        stir_sc_seg_cropped, _ = crop_image_around_centerline(stir_sc_seg_arr, ctr_data, args.crop_size)
        # Apply closing for smoother edges and slice-wise opening to remove small isolated regions on outer slices
        stir_sc_seg_cropped = close_and_open(stir_sc_seg_cropped)
        stir_sc_seg_cropped = new_image_from_ref(stir_sc_seg_cropped, t2_resampled)
        sitk.WriteImage(stir_sc_seg_cropped, intermediate_dir / 'stir_sc_seg_cropped.nii.gz')

    # Lesion seg
    if seg_path.exists():
        seg_cropped, _ = crop_image_around_centerline(seg_arr, ctr_data, args.crop_size)
        seg_cropped = new_image_from_ref(seg_cropped, t2_resampled)
        sitk.WriteImage(seg_cropped, intermediate_dir / 'seg_cropped.nii.gz')

    # Finally, create symlinks in the main directory
    if not (args.output_dir / volume_id / 't2.nii.gz').exists():
        (args.output_dir / volume_id / 't2.nii.gz').symlink_to('intermediate_files/t2_cropped.nii.gz')
    if not (args.output_dir / volume_id / 't2_sc_seg.nii.gz').exists():
        (args.output_dir / volume_id / 't2_sc_seg.nii.gz').symlink_to('intermediate_files/t2_sc_seg_cropped.nii.gz')
    if seg_path.exists() and not (args.output_dir / volume_id / 'seg.nii.gz').exists():
        (args.output_dir / volume_id / 'seg.nii.gz').symlink_to('intermediate_files/seg_cropped.nii.gz')
    if stir_path.exists() and not (args.output_dir / volume_id / 'stir.nii.gz').exists():
        (args.output_dir / volume_id / 'stir.nii.gz').symlink_to('intermediate_files/stir_cropped.nii.gz')
    if stir_path.exists() and not (args.output_dir / volume_id / 'stir_sc_seg.nii.gz').exists():
        (args.output_dir / volume_id / 'stir_sc_seg.nii.gz').symlink_to('intermediate_files/stir_sc_seg_cropped.nii.gz')

    return {'uncrop': coords_uncrop, 'bbox': bbox_bounds}


def process_lesions(vol_id: str, data_dir: Path, out_dirpath: Path) -> None:
    """
    Process, extract (crop) and save lesions from the given volume ID.
    Args:
        vol_id: ID of the volume to process.
        data_dir: Path to the directory containing the volumes.
        out_dirpath: Path to the output directory where the processed lesions should be saved.
    Returns:
        None. Saves the cropped lesions to the output directory as lesion_{lesion_num}.nii.gz and
        corresponding segmentation mask as lesion_{lesion_num}_seg.nii.gz.
    """
    vol_dir = data_dir / vol_id
    if not vol_dir.is_dir():
        return
    outdir = out_dirpath / vol_id
    outdir.mkdir(parents=True, exist_ok=True)

    seg_arr, seg_im = read_extract(vol_dir / 'seg.nii.gz')
    t2_arr, t2_im = read_extract(vol_dir / 't2.nii.gz')

    cc_arr, num_cc = label(seg_arr, return_num=True)

    # Normalise the T2 image
    t2_arr = (t2_arr - t2_arr.mean()) / t2_arr.std()

    for cc in range(1, num_cc+1):
        t2_cc = deepcopy(t2_arr)
        t2_cc[cc_arr != cc] = 0
        # Crop along S-I axis to the maximum extent of the lesion
        x_lst, y_lst, z_lst = np.where(cc_arr == cc)
        z_start, z_end = z_lst.min(), z_lst.max()
        t2_cc = t2_cc[:, :, z_start:z_end+1]

        # Convert to SimpleITK image and save
        t2_cc_im = new_image_from_ref(t2_cc, t2_im)
        sitk.WriteImage(t2_cc_im, outdir / f'lesion_{cc}.nii.gz')

        cc_cropped = (cc_arr == cc)[..., z_start:z_end+1].astype(np.uint8)
        cc_im = new_image_from_ref(cc_cropped, seg_im)
        sitk.WriteImage(cc_im, outdir / f'lesion_{cc}_seg.nii.gz')


def main(args):
    args.output_dir.mkdir(exist_ok=True, parents=True)
    tstamp = datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(filename=args.output_dir / f'preprocess_t2stir_{tstamp}.log', level=logging.INFO)

    metadata_df = pd.read_csv(args.metadata_path, low_memory=False)
    metadata_df.id = metadata_df.id.astype(str)

    vols = [vol_dir.name for vol_dir in args.data_dir.iterdir() if vol_dir.is_dir()]
    if args.subset is not None:
        vols = [vol for vol in vols if vol in args.subset]

    bounds = {}
    for vol in tqdm(vols):
        print(f'Processing volume {vol}')
        if args.no_error:
            try:
                bounds[vol] = process_single_volume(vol, metadata_df, args)
            except Exception as e:
                logging.error(f'Error processing {vol}: {e}')
        else:
            bounds[vol] = process_single_volume(vol, metadata_df, args)

    with open(args.output_dir / f'bounds_{tstamp}.json', 'w') as f:
        json.dump(bounds, f)

    if args.lesion_outdir is not None:
        for vol in tqdm(vols, desc='Processing lesions'):
            if args.no_error:
                try:
                    process_lesions(vol, args.data_dir, args.lesion_outdir)
                except Exception as e:
                    logging.error(f'Error processing lesions for {vol}: {e}')
            else:
                process_lesions(vol, args.output_dir, args.lesion_outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess T2 and STIR images (cropped+shifted)')
    parser.add_argument('-d', '--data_dir', type=Path, required=True,
                        help='Path to the directory containing the volumes to be preprocessed.')
    parser.add_argument('-o', '--output_dir', type=Path, required=True,
                        help='Path to the output directory where the preprocessed data should be saved.')
    parser.add_argument('-m', '--metadata_path', type=Path, required=True,
                         help='Path to the metadata file with one row per case.')
    parser.add_argument('-s', '--spacing', type=float, default=[0.5]*3, nargs='+',
                        help='Desired voxel spacing. Either a single value or three values for R-L, A-P, S-I.')
    parser.add_argument('-c', '--crop_size', type=int, default=[48, 48], nargs=2,
                        help='Size of the crop around the centreline in the R-L and A-P directions.')
    parser.add_argument('-sub', '--subset', type=str, default=None, nargs='+',
                        help='Subset of volumes to process. If None, all volumes will be processed.')
    parser.add_argument('-noerr', '--no_error', action='store_true', help='Ignore errors and continue processing.')
    parser.add_argument('-lesion_out', '--lesion_outdir', type=Path, default=None,
                        help='Path to the output directory for extracted lesions. '
                             'If None, lesions will not be extracted.')

    args = parser.parse_args()
    main(args)


