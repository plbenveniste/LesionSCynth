from pathlib import Path
import numpy as np
import SimpleITK as sitk
from skimage import morphology, filters
import torchio as tio
from torchio.transforms import Pad
from typing import Union, Optional


def read_orient_extract(path: Path) -> tuple[np.ndarray, sitk.Image]:
    """
    Reads a saved image and converts it to LAS+ (RPI-) orientation.
    Args:
        path (Path): Path to the input image file.
    Returns:
        tuple[np.ndarray, sitk.Image]: A tuple containing the image as a numpy array and the reoriented SimpleITK image.
    """
    im_orig = sitk.ReadImage(path)
    im_LAS = sitk.DICOMOrient(im_orig, 'LAS')
    return sitk_to_numpy(im_LAS), im_LAS


def read_extract(path: Path) -> tuple[np.ndarray, sitk.Image]:
    """
    Reads a saved image without changing its orientation.
    Args:
        path (Path): Path to the input image file.
    Returns:
        tuple[np.ndarray, sitk.Image]: A tuple containing the image as a numpy array and the original SimpleITK image.
    """
    im_orig = sitk.ReadImage(path)
    return sitk_to_numpy(im_orig), im_orig


def get_bbox_bounds(im):
    """ Returns the min and max indices of the bounding box of non-zero voxels in im. """
    ax0 = np.any(im, axis=(1, 2))
    ax1 = np.any(im, axis=(0, 2))
    ax2 = np.any(im, axis=(0, 1))
    ax0_min, ax0_max = np.where(ax0)[0][[0, -1]]
    ax1_min, ax1_max = np.where(ax1)[0][[0, -1]]
    ax2_min, ax2_max = np.where(ax2)[0][[0, -1]]
    bounds = [(ax0_min, ax0_max), (ax1_min, ax1_max), (ax2_min, ax2_max)]
    # Convert to int (rather than numpy.int64) to avoid errors when saving to json
    bounds = [(int(min_), int(max_)) for min_, max_ in bounds]
    return bounds


def bbox(ref_im, other_im=None):
    """ Crops ref_im (and other_im) to the bounding box of non-zero voxels of ref_im"""
    (ax0_min, ax0_max), (ax1_min, ax1_max), (ax2_min, ax2_max) = get_bbox_bounds(ref_im)
    if other_im is None:
        return ref_im[ax0_min:ax0_max+1, ax1_min:ax1_max+1, ax2_min:ax2_max+1], None
    elif isinstance(other_im, list):
        return ref_im[ax0_min:ax0_max+1, ax1_min:ax1_max+1, ax2_min:ax2_max+1], [im[ax0_min:ax0_max+1, ax1_min:ax1_max+1, ax2_min:ax2_max+1] for im in other_im]
    else:
        return ref_im[ax0_min:ax0_max+1, ax1_min:ax1_max+1, ax2_min:ax2_max+1], other_im[ax0_min:ax0_max+1, ax1_min:ax1_max+1, ax2_min:ax2_max+1]


def bbox_sitk(im: sitk.Image) -> tuple:
    """ Returns the voxel coordinates of the bounding box of non-zero voxels in im.
    Returns:  (x_min, x_max, y_min, y_max, z_min, z_max)"""
    # Binarise the image. Do not use BinaryThreshold as it won't work for int images -
    #   if the threshold is 1.0e-7 then the result will be all ones.
    im_thr = sitk.Cast(im > 0, sitk.sitkUInt8)
    bb = sitk.LabelStatisticsImageFilter()
    bb.Execute(im_thr, im_thr)  # Execute the label statistics filter on the binarised image
    return bb.GetBoundingBox(1)  # Return the bounding box voxel coordinates as tuple of 6 values


def sitk_to_numpy(sitk_im: sitk.Image) -> np.ndarray:
    """ Convert SimpleITK image to numpy array. SimpleITK and numpy use different axis orders, so we swap them."""
    return np.swapaxes(sitk.GetArrayFromImage(sitk_im), 0, 2)


def new_image_from_ref(new_data, ref_im, numpy_to_sitk=True) -> sitk.Image:
    """ Use new data array but copy spacing, origin and affine from ref_im, with option to swap axes to convert from
    numpy to SimpleITK axis order.
    Args:
        new_data (np.ndarray): numpy array containing image data
        ref_im (sitk.Image): reference image from which to copy spacing, origin and affine
        numpy_to_sitk (bool): if True, swap axes to convert from numpy to SimpleITK axis order
    Returns:
        sitk.Image: new SimpleITK image with the same spacing, origin and affine as ref_im
    """
    # SITK uses different axis order to numpy, so we need to swap axes
    if numpy_to_sitk:
        new_data = np.swapaxes(new_data, 0, 2)
    new_im = sitk.GetImageFromArray(new_data)
    new_im.SetSpacing(ref_im.GetSpacing())
    new_im.SetOrigin(ref_im.GetOrigin())
    new_im.SetDirection(ref_im.GetDirection())
    return new_im


def resample_to_ref(im: sitk.Image, ref_im: sitk.Image, transform=sitk.AffineTransform(3), interpolator=sitk.sitkLinear,
                    dtype: Optional[Union[int, str]] = None) -> sitk.Image:
    """ Resample an image to match the geometry of a reference image.
    Args:
        im (sitk.Image): Image to be resampled.
        ref_im (sitk.Image): Reference image to match geometry.
        transform (sitk.Transform): Transform to apply during resampling. Default is an identity transform.
        interpolator (int): Interpolator type for resampling. Default is linear interpolation.
        dtype (int): Output pixel type. If None, uses the pixel type of the input image.
    """
    if dtype is None:
        dtype = im.GetPixelID()
    return sitk.Resample(im, size=ref_im.GetSize(), transform=transform, interpolator=interpolator,
                         outputOrigin=ref_im.GetOrigin(), outputSpacing=ref_im.GetSpacing(),
                         outputDirection=ref_im.GetDirection(), defaultPixelValue=0, outputPixelType=dtype)


def resample_spacing(im, new_spacing, interpolator=sitk.sitkLinear):
    """ Resamples image to new spacing. """
    old_spacing = im.GetSpacing()
    old_size = im.GetSize()
    new_size = [int(round(size * old_space/new_space)) for size, old_space, new_space in zip(old_size, old_spacing, new_spacing)]
    return sitk.Resample(im, size=new_size, transform=sitk.Transform(), interpolator=interpolator,
                         outputOrigin=im.GetOrigin(), outputSpacing=new_spacing, outputDirection=im.GetDirection(),
                         defaultPixelValue=0, outputPixelType=im.GetPixelID())


def dilate_slicewise(mask_arr: np.ndarray, dilation_element: np.ndarray = morphology.disk(1),
                     slice_axis: int = 0, multi_values=False) -> np.ndarray:
    """ Dilates the blobs in a 3D mask array slicewise along the specified axis.
    Args:
        mask_arr         - 3D numpy array of the mask.
        dilation_element - Element to use for dilation.
        slice_axis       - Axis along which to dilate the blobs.
        multi_values     - If True, the mask is allowed to have several values, which are kept when dilated.
    Returns:
        Array with same shape as mask_arr with the blobs dilated along the specified axis.
    """
    if not multi_values and len(np.unique(mask_arr)) > 2:
        raise ValueError('The mask has more than 2 unique values. If multi_values is False, the mask should be binary.')

    func = morphology.binary_dilation if not multi_values else filters.rank.maximum

    dilated_mask = np.zeros_like(mask_arr)
    for i in range(mask_arr.shape[slice_axis]):
        if slice_axis == 0:
            dilated_mask[i] = func(mask_arr[i], footprint=dilation_element)
        elif slice_axis == 1:
            dilated_mask[:, i] = func(mask_arr[:, i], footprint=dilation_element)
        elif slice_axis == 2:
            dilated_mask[:, :, i] = func(mask_arr[:, :, i], footprint=dilation_element)
    return dilated_mask


def erode_slicewise(mask_arr: np.ndarray, erosion_element: np.ndarray = morphology.disk(1),
                    slice_axis: int = 0, multi_values=False) -> np.ndarray:
    """ Erodes the blobs in a 3D mask array slicewise along the specified axis.
    Args:
        mask_arr         - 3D numpy array of the mask.
        erosion_element  - Element to use for erosion.
        slice_axis       - Axis along which to erode the blobs.
        multi_values     - If True, the mask is allowed to have several values, which are kept when eroded.
    Returns:
        Binary array with the eroded blobs along the specified axis.
    """
    if not multi_values and len(np.unique(mask_arr)) > 2:
        raise ValueError('The mask has more than 2 unique values. If multi_values is False, the mask should be binary.')

    eroded_mask = np.zeros_like(mask_arr)
    for i in range(mask_arr.shape[slice_axis]):
        if slice_axis == 0:
            eroded_mask[i] = morphology.binary_erosion(mask_arr[i], footprint=erosion_element)
        elif slice_axis == 1:
            eroded_mask[:, i] = morphology.binary_erosion(mask_arr[:, i], footprint=erosion_element)
        elif slice_axis == 2:
            eroded_mask[:, :, i] = morphology.binary_erosion(mask_arr[:, :, i], footprint=erosion_element)
    return eroded_mask


def check_matching_geom(im1: sitk.Image, im2: sitk.Image, tol=1e-5):
    """Given two SITK images, check that they have the same geometry: spacing, origin, direction, size"""
    if (np.allclose(im1.GetSpacing(), im2.GetSpacing(), atol=tol) and
            np.allclose(im1.GetOrigin(), im2.GetOrigin(), atol=tol) and
            np.allclose(im1.GetDirection(), im2.GetDirection(), atol=tol)):
        return True
    else:
        return False


class PadToTargetShape(tio.CropOrPad):
    """Pad, if necessary, to match a target shape."""
    def __init__(self, target_shape, **kwargs):
        super().__init__(target_shape=target_shape, **kwargs)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        padding_params, _ = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None:
            pad = Pad(padding_params, **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        return subject