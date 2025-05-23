import torchio as tio
import pandas as pd
from scipy.stats import truncnorm

from sslearn.model_and_training.data_module import SyntheticMixedDataModule
from sslearn.model_and_training.model import Model
from sslearn.model_and_training.data_augmentation import OptionalAddLesionContrast
from sslearn.configs.config import Config


class SyntheticMixedConfig(Config):
    model_description = 'Mixed Synthetic Model'
    data_module = SyntheticMixedDataModule
    model_class = Model

    sampler_args = {'balance_attribute': 'label',
                    'num_samples': None,
                    'balance_with': 1}  # Which value of balance_attribute to balance with

    # The following attributes will be initialised in the update_params method
    lesion_augmentation, transform_kwargs = None, None

    def get_truncnorm_params(self):
        # Parameters for truncnorm distribution
        contrast_summary = pd.read_csv(self.contrast_summary_path, dtype={'first_fold': str, 'subset': str})
        contrast_summary = contrast_summary[(contrast_summary['first_fold'] == str(self.first_fold)) &
                                            (contrast_summary['subset'] == self.exp_scale)]

        a = contrast_summary[contrast_summary['statistic'] == 'pct_20']['surround_5_contrast'].values[0]
        loc = contrast_summary[contrast_summary['statistic'] == 'mean']['surround_5_contrast'].values[0]
        scale = contrast_summary[contrast_summary['statistic'] == 'std']['surround_5_contrast'].values[0]
        b = 1.0  # Effectively truncated only at left side
        print(f"Truncnorm parameters: a={a}, loc={loc}, scale={scale}, b={b}")
        a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
        return a_transformed, b_transformed, loc, scale

    def init_lesion_augmentation(self, lesion_paths, a_transformed, b_transformed, loc, scale):
        aug_args = {
            'lesion_paths': lesion_paths,
            'modalities': self.modalities,
            'factor_distribution': truncnorm(a=a_transformed, b=b_transformed, loc=loc, scale=scale),
            'gaussian_spatial': True,
            'min_factor_gaussian': 0.015,
            'min_size': 375,
            'max_size': 3500,
            'blur_radius': 2,
            'blur_sigma': 0.67,
            'other_transforms': tio.RandomAffine(scales=0.1, degrees=(5, 5, 45), center='image', p=0.5),
        }
        lesion_augmentation = OptionalAddLesionContrast(**aug_args)
        return lesion_augmentation, aug_args

    def update_params(self, mode='train', **kwargs):
        super().update_params(mode=mode, **kwargs)
        if mode == 'train':
            # Get the paths to the relevant stored lesion masks
            lesion_paths = []
            for d in self.train_dirs_lesions:
                for f in (self.lesion_dir / d).glob('*/*.nii.gz'):
                    if 'seg' in f.name:
                        lesion_paths.append(f)
            print(f"Number of lesion masks: {len(lesion_paths)}")

            a_transformed, b_transformed, loc, scale = self.get_truncnorm_params()
            self.lesion_augmentation, _ = self.init_lesion_augmentation(
                lesion_paths, a_transformed, b_transformed, loc, scale)
            self.transform_kwargs = {'pre_transform': self.lesion_augmentation}

    def save(self, path):
        self.save_config(__file__, path)


config = SyntheticMixedConfig()
