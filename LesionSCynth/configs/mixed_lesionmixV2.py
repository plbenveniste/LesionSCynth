from pathlib import Path
import pandas as pd
from scipy.stats import truncnorm
from ..model_and_training.data_module import SyntheticMixedDataModule
from ..model_and_training.model import Model
from ..model_and_training.data_augmentation import OptionalLesionMixPopulate
from ..configs.config import Config


class SyntheticMixedConfig(Config):
    model_description = 'LesionMix V2'
    data_module = SyntheticMixedDataModule
    model_class = Model

    lesion_augmentation = None  # This will be initialised in the update_params method

    sampler_args = {'balance_attribute': 'label',
                    'num_samples': None,
                    'balance_with': 1}

    def get_truncnorm_params(self):
        # Parameters for truncnorm distribution
        contrast_summary = pd.read_csv(self.contrast_summary_path, dtype={'first_fold': str, 'subset': str})
        contrast_summary = contrast_summary[(contrast_summary['first_fold'] == str(self.first_fold)) &
                                            (contrast_summary['subset'] == self.exp_scale)]

        a = contrast_summary[contrast_summary['statistic'] == 'pct_20']['surround_5_contrast'].values[0]
        loc = contrast_summary[contrast_summary['statistic'] == 'mean']['surround_5_contrast'].values[0]
        scale = contrast_summary[contrast_summary['statistic'] == 'std']['surround_5_contrast'].values[0]
        b = 10.0  # Effectively truncated only at left side
        print(f"Truncnorm parameters: a={a}, loc={loc}, scale={scale}, b={b}")
        a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
        return a_transformed, b_transformed, loc, scale

    def init_lesion_augmentation(self, lesion_paths):
        a_transformed, b_transformed, loc, scale = self.get_truncnorm_params()
        lesion_augmentation = OptionalLesionMixPopulate(
            lesion_paths=lesion_paths,
            load_distribution_type='uniform',
            im_name='t2',
            seg_name='segmentation',
            sc_seg_name='sc_seg',
            factor_distribution=truncnorm(a=a_transformed, b=b_transformed, loc=loc, scale=scale)
        )
        return lesion_augmentation

    def update_params(self, mode='train', **kwargs):
        super().update_params(mode=mode, **kwargs)
        if mode == 'train':
            # Get the paths to the relevant stored lesion masks
            mask_paths = []
            for d in self.train_dirs_lesions:
                for f in (self.lesion_dir / d).glob('*/*.nii.gz'):
                    if 'seg' in f.name:
                        mask_paths.append(f)
            # Get the paths to the lesion intensity images along with the corresponding segmentation masks
            lesion_paths = [(p, Path(str(p).replace('_seg.nii.gz', '.nii.gz'))) for p in mask_paths]

            self.lesion_augmentation = self.init_lesion_augmentation(lesion_paths)
            self.transform_kwargs = {'pre_transform': self.lesion_augmentation}

    def save(self, path):
        self.save_config(__file__, path)


config = SyntheticMixedConfig()
