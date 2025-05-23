from pathlib import Path
from sslearn.model_and_training.data_module import SyntheticMixedDataModule
from sslearn.model_and_training.model import Model
from sslearn.model_and_training.data_augmentation import OptionalLesionMixPopulate
from sslearn.configs.config import Config


class LesionMixConfig(Config):
    model_description = 'LesionMix'
    data_module = SyntheticMixedDataModule
    model_class = Model

    lesion_augmentation = None  # This will be initialised in the update_params method

    sampler_args = {'balance_attribute': 'label',
                    'num_samples': None,
                    'balance_with': 1}

    @staticmethod
    def init_lesion_augmentation(lesion_paths):
        lesion_augmentation = OptionalLesionMixPopulate(
            lesion_paths=lesion_paths,
            load_distribution_type='uniform',
            im_name='t2',
            seg_name='segmentation',
            sc_seg_name='sc_seg',
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


config = LesionMixConfig()
