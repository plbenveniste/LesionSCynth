from sslearn.model_and_training.data_module import LesionsOnlyDataModule
from sslearn.model_and_training.model import FinetuneModel
from sslearn.configs.config import Config


class LesionsOnlyConfig(Config):
    model_description = 'Real Lesions Only Model'
    data_module = LesionsOnlyDataModule
    model_class = FinetuneModel

    # Double the number of samples per volume since we are not taking any extra no-lesion volumes
    samples_per_volume = 4

    def update_params(self, mode='train', **kwargs):
        super().update_params(mode=mode, **kwargs)

        if self.exp_scale == '145':
            self.train_dirs += ['0']

    def save(self, path):
        self.save_config(__file__, path)


config = LesionsOnlyConfig()
