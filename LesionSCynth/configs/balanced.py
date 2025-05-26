from ..model_and_training.data_module import SyntheticMixedDataModule
from ..model_and_training.model import Model
from ..configs.config import Config


class BalancedConfig(Config):
    model_description = 'Balanced - Real Only'
    data_module = SyntheticMixedDataModule
    model_class = Model

    sampler_args = {'balance_attribute': 'label',
                    'num_samples': None,
                    'balance_with': 1}

    def save(self, path):
        self.save_config(__file__, path)


config = BalancedConfig()
