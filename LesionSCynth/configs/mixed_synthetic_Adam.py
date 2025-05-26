from torch.optim import AdamW
from torch.nn import GroupNorm
from sslearn.configs.mixed_synthetic import SyntheticMixedConfig


class SyntheticMixedAdamConfig(SyntheticMixedConfig):
    model_description = 'Mixed Synthetic - AdamW'

    def update_params(self, mode='train', **kwargs):
        super().update_params(mode=mode, **kwargs)
        self.archi_params['norm_op'] = GroupNorm

    def update_optimizer_scheduler(self):
        optimizer_args = {
            'optimizer': AdamW,
            'lr': 0.01,
            'weight_decay': 0.01,
        }
        self.optimizer_args = optimizer_args
        self.scheduler_args = None

    def save(self, path):
        # Save the current file, the base config file and pickle the current version of the config
        self.save_config(__file__, path)


config = SyntheticMixedAdamConfig()
