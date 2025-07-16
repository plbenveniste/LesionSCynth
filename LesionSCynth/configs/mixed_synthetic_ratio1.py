from ..configs.mixed_synthetic_Adam import SyntheticMixedAdamConfig


class SyntheticMixedRatio1(SyntheticMixedAdamConfig):
    model_description = 'Mixed - 0.25 synthetic / 0.75 real'

    sampler_args = {'balance_attribute': 'label',
                    'balance_with': 1,
                    'ratio': {0: 0.25, 1: 0.75}}

    def update_params(self, mode='train', **kwargs):
        super().update_params(mode=mode, **kwargs)
        # Maintain consistency of number of iterations/epochs with previous experiments
        self.sampler_args['num_samples'] = int(self.exp_scale) * 2

    def save(self, path):
        self.save_config(__file__, path)


config = SyntheticMixedRatio1()
