from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, PolynomialLR
import torchio as tio
import lightning as L
from typing import Optional

from ..configs.config import Config

CHANNELS_DIMENSION = 1


def get_lr_scheduler(optimizer, args):
    if args is None or len(args) == 0:
        return None
    if args['scheduler_type'] == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=args['T_max'], eta_min=args['eta_min'])
    elif args['scheduler_type'] == 'LinearLR':
        return LinearLR(optimizer, start_factor=args['start_factor'], end_factor=args['end_factor'],
                        total_iters=args['total_iters'])
    elif args['scheduler_type'] == 'PolynomialLR':
        return PolynomialLR(optimizer, total_iters=args['total_iters'], power=args['power'])
    else:
        raise ValueError(f"Scheduler type {args['scheduler_type']} not recognized.")


class Model(L.LightningModule):
    def __init__(self, config: Config, unet: Optional[nn.Module] = None):
        super().__init__()
        self.unet = config.unet_class(**config.archi_params) if unet is None else unet
        self.config = config
        self.training_batch_size = config.training_batch_size
        self.validation_batch_size = config.validation_batch_size

        loss_params = {} if not hasattr(config, 'loss_params') else config.loss_params
        if 'unet' in loss_params and loss_params['unet'] is None:
            loss_params['unet'] = self.unet
        self.loss_func = config.loss(**loss_params)
        # Alternative to torch.nn.modules.utils._triple:
        example_array = torch.Tensor(2, 1, *config.patch_size)
        self.example_input_array = {seq: example_array for seq in self.config.modalities}

    def get_scale_factors(self):
        """ Get the scale factors for each depth in deep supervision based on the kernel sizes of the pooling
        operations in the encoder."""
        pool_op_kernel_sizes = self.config.archi_params['pool_op_kernel_sizes']  # List of lists of 3 ints
        scale_factors = [(1, 1, 1)] * len(pool_op_kernel_sizes)  # Default to no pooling/rescaling
        for i, kernel_size in enumerate(pool_op_kernel_sizes):
            # Divide the previous scale factor by the current pooling factor to get the current scale factor
            scale_factors[i] = tuple(np.array(scale_factors[i - 1]) / np.array(kernel_size))
        return scale_factors

    def prepare_batch(self, batch: dict):
        modalities = self.config.modalities

        inputs = {modality: batch[modality][tio.DATA] for modality in modalities if modality in batch}
        targets = batch['segmentation'][tio.DATA]

        if self.unet.deep_supervision:
            n = len(self.unet.decoder)  # Excludes the bottleneck
            # Get scale factors for each decoder block from the pool_op_kernel_sizes (which may differ for each axis)
            scale_factors = self.get_scale_factors()
            targets = [torch.nn.functional.interpolate(targets, scale_factor=scale_factors[i], mode='nearest')
                       for i in range(n)]
            # Reverse the list so that the lowest resolution (bottleneck) is first
            targets = targets[::-1]

        return inputs, targets

    def forward(self, x: dict):
        if len(self.config.modalities) > 1:
            input_tensor = torch.cat([x[modality] for modality in self.config.modalities], dim=CHANNELS_DIMENSION)
        else:
            input_tensor = x[self.config.modalities[0]]
        return self.unet(input_tensor)

    def step(self, batch: dict):
        inputs, targets = self.prepare_batch(batch)
        logits = self.forward(inputs)
        loss = self.loss_func(logits, targets['segmentation'])

        # Save examples to file
        if getattr(self.config, 'save_examples_dir', None) is not None and not self.trainer.sanity_checking:
            save_epoch = getattr(self.config, 'save_examples_epoch', 0)
            if not isinstance(save_epoch, list):
                save_epoch = [save_epoch]
            if self.current_epoch in save_epoch:
                if 'label' in batch:
                    labels = batch['label'].detach().cpu()
                else:
                    labels = None

                save_dir = Path(self.config.save_examples_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                iteration = self.trainer.global_step
                for i in range(len(batch['name'])):
                    im_id = batch['name'][i]
                    label_str = f'_label-{labels[i]}' if labels is not None else ''
                    for modality in inputs.keys():
                        fpath = save_dir / f'{modality}_it-{iteration}_{i}{label_str}_id-{im_id}_input.nii.gz'
                        tio.ScalarImage(tensor=inputs[modality][i].detach().cpu(),
                                        affine=batch[modality][tio.AFFINE][i].detach().cpu()
                                        ).save(fpath)
                    if self.config.deep_supervision_levels > 1:
                        target = targets['segmentation'][-1][i][1:, ...].detach().cpu()
                    else:
                        target = targets['segmentation'][i][1:, ...].detach().cpu()
                    tio.ScalarImage(tensor=target, affine=batch[modality][tio.AFFINE][i].detach().cpu()
                                    ).save(str(fpath).replace('input', 'target'))

        return loss, targets, logits

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)

        dict_log = {'losses/train_loss': loss,
                    'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                    'step': self.current_epoch + 1}

        self.log_dict(dict_log, on_step=False, on_epoch=True, batch_size=self.training_batch_size)

        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        loss, targets, logits = self.step(batch)
        # Log step as epoch number, so that epoch shows on x-axis of tensorboard (and as float)
        self.log('step', self.current_epoch + 1.0, batch_size=self.validation_batch_size)
        self.log('losses/val_loss', loss, on_step=False, on_epoch=True, batch_size=self.validation_batch_size)

        return loss

    def test_step(self, batch: dict, batch_idx: int):
        batch_loss = self.step(batch)[0]
        self.log("test_loss", batch_loss, batch_size=self.validation_batch_size)
        return batch_loss

    def configure_optimizers(self):
        optimizer = self.config.optimizer_args.pop('optimizer')(self.parameters(), **self.config.optimizer_args)
        scheduler = get_lr_scheduler(optimizer, getattr(self.config, 'scheduler_args', None))

        if scheduler is not None:
            lr_scheduler = {
                'scheduler': scheduler,
                'interval': getattr(self.config.scheduler_args, 'interval', 'epoch'),
                'frequency': getattr(self.config.scheduler_args, 'frequency', 1),
            }
            return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)
        else:
            return optimizer
