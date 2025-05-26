from pathlib import Path
import pickle as pkl
import shutil
import traceback
import warnings

import torch
from lightning.pytorch.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
import torchio as tio

from ..model_and_training.unet_3d import UNet
from ..model_and_training.loss import DeepSupervisionLoss, CrossEntropyDiceLoss
from ..im_utils import PadToTargetShape


def get_git_hash(parent_level=0):
    """ Get the hash of the current git commit.
    Args:
        parent_level: int. The number of parent directories to go up to find the directory containing .git directory.
    Returns:
        str: The hash of the current git commit.
    """
    # Get the path to where the function is called (not the path to the utils package script).
    path_to_calling_file = Path(traceback.extract_stack()[-2].filename)
    parent_dir = path_to_calling_file.parents[parent_level]
    git_dir = parent_dir / '.git'
    if not git_dir.exists():
        parent0 = path_to_calling_file.parents[0]
        parent1 = path_to_calling_file.parents[1]
        msg = f'Could not find .git directory in {parent_dir}. Try changing the parent_level parameter. ' \
              f'E.g. parent_level=0 for {parent0}, parent_level=1 for {parent1}, etc.'
        # raise FileNotFoundError(msg)
        warnings.warn(msg)

    with (git_dir / 'HEAD').open('r') as head:
        ref = head.readline().split(' ')[-1].strip()

    with (git_dir / ref).open('r') as git_hash:
        return git_hash.readline().strip()


def save_config(file, path):
    file = Path(file)
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(file, path / file.name)
    base_config_path = file.parent / 'config.py'
    if base_config_path.exists():
        shutil.copyfile(base_config_path, path / 'config.py')
    with open(path / 'git_hash.txt', 'w') as f:
        f.write(get_git_hash())


class Config:
    model_name = 'Default Model'
    data_module = None

    random_seed = 0
    precision: _PRECISION_INPUT = 32
    gradient_clip_val = 12

    # region Data --------------------------------------------------------------------------
    modalities = ['t2']

    data_dir = None  # The root directory for the training data
    contrast_summary_path = None  # Path to the CSV file containing the contrast summary stats
    save_examples_dir = None  # Directory to save training examples after augmentation
    lesion_dir = None  # Directory containing the lesion masks (& maybe intensity images). subdir defined in training_dirs_lesions

    training_dirs = ['train']  # or e.g., ['fold_1', 'fold_2', 'fold_3']
    training_dirs_lesions = training_dirs.copy()  # the directory names within lesion_dir to be used for training

    # Run checks on the paths and directories
    for p in [data_dir, contrast_summary_path, save_examples_dir, lesion_dir]:
        if p is None or not isinstance(p, Path) or not p.exists():
            raise ValueError(f'Invalid path: {p} for {p.__name__}. Please set paths as pathlib.Path objects in'
                             f'the config.py file.')
    for d in training_dirs:
        if not (data_dir / d).exists():
            raise FileNotFoundError(f'Training directory {d} does not exist in {data_dir}. '
                                    f'Please set the correct training_dirs in the config.py file.')
    for d in training_dirs_lesions:
        if not (lesion_dir / d).exists():
            raise FileNotFoundError(f'Training directory {d} does not exist in {lesion_dir}. '
                                    f'Please set the correct training_dirs_lesions in the config.py file.')
    # endregion
    # region Training -----------------------------------------------------------------------
    max_epochs = 2000
    max_steps = -1
    learning_rate = 0.01
    end_lr_ratio = 0.001
    training_batch_size = 8
    validation_batch_size = None  # To be determined based on 2*training_batch_size in update_params()

    patch_size = (48, 48, 320)
    sampler = tio.data.UniformSampler(patch_size)
    patch_overlap = (24, 24, 160)  # for inference only

    max_queue_length = 64
    samples_per_volume = 2

    pin_memory = True
    def num_workers(self): return 8

    # endregion
    # region Loss ---------------------------------------------------------------------
    loss = DeepSupervisionLoss

    # Number of levels to use for deep supervision. 1 means no deep supervision. Only used for supervised training.
    deep_supervision_levels = 4

    loss_params = {
        'loss': CrossEntropyDiceLoss(
            weight_ce=0.5,
            weight_dice=0.5,
            class_weights_ce=None  # e.g. [1, 10] to give 10x more weight to class 1
        ),
        'unet': None,
        'n_levels': deep_supervision_levels,
    }

    # endregion
    # region Architecture ---------------------------------------------------------------------
    unet_class = UNet
    n_classes = 2
    n_stages = 5
    archi_params = {
        'dropout': 0.0,
        'norm_op': torch.nn.InstanceNorm3d,
        # 'feature_noise_sigma': 0.1,
        'n_modalities': len(modalities),
        'conv_kernel_sizes': [[3, 3, 3]] * n_stages,
        'pool_op_kernel_sizes': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2]][:n_stages],
        'max_num_features': 320,
        'deep_supervision': deep_supervision_levels > 1,
        'residual_encoder': True,
        'n_conv_per_stage_encoder': [1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6][:n_stages],
        'n_conv_per_stage_decoder': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1][:n_stages],
    }

    num_pool_per_axis = [sum([p > 1 for p in pool])
                         for pool in zip(*archi_params['pool_op_kernel_sizes'])]
    must_be_divisible_by = [2 ** n for n in num_pool_per_axis]
    # Check each dimension of patch is divisible by the corresponding number
    for i, (p, n) in enumerate(zip(patch_size, must_be_divisible_by)):
        if p % n != 0:
            raise ValueError(f"Patch size {patch_size} is not divisible by {must_be_divisible_by} in dimension {i}")

    # endregion
    # region Callbacks/Logging -------------------------------------------------------

    # Note: The 'monitor' metric in ModelCheckpoint and EarlyStopping should be logged in validation_step in model.py
    loss_checkpoint = ModelCheckpoint(save_top_k=1,
                                      monitor="losses/val_loss",
                                      mode="min",
                                      filename="best_loss_epoch-{epoch}_loss-{losses/val_loss:.4f}_f1-{lesion_metrics/f1:.4f}",
                                      auto_insert_metric_name=False)

    last_checkpoint = ModelCheckpoint(filename="last_epoch-{epoch}_loss-{losses/val_loss:.4f}_f1-{lesion_metrics/f1:.4f}",
                                      auto_insert_metric_name=False)

    callbacks = [loss_checkpoint, last_checkpoint, DeviceStatsMonitor()]

    # endregion
    # region Params to be defined later -------------------------------------------------------
    scheduler_args, optimizer_args = None, None  # These are set in update_optimizer_scheduler()
    # These other params are set in update_params()
    train_dirs, val_dirs, test_dirs = None, None, None
    train_dirs_lesions = None

    training_transform, validation_transform = None, None
    transform_kwargs = {}
    check_val_every_n_epoch = None
    sampler_args = {}

    # endregion
    # region Transforms -------------------------------------------------------
    @staticmethod
    def gt_zero(x):
        return x > 0

    def create_training_transform(self, **kwargs):
        transform = tio.Compose([
            tio.RandomGamma(p=0.3),  # tio default for log gamma is U(-0.3, 0.3), nnUnet paper uses gamma ~ U(0.7, 1.5)
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5, p=1.0),  # Seems to be the same as nnUNet paper
            # sigma is given in mm, so does not need to be adjusted for anisotropic data
            tio.RandomBlur(std=(0.5, 1.), p=0.3),  # nnUNet std=(0.5, 1.5) but blurring is too strong
            tio.RandomBiasField(p=0.2),  # Not used in nnUNet paper, but seems relevant
            tio.RandomAnisotropy(axes=(0, 1), downsampling=(1.0, 2.0), p=0.25),
            # Changed to params from nnUNet paper supplementary notes, from downsampling=(1.1, 2.0), p=0.3
            tio.RandomAffine(scales=0.15, degrees=(0, 0, 30), center='image', p=0.5),
            # nnUNet does not apply elastic deformation, and applies scaling and rotation each with prob of 0.2
            PadToTargetShape(target_shape=kwargs['patch_size']),
            tio.ZNormalization(masking_method=self.gt_zero),
            tio.RandomNoise(std=(0, 0.1), p=0.15),
            # Changed to params from nnUNet paper supplementary notes, from std=(0, 0.2), p=0.3
            tio.OneHot(num_classes=2)
        ])

        if 'pre_transform' in kwargs:
            transform = tio.Compose([kwargs['pre_transform'], transform])
        if 'post_transform' in kwargs:
            transform = tio.Compose([transform, kwargs['post_transform']])

        self.training_transform = transform
        return

    def create_validation_transform(self, **kwargs):
        self.validation_transform = tio.Compose([
            PadToTargetShape(target_shape=kwargs['patch_size']),
            tio.ZNormalization(masking_method=self.gt_zero),
            tio.OneHot(num_classes=2),
        ])
        return

    def create_transforms(self, **kwargs):
        self.create_training_transform(**kwargs)
        self.create_validation_transform(**kwargs)

    # endregion

    def update_params(self, mode='train', **kwargs):
        if mode == 'train':
            self.check_val_every_n_epoch = max(self.max_epochs // 100, 1)
            self.update_optimizer_scheduler()

        # Determine the validation batch size based on the training batch size
        self.validation_batch_size = 2 * self.training_batch_size

        # Update any remaining parameters supplied
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_optimizer_scheduler(self):
        optimizer_args = {
            'optimizer': torch.optim.SGD,
            'lr': self.learning_rate,
            'weight_decay': 3e-5,
            'momentum': 0.99,
            'nesterov': True,
        }

        scheduler_args = {
            'scheduler_type': 'PolynomialLR',
            'total_iters': self.max_epochs,
            'power': 0.9,
        }

        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def save_config(self, file, path):
        # Pickle the config
        with open(path / 'config.pkl', 'wb') as f:
            pkl.dump(self, f)

        save_config(file, path)
