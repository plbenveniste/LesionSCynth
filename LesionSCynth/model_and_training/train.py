from pathlib import Path
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from importlib import import_module


def run(args, config_path):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the configuration
    config = import_module(config_path).config

    # Create the training and validation transforms
    config.update_params(mode='train')
    config.create_transforms(patch_size=config.patch_size, **getattr(config, 'transform_kwargs', {}))
    L.seed_everything(seed=config.random_seed, workers=True)  # Set the random seed

    # Save the configuration
    config.save(out_dir)

    # Initialise the model, including architecture, how training steps are defined, etc.
    model = config.model_class(config=config)

    callbacks = []
    if config.callbacks is not None:
        # Add the output directory to the callbacks as this is supplied as arg to training script, not in config
        for cb in config.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = out_dir
            callbacks.append(cb)

    if args.gpu_ids is None:
        args.gpu_ids = 'auto'

    data = config.data_module(Path(config.data_dir), config)
    logger = L.pytorch.loggers.TensorBoardLogger(out_dir, name='lightning_logs')

    trainer = L.Trainer(
        default_root_dir=out_dir, max_epochs=config.max_epochs, max_steps=config.max_steps, logger=logger,
        callbacks=callbacks, check_val_every_n_epoch=config.check_val_every_n_epoch, devices=args.gpu_ids,
        accelerator="gpu", profiler="simple", precision=config.precision, gradient_clip_val=config.gradient_clip_val,
    )

    trainer.fit(model=model, datamodule=data)

    return model


def main(args):
    run(args, args.config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--out_dir', required=True, type=Path, help='Path to the output folder.')
    parser.add_argument('-cfg', '--config', default=None, help='Relative path to the config file.')
    parser.add_argument('-gpu', '--gpu_ids', default=None, type=int, nargs='+', help='e.g. 0 or 0 1')
    args = parser.parse_args()

    main(args)
