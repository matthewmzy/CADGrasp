import fire
import os
from network.model_trainer import DiffusionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import loggers as pl_loggers
from LASDiffusion.utils.utils import ensure_directory, run, get_tensorboard_dir, find_best_epoch, exists


def train_from_folder(
    ibs_path: str = "IBSGrasp/ibsdata",
    scene_pc_path: str = "data/scenes",
    ibs_load_per_scene: int = 64,
    results_folder: str = 'LASDiffusion/results',
    name: str = "debug",
    model_mode: str = "diffusion",
    image_size: int = 40,
    base_channels: int = 32,
    optimizier: str = "adamw",
    attention_resolutions: str = "4, 8",
    lr: float = 2e-4,
    batch_size: int = 64,
    with_attention: bool = True,
    num_heads: int = 4,
    dropout: float = 0.1,
    noise_schedule: str = "linear",
    kernel_size: float = 2.0,
    ema_rate: float = 0.999,
    save_last: bool = True,
    verbose: bool = False,
    training_epoch: int = 200000,
    in_azure: bool = False,
    new: bool = True,
    continue_training: bool = False,
    debug: bool = False,
    use_sketch_condition: bool = True,
    use_text_condition: bool = False,
    seed: int = 777,
    save_every_epoch: int = 1,
    gradient_clip_val: float = 1.,
    feature_drop_out: float = 0.1,
    data_augmentation: bool = False,
    view_information_ratio: float = 2.0,
    vit_global: bool = False,
    vit_local: bool = True,
    split_dataset: bool = False,
    elevation_zero: bool = False,
    detail_view: bool = False,
    zero_num_workers: bool = False,
    scaling: int = 1,
):
    if not in_azure:
        debug = True
    else:
        debug = False

    results_folder = results_folder + "/" + name
    ensure_directory(results_folder)
    if continue_training:
        new = False

    if new:
        run(f"rm -rf {results_folder}/*")

    model_args = dict(
        results_folder=results_folder,
        ibs_path=ibs_path,
        ibs_load_per_scene=ibs_load_per_scene,
        scene_pc_path=scene_pc_path,
        model_mode=model_mode,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        noise_schedule=noise_schedule,
        use_sketch_condition=use_sketch_condition,
        use_text_condition=use_text_condition,
        base_channels=base_channels,
        optimizier=optimizier,
        attention_resolutions=attention_resolutions,
        with_attention=with_attention,
        num_heads=num_heads,
        dropout=dropout,
        ema_rate=ema_rate,
        verbose=verbose,
        save_every_epoch=save_every_epoch,
        kernel_size=kernel_size,
        training_epoch=training_epoch,
        gradient_clip_val=gradient_clip_val,
        debug=debug,
        image_feature_drop_out=feature_drop_out,
        view_information_ratio=view_information_ratio,
        data_augmentation=data_augmentation,
        vit_global=vit_global,
        vit_local=vit_local,
        split_dataset=split_dataset,
        elevation_zero=elevation_zero,
        detail_view=detail_view,
        zero_num_workers=zero_num_workers,
        scaling=scaling
    )
    seed_everything(seed)

    model = DiffusionModel(**model_args)

    if in_azure:
        try:
            log_dir = get_tensorboard_dir()
        except Exception as e:
            log_dir = results_folder
    else:
        log_dir = results_folder

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version=None,
        name='logs',
        default_hp_metric=False
    )

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="current_epoch",
    #     dirpath=results_folder,
    #     filename="{epoch:02d}",
    #     save_top_k=10,
    #     save_last=save_last,
    #     every_n_epochs=save_every_epoch,
    #     mode="max",
    # )

    checkpoint_recent = ModelCheckpoint(
        dirpath=os.path.join(results_folder, "recent"),  # 新增recent子目录
        filename="epoch={epoch:04d}",
        monitor="current_epoch",
        save_top_k=10,
        save_last=True,
        every_n_epochs=save_every_epoch,
        mode="max",
    )

    checkpoint_history = ModelCheckpoint(
        dirpath=os.path.join(results_folder, "history"),  # 新增history子目录
        filename="epoch_{epoch:04d}",
        every_n_epochs=1,
        save_top_k=-1,  # 保存所有符合条件的历史点
        save_on_train_epoch_end=True,
    )
    
    last_epoch = find_best_epoch(results_folder)
    if os.path.exists(os.path.join(results_folder, "last.ckpt")):
        last_ckpt = "last.ckpt"
    else:
        if exists(last_epoch):
            last_ckpt = f"epoch={last_epoch:02d}.ckpt"
        else:
            last_ckpt = "last.ckpt"

    find_unused_parameters = True
    if in_azure:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy=DDPStrategy(
                              find_unused_parameters=find_unused_parameters,
                            #   start_method='spawn'
                              ),
                          logger=tb_logger,
                          max_epochs=training_epoch,
                          log_every_n_steps=10,
                          callbacks=[checkpoint_recent, checkpoint_history])
    else:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy=DDPStrategy(
                              find_unused_parameters=find_unused_parameters,
                            #   start_method='spawn'
                              ),
                          logger=tb_logger,
                          max_epochs=training_epoch,
                          log_every_n_steps=1,
                          callbacks=[checkpoint_recent, checkpoint_history])

    if continue_training and os.path.exists(os.path.join(results_folder, "recent", last_ckpt)):
        trainer.fit(model, ckpt_path=os.path.join(results_folder, "recent", last_ckpt))
    else:
        trainer.fit(model)


if __name__ == '__main__':
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn')
    fire.Fire(train_from_folder)
