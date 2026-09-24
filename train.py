import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L

import utils.instantiator as instantiator

@hydra.main(version_base=None, config_path="configs", config_name="general")
def train(cfg : DictConfig):
	L.seed_everything(cfg.setup.seed)

	# instantiate all required objects for training
	train_dataloader = instantiator.get_dataloader_from_cfg(cfg, phase = 'train')
	val_dataloader = instantiator.get_dataloader_from_cfg(cfg, phase = 'val')
	model = instantiator.get_model_from_cfg(cfg)
	wandb_logger = instantiator.get_logger_from_cfg(cfg)
	checkpoint_callback = instantiator.get_checkpoint_callback_from_cfg(cfg)

	# resume from checkpoint if specified
	if 'checkpoint_path' in cfg and cfg.checkpoint_path is not None:
		print("Resuming training from checkpoint:", cfg.checkpoint_path)
		model = instantiator.get_checkpoint_from_cfg(cfg)

	# show training information
	print("===== Training Information =====")
	print(f"Dataset: {cfg.dataset._target_}")
	print(f"Model: {cfg.model._target_}")
	print(f"Optimizer: {cfg.optimizer._target_}")
	print("================================")
	print("Full Config:")
	print(OmegaConf.to_yaml(cfg))

	# start training
	trainer = L.Trainer(max_epochs=cfg.training.max_epochs, callbacks=[checkpoint_callback], logger=wandb_logger)
	trainer.fit(model, train_dataloaders=(train_dataloader), val_dataloaders=(val_dataloader))


if __name__ == "__main__":
	train()