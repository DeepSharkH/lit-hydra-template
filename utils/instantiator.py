import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig, OmegaConf


def get_dataset_from_cfg(cfg: DictConfig, phase: str = 'train') -> torch.utils.data.Dataset:
	"""
	Create dataset instance based on the configuration provided in the yaml file.

	Args:
		cfg (DictConfig). Hydra configuration dictionary containing dataset.
		phase (str): Phase of the dataset, e.g., 'train', 'val', 'test'.
	Returns:
		An instance of the specified dataset.
	"""

	if phase not in ['train', 'val', 'test']:
		raise ValueError(f"Invalid phase: {phase}. Must be 'train', 'val', or 'test'.")

	dataset = hydra.utils.instantiate(cfg.dataset, phase=phase)
	return dataset

def get_dataloader_from_cfg(cfg: DictConfig, phase: str = 'train') -> torch.utils.data.DataLoader:
	"""
	Create dataloader instance based on the configuration provided in the yaml file.

	Args:
		cfg (DictConfig). Hydra configuration dictionary containing dataloader.
		phase (str): Phase of the dataloader, e.g., 'train', 'val', 'test'.
	Returns:
		An instance of the specified dataloader.
	"""
	if phase not in ['train', 'val', 'test']:
		raise ValueError(f"Invalid phase: {phase}. Must be 'train', 'val', or 'test'.")

	dataset = get_dataset_from_cfg(cfg, phase)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size = cfg.training.batch_size,
		shuffle = True if phase == 'train' else False,
		num_workers = cfg.training.num_workers
	)
	return dataloader

def get_model_from_cfg(cfg: DictConfig) -> L.LightningModule:
	"""
	Create model instance based on the configuration provided in the DictConfig file.

	Args:
		cfg (DictConfig). Hydra configuration dictionary containing model.
	Returns:
		An instance of the specified model.
	"""
	model = hydra.utils.instantiate(cfg.model, cfg)
	return model

def get_checkpoint_from_cfg(cfg: DictConfig) -> L.LightningModule:
	"""
	Create model instance from checkpoint based on the configuration provided in the DictConfig file.

	Args:
		cfg (DictConfig). Hydra configuration dictionary containing checkpoint path.
	Returns:
		An instance of the specified model loaded from checkpoint.
	"""
	model_class = hydra.utils.get_class(cfg.model._target_)
	model = model_class.load_from_checkpoint(cfg.checkpoint_path, weights_only=False)
	return model

def get_logger_from_cfg(cfg: DictConfig) -> WandbLogger:
	"""
	Create logger instance based on the configuration provided in the DictConfig file.

	Args:
		cfg (DictConfig). Hydra configuration dictionary containing logger.
	Returns:
		An instance of the specified logger.
	"""
	logger = WandbLogger(
		entity = cfg.setup.wandb.entity,
		save_dir = cfg.setup.save_dir,
		project = cfg.setup.wandb.project,
		config = OmegaConf.to_container(cfg, resolve=True)
	)
	return logger

def get_checkpoint_callback_from_cfg(cfg: DictConfig) -> ModelCheckpoint:
	"""
	Create checkpoint callback instance based on the configuration provided in the DictConfig file.

	Args:
		cfg (DictConfig). Hydra configuration dictionary containing checkpoint callback.
	Returns:
		An instance of the specified checkpoint callback.
	"""
	checkpoint_callback = ModelCheckpoint(
		every_n_epochs = cfg.training.checkpoint_every_n_epochs,
		save_on_train_epoch_end = cfg.training.save_on_end,
		save_last = cfg.training.save_last
	)
	return checkpoint_callback