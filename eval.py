import torch
import pytorch_lightning as L

import hydra
from omegaconf import DictConfig, OmegaConf

from models import SampleModel

import utils.instantiator as instantiator

@hydra.main(version_base=None, config_path="configs", config_name="general")
def eval(cfg: DictConfig):
	L.seed_everything(cfg.setup.seed)

	# instantiate required elements
	test_dataloader = instantiator.get_dataloader_from_cfg(cfg, phase = 'test')
	model = instantiator.get_checkpoint_from_cfg(cfg)

	# show evaluation information
	print("===== Evaluation Information =====")
	print(f"Dataset: {cfg.dataset._target_}")
	print(f"Checkpoint: {cfg.checkpoint_path}")
	print("==================================")

	trainer = L.Trainer()
	out = trainer.test(model, dataloaders=(test_dataloader))

	print(out)


if __name__ == "__main__":
	eval()