import torch
import pytorch_lightning as L
import hydra
from omegaconf import DictConfig

class SampleModel(L.LightningModule):
	def __init__(self, cfg: DictConfig, input_dim=1, output_dim=1):
		super().__init__()
		self.save_hyperparameters(cfg, logger=False)
		self.model = torch.nn.Sequential(
			torch.nn.Linear(input_dim, output_dim)
		)

	def forward(self, x):
		out = self.model(x)
		return out

	def training_step(self, batch, batch_idx):
		x, label = batch
		single_input = (x.dim() == 1)
		
		x = x.unsqueeze(-1) if single_input else x
		out = self.model(x)
		out = out.squeeze(-1) if single_input else out

		loss = torch.nn.functional.mse_loss(out, label)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, batch, batch_idx):
		return self.training_step(batch, batch_idx) # same as training step

	def test_step(self, batch, batch_idx):
		return self.training_step(batch, batch_idx) # same as training step

	def configure_optimizers(self):
		optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
		return optimizer
