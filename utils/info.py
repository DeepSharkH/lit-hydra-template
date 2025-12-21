from omegaconf import DictConfig, OmegaConf

def print_cfg_info(cfg: DictConfig, sep_width: int = 45) -> None:
	"""
	Print info for a DictConfig to std output.
	
	:param cfg: Description
	:type cfg: DictConfig
	"""
	print("=" * sep_width)
	print("Config file configuration")
	print("=" * sep_width)

	# relevant information
	for key, elem in cfg.dataset.items():
		print(f"[DATASET] {key} = {elem}")

	print("")

	for key, elem in cfg.model.items():
		print(f"[MODEL] {key} = {elem}")

	print("")

	for key, elem in cfg.optimizer.items():
		print(f"[OPTIMIZER] {key} = {elem}")

	print("")

	# rest of information
	print("-" * sep_width)
	print("Full configuration:")
	print(OmegaConf.to_yaml(cfg))
	print("=" * sep_width)