import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
from utils.info import print_cfg_info

@hydra.main(version_base=None, config_path="../configs", config_name="general")
def my_app(cfg : DictConfig) -> None:
    print("""
===============================
   Hydra Config Test Script
===============================
This is to see configs loaded by Hydra. Add new configs via CLI if \
needed until get your desired setup. 

A full config file for training should contain at least the following sections:
    1. dataset. E.g., +dataset=sample_dataset
    2. model. E.g., +model=sample_model
    3. optimizer. E.g., +optimizer=adam

A full example setup with everything required for training is as follows:

python scripts/try_hydra.py +dataset=sample_dataset +model=sample_model \
+optimizer=adam
===============================
""")

    print("*" * 60)
    print("")

    print_cfg_info(cfg)

if __name__ == "__main__":
    my_app()

