import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "reproducibility.json"


def load_reproducibility_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


def apply_reproducibility_defaults() -> Dict[str, Any]:
    config = load_reproducibility_config()
    seed_cfg = config.get("seed", {})
    set_global_seed(
        seed=int(seed_cfg.get("global_seed", 42)),
        deterministic=bool(seed_cfg.get("deterministic", True)),
        benchmark=bool(seed_cfg.get("benchmark", False)),
    )
    return config
