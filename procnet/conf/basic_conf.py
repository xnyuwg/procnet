from dataclasses import dataclass
import torch


@dataclass
class BasicConfig:
    model_load_name: str = None
    model_save_name: str = None
    data_loader_shuffle: bool = None
    model_name: str = None
    device: torch.device = None
    max_len: int = 510
    learning_rate_slow: float = 1e-5
    learning_rate_fast: float = 1e-4
    gradient_accumulation_steps: int = None
    max_epochs: int = None
