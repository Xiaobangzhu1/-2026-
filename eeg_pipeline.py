import json
import random
from pathlib import Path

import numpy as np
import torch

from RNN import EEGGRU, EEGLSTM
from sleep_ctnet import CTNetClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_dir(data_root: str | Path, dataset: str) -> Path:
    return Path(data_root) / dataset


def load_dataset_info(dataset_dir: Path) -> dict:
    for name in ("dataset_info.json", "dataset_info_fixed.json"):
        path = dataset_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No dataset info file found under {dataset_dir}")


def dataset_paths(dataset_dir: Path) -> dict[str, Path]:
    return {
        "train": dataset_dir / "train.h5",
        "val": dataset_dir / "val.h5",
        "test": dataset_dir / "test_x_only.h5",
    }


def build_model(model_name: str, channels: int, num_classes: int, args) -> torch.nn.Module:
    if model_name == "gru":
        return EEGGRU(
            chans=channels,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            dropout=args.dropout,
            bidirectional=not args.unidirectional,
            grad_clip=args.grad_clip,
        )
    if model_name == "lstm":
        return EEGLSTM(
            chans=channels,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            dropout=args.dropout,
            bidirectional=not args.unidirectional,
            grad_clip=args.grad_clip,
        )
    if model_name == "sleep_ctnet":
        return CTNetClassifier(
            chans=channels,
            num_classes=num_classes,
            emb_size=args.emb_size,
            depth=args.depth,
            num_heads=args.num_heads,
            dropout=args.dropout,
            temporal_filters=args.temporal_filters,
            depth_multiplier=args.depth_multiplier,
            kernel_size=args.kernel_size,
            pool_size_1=args.pool_size_1,
            pool_size_2=args.pool_size_2,
            mlp_ratio=args.mlp_ratio,
            grad_clip=args.grad_clip,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)
