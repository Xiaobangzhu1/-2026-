import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:
    yaml = None

from TEST_DATASET import TrainDataset
from eeg_pipeline import (
    build_model,
    choose_device,
    dataset_paths,
    load_dataset_info,
    resolve_dataset_dir,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG classifier.")
    parser.add_argument("--config", default=None, help="Path to YAML config to load (overrides defaults).")
    parser.add_argument("--dataset", default="MDD", help="Dataset name under data-root.")
    parser.add_argument("--data-root", default="course project", help="Root directory containing datasets.")
    parser.add_argument("--model", default="sleep_ctnet", choices=["sleep_ctnet", "gru", "lstm"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--temporal-filters", type=int, default=16)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=64)
    parser.add_argument("--pool-size-1", type=int, default=8)
    parser.add_argument("--pool-size-2", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    args = parser.parse_args()
    apply_config_overrides(args, parser)
    return args


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def resolve_default_config_path(dataset: str, model: str) -> Path:
    if normalize_name(dataset) == "sleep" and normalize_name(model) == "sleep_ctnet":
        return Path("configs/sleep/ctnet.yaml")
    return Path("configs/default.yaml")


def load_yaml_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def apply_config_overrides(args, parser) -> None:
    config_path = Path(args.config) if args.config else resolve_default_config_path(args.dataset, args.model)
    config_values = load_yaml_config(config_path)
    cli_flags = {arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")}

    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        key = action.dest.replace("-", "_")
        if key not in config_values:
            continue
        option_strings = set(action.option_strings)
        if option_strings & cli_flags:
            continue
        setattr(args, action.dest, config_values[key])

    args.config = str(config_path)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            logits = model(data)
            loss = criterion(logits, label)

            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == label).sum().item()
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    dataset_dir = resolve_dataset_dir(args.data_root, args.dataset)
    info = load_dataset_info(dataset_dir)
    paths = dataset_paths(dataset_dir)

    num_classes = len(info["dataset"]["category_list"])
    channels = len(info["dataset"]["channels"])

    train_ds = TrainDataset(paths["train"])
    val_ds = TrainDataset(paths["val"])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, channels, num_classes, args).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_name = args.run_name or f"{args.dataset.lower()}_{args.model}"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    history_path = checkpoint_dir / f"{run_name}_history.json"

    best_val_acc = -1.0
    best_epoch = -1
    history = []

    print(f"Dataset: {args.dataset}")
    print(f"Train shape: {tuple(train_ds.x.shape)}")
    print(f"Val shape:   {tuple(val_ds.x.shape)}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            if hasattr(model, "clip_gradients"):
                model.clip_gradients()
            optimizer.step()

            batch_size = label.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == label).sum().item()
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        train_acc = train_correct / train_count
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_record)
        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model,
                    "model_state": model.state_dict(),
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                    "channels": channels,
                    "category_list": info["dataset"]["category_list"],
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "model": args.model,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
