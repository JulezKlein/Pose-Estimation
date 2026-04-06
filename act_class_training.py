#!/usr/bin/env python3
"""Action-classifier training entry-point.

Usage
-----
    # default config
    python act_class_training.py

    # override individual values via dotlist syntax
    python act_class_training.py training.epochs=100 model.hidden_size=256

    # point to a different config file
    python act_class_training.py --config action_classification/model/config.yml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf

from action_classification.model.lstm_model import ActionClassifier
from action_classification.utils.dataloader import build_dataloaders

_DEFAULT_CONFIG = Path(__file__).parent / "action_classification" / "model" / "config.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ActionClassifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=str(_DEFAULT_CONFIG), type=Path,
        help="Path to the OmegaConf YAML config file.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dotlist overrides, e.g. training.epochs=100.",
    )
    return parser.parse_args()


def build_config(config_path: Path, overrides: list[str]):
    """Load YAML config and apply CLI overrides."""
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args.config, args.overrides)

    print("[CONFIG]")
    print(OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------ #
    # Reproducibility
    # ------------------------------------------------------------------ #
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    device = torch.device(cfg.training.device)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"[INFO] Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}"
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = ActionClassifier.from_config(cfg)
    model.to(device)
    print(f"[INFO] Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ------------------------------------------------------------------ #
    # Optimizer & loss
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    binary = cfg.model.output_size == 1
    if binary:
        pos_weight = train_loader.dataset.dataset.pos_weight().to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    ckpt_dir = Path(cfg.checkpoints.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)                          # (B, output_size)
            if binary:
                loss = loss_fn(logits.squeeze(1), yb.float())
            else:
                loss = loss_fn(logits, yb.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                if binary:
                    loss = loss_fn(logits.squeeze(1), yb.float())
                    preds = (logits.squeeze(1) > 0).long()
                else:
                    loss = loss_fn(logits, yb.long())
                    preds = logits.argmax(dim=1)
                val_loss += loss.item() * xb.size(0)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total else 0.0

        print(
            f"[Epoch {epoch:>4}/{cfg.training.epochs}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}"
        )

        # --- checkpoint ---
        if cfg.checkpoints.save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / "best.pt"
            model.save(str(ckpt_path))
            print(f"[INFO] Saved best checkpoint -> {ckpt_path}")

    # Always save the final model
    final_path = ckpt_dir / "last.pt"
    model.save(str(final_path))
    print(f"[INFO] Saved final checkpoint -> {final_path}")

    # ------------------------------------------------------------------ #
    # Export after training (if configured)
    # ------------------------------------------------------------------ #
    if cfg.export.onnx:
        model.export_onnx(cfg.export.onnx, seq_len=cfg.training.seq_len)
        print(f"[INFO] Exported ONNX -> {cfg.export.onnx}")
    if cfg.export.coreml:
        model.export_coreml(cfg.export.coreml, seq_len=cfg.training.seq_len)
        print(f"[INFO] Exported CoreML -> {cfg.export.coreml}")


if __name__ == "__main__":
    main()
