"""Training script for the Transformer Chatbot.

Automatically:
  1. Loads config.yaml (hierarchical: model / train / dataset / save)
  2. Checks for / downloads the dataset
  3. Trains the model with mixed-precision + gradient accumulation + checkpointing
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``src/`` and the project root importable for both ``python main.py``
# and ``python -m scripts.train_model`` invocations.
_SCRIPT_DIR   = Path(__file__).resolve().parent            # .../scripts
_PROJECT_ROOT = _SCRIPT_DIR.parent                         # ExplainableModel/
_SRC_DIR      = _PROJECT_ROOT / "src"                      # ExplainableModel/src
for p in (str(_SRC_DIR), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import AppConfig, load_config
from model.dataset.download_dataset import download_dataset
from model.dataset.conversation_dataset import ConversationDataset, create_dataloaders
from model.dataset.tokenizer import Tokenizer
from model import TransformerChatter


# ────────────────────────────────────────────────────────────────────────────
def build_tokenizer(texts: list[str], cfg: AppConfig) -> Tokenizer:
    """Build a character-level tokenizer and align its special token IDs."""
    tok = Tokenizer()
    tok.build_vocab(texts, min_freq=1)
    tk = cfg.model.tokenizer
    tok.pad_token_id = tk.pad_token_id
    tok.bos_token_id = tk.bos_token_id
    tok.eos_token_id = tk.eos_token_id
    return tok


def train(cfg: AppConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    from model.dataset.download_dataset import download_dataset as _download
    dataset_path = _download(cfg.dataset.dir_path, cfg.dataset.file_name, cfg.dataset.dataset_name)
    texts = ConversationDataset([], Tokenizer(), 0).load_text_file(str(dataset_path))
    print(f"[train] Loaded {len(texts)} samples from {dataset_path}")

    # ── 2. Tokenizer & vocab ────────────────────────────────────────────────
    tokenizer = build_tokenizer(texts, cfg)
    print(f"[train] Vocab size: {tokenizer.vocab_size}")

    # The actual vocab grows past the configured cap once characters are added.
    # Pydantic models are immutable by default — use model_copy(update=...) instead.
    cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"vocab_size": tokenizer.vocab_size})})

    # ── 3. DataLoaders ──────────────────────────────────────────────────────
    split = int(len(texts) * 0.9)
    train_loader, val_loader = create_dataloaders(
        texts[:split], texts[split:], tokenizer,
        batch_size=cfg.train.batch_size,
        max_length=cfg.model.max_seq_len,
        pad_token_id=cfg.model.tokenizer.pad_token_id,
    )

    # ── 4. Model ────────────────────────────────────────────────────────────
    model = TransformerChatter(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_decoder_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
    ).to(device)
    print(f"[train] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 5. Optimizer / Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    pad_id = cfg.model.tokenizer.pad_token_id

    total_steps = max(1, len(train_loader) * cfg.train.num_epochs)
    def lr_lambda(step: int) -> float:
        if step < cfg.train.warmup_steps:
            return step / max(1, cfg.train.warmup_steps)
        return max(0.0, 1.0 - (step - cfg.train.warmup_steps) / max(1, total_steps - cfg.train.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 6. Training loop ─────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ckpt_dir = Path(cfg.save.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.num_epochs}")

        for batch in pbar:
            input_ids = batch.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(input_ids)                          # [B, T, V]
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, cfg.model.vocab_size),
                    input_ids[:, 1:].reshape(-1),
                    ignore_index=pad_id,
                )
                loss = loss / cfg.train.accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg.train.accumulation_steps

            if (global_step + 1) % cfg.train.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1
            pbar.set_postfix(
                loss=f"{epoch_loss / max(1, pbar.n):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits = model(input_ids)
                    vl = F.cross_entropy(
                        logits[:, :-1, :].reshape(-1, cfg.model.vocab_size),
                        input_ids[:, 1:].reshape(-1),
                        ignore_index=pad_id,
                    )
                val_loss += vl.item()

        val_loss /= max(1, len(val_loader))
        print(
            f"[train] Epoch {epoch + 1} — "
            f"train_loss: {epoch_loss / max(1, len(train_loader)):.4f}  "
            f"val_loss: {val_loss:.4f}  "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Checkpoint ──────────────────────────────────────────────────────
        ckpt_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": cfg_to_dict(cfg),
            },
            ckpt_path,
        )
        print(f"[train] Checkpoint saved: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / "best.pt"
            torch.save(torch.load(ckpt_path, weights_only=False), best_path)
            print(f"[train] New best model! val_loss={val_loss:.4f}")

    print("[train] Training complete.")


# ────────────────────────────────────────────────────────────────────────────
def cfg_to_dict(cfg: AppConfig) -> dict:
    """Recursively convert an AppConfig (pydantic) into a plain dict (for checkpoint saving)."""
    return cfg.model_dump(mode="json")


def main() -> None:
    config_path = _PROJECT_ROOT / "config.yaml"
    cfg = load_config(str(config_path))
    print(f"[train] Config loaded from {config_path}")
    print(f"[train] model.d_model={cfg.model.d_model}, train.lr={cfg.train.learning_rate}")
    train(cfg)


if __name__ == "__main__":
    main()