"""Stage 1 post-training: DPO on golden paths / preference pairs.

Initializes the policy toward good trajectories before the online RL stage.
Uses TRL's DPOTrainer with a SMALL base model + LoRA so it trains on a 16GB
Mac (MPS). Reads dpo_pairs.jsonl from curation.

  # real training (downloads a ~135M model, trains on MPS/CPU):
  python -m p1_agent_eval.posttrain.dpo --base HuggingFaceTB/SmolLM2-135M-Instruct

  # verify the loop with no heavy deps (simulation, writes metrics):
  python -m p1_agent_eval.posttrain.dpo --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ART = Path("p1_agent_eval/posttrain/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def run_real(args) -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"DPO on {args.base}  device={device}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base)

    ds = load_dataset("json", data_files=args.pairs, split="train")
    ds = ds.select(range(min(len(ds), args.max_samples)))

    cfg = DPOConfig(
        output_dir=str(ART / "dpo"), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=4, num_train_epochs=args.epochs,
        learning_rate=5e-5, beta=0.1, logging_steps=10,
        bf16=False, report_to=[], max_length=256, max_prompt_length=128,
    )
    peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(ART / "dpo"))
    logs = [l for l in trainer.state.log_history if "loss" in l]
    return {"mode": "real", "base": args.base, "samples": len(ds),
            "final_loss": round(logs[-1]["loss"], 4) if logs else None,
            "steps": len(logs)}


def run_dry(args) -> dict:
    """No-deps simulation of the DPO objective so you can watch it converge.

    We model a toy 'policy' as a logistic preference model with a single scalar
    margin theta that must learn to prefer chosen over rejected. This is the
    exact loss DPO minimizes (sigmoid of the reward margin), just without an LLM.
    """
    import math, random
    pairs = [json.loads(l) for l in Path(args.pairs).read_text().splitlines()]
    pairs = pairs[: args.max_samples]
    rng = random.Random(0)

    # crude features: chosen gets +1 signal, rejected -1, with noise.
    theta, lr, beta = 0.0, 0.1, 0.1
    losses = []
    for epoch in range(5):
        rng.shuffle(pairs)
        tot = 0.0
        for p in pairs:
            # margin the policy assigns (chosen should score higher)
            r_chosen = 1.0 + rng.gauss(0, 0.3)
            r_reject = -1.0 + rng.gauss(0, 0.3)
            margin = theta * (r_chosen - r_reject)
            # DPO loss: -log sigmoid(beta * margin)
            loss = -math.log(1 / (1 + math.exp(-beta * margin)) + 1e-9)
            grad = -beta * (r_chosen - r_reject) * (1 - 1 / (1 + math.exp(-beta * margin)))
            theta -= lr * grad
            tot += loss
        losses.append(tot / len(pairs))
        print(f"  epoch {epoch+1}/5  dpo_loss={losses[-1]:.4f}  theta={theta:.3f}")
    return {"mode": "dry-run (simulation)", "samples": len(pairs),
            "loss_curve": [round(x, 4) for x in losses],
            "final_loss": round(losses[-1], 4), "theta": round(theta, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="p1_agent_eval/data/curated/dpo_pairs.jsonl")
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        result = run_dry(args) if args.dry_run else run_real(args)
    except Exception as e:  # noqa: BLE001
        print(f"[dpo] real training unavailable ({type(e).__name__}: {e})")
        print("[dpo] falling back to simulation so the loop still runs")
        result = run_dry(args)

    (ART / "dpo_metrics.json").write_text(json.dumps(result, indent=2))
    print("saved ->", ART / "dpo_metrics.json")


if __name__ == "__main__":
    main()
