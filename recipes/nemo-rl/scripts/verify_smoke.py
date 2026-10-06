"""Reload the saved PEFT adapter and prove it changes the base model's logits."""

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/workspace/results/smoke"))
    args = parser.parse_args()
    summary = json.loads((args.output / "summary.json").read_text())
    adapter = args.output / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            summary["model"],
            revision=summary.get("model_revision"),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    inputs = tokenizer("What is 17 + 26?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        baseline = model(**inputs).logits.float()
    model = PeftModel.from_pretrained(model, adapter).eval()
    with torch.no_grad():
        adapted = model(**inputs).logits.float()
    if not torch.isfinite(baseline).all() or not torch.isfinite(adapted).all():
        raise RuntimeError("Checkpoint reload produced non-finite logits")
    delta = float((baseline - adapted).abs().max())
    if not math.isfinite(delta) or delta <= 0:
        raise RuntimeError("Reloaded checkpoint did not change model logits")
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    if not tensors or any(
        not torch.isfinite(value).all() for value in tensors.values()
    ):
        raise RuntimeError("Checkpoint adapter tensors are empty or non-finite")
    evidence = {
        "checkpoint_reload_passed": True,
        "max_logit_delta_vs_base": delta,
        "adapter_tensors": len(tensors),
        "nonzero_lora_B_values": sum(
            int(v.count_nonzero()) for k, v in tensors.items() if "lora_B" in k
        ),
    }
    (args.output / "reload.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
