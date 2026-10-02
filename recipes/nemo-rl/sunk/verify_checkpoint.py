"""Compare native GRPO checkpoints, export the final policy, and reload it."""

import json
import math
from pathlib import Path
import tempfile

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf


def main():
    results = Path("/results")
    events = EventAccumulator(str(results / "logs/tensorboard")).Reload()
    gradients = [
        {"step": point.step, "value": point.value}
        for point in events.Scalars("train/grad_norm")
    ]
    assert (
        len(gradients) >= 2
        and all(math.isfinite(point["value"]) for point in gradients)
        and any(point["value"] > 0 for point in gradients)
    ), "Expected two recorded optimizer updates and a nonzero gradient"
    checkpoints = results / "checkpoints"
    states = []
    with tempfile.TemporaryDirectory() as scratch:
        for step in (1, 2):
            destination = Path(scratch) / f"step{step}.pt"
            dcp_to_torch_save(
                str(checkpoints / f"step_{step}/policy/weights"), str(destination)
            )
            states.append(
                torch.load(destination, map_location="cpu", weights_only=True)["model"]
            )
    changed = []
    max_delta = 0.0
    for name in states[0]:
        first, second = states[0][name], states[1][name]
        assert torch.isfinite(first).all() and torch.isfinite(second).all(), (
            f"Checkpoint contains non-finite values in {name}"
        )
        if not torch.equal(first, second):
            changed.append(name)
            max_delta = max(
                max_delta, (first.float() - second.float()).abs().max().item()
            )
    assert changed, "The two optimizer steps saved identical model weights"
    assert math.isfinite(max_delta) and max_delta > 0, "Invalid weight delta"
    del states
    exported = results / "hf-policy"
    convert_dcp_to_hf(
        str(checkpoints / "step_2/policy/weights"),
        str(exported),
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )
    tokenizer = AutoTokenizer.from_pretrained(exported, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        exported, local_files_only=True, torch_dtype=torch.bfloat16
    ).eval()
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Calculate 3 + 4. Reply with only the integer answer.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        assert torch.isfinite(model(**inputs).logits).all(), (
            "Reloaded logits are non-finite"
        )
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    evidence = {
        "checkpoints_compared": [1, 2],
        "changed_tensors": len(changed),
        "gradient_norms": gradients,
        "max_absolute_weight_delta": max_delta,
        "hf_policy": str(exported),
        "reload_success": True,
        "reload_logits_finite": True,
        "reloaded_policy_response": tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ),
    }
    (results / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
