"""Small single-GPU integration smoke using NeMo RL's GRPO loss.

This transparent custom driver is separate from the native NeMo RL trainer in
sunk/native_grpo.py. Generation uses the current in-memory policy every step.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl_sandbox.environment import SandboxRewardClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--revision", default="7ae557604adf67be50417f59c2c2f167def9a775"
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", default="/workspace/results/smoke")
    args = parser.parse_args()
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    loss_fn = ClippedPGLossFn(
        dict(
            reference_policy_kl_penalty=0.0,
            ratio_clip_min=0.2,
            ratio_clip_max=0.2,
            ratio_clip_c=None,
            use_on_policy_kl_approximation=False,
            use_importance_sampling_correction=False,
            token_level_loss=True,
        )
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    with SandboxRewardClient() as env:
        for step in range(args.steps):
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "What is 17 + 26? Give the answer as an integer.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            model.eval()
            with torch.no_grad():
                seq = model.generate(
                    **inputs,
                    num_return_sequences=4,
                    max_new_tokens=48,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    # Match raw policy logprobs and the EOS used by token_mask.
                    repetition_penalty=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                seq[:, prompt_length:], skip_special_tokens=True
            )
            rewards = torch.tensor(env.score(responses, ["43"] * len(responses)))
            prompts = inputs["input_ids"].cpu().repeat(len(responses), 1)
            baseline, std = calculate_baseline_and_std_per_prompt(
                prompts, rewards, torch.ones_like(rewards), leave_one_out_baseline=False
            )
            advantage = ((rewards - baseline) / (std + 1e-6)).cuda()
            token_mask = torch.zeros_like(seq, dtype=torch.float32)
            for row in range(len(seq)):
                generated = seq[row, prompt_length:]
                eos = (generated == tokenizer.eos_token_id).nonzero()
                length = int(eos[0].item()) + 1 if len(eos) else len(generated)
                token_mask[row, prompt_length : prompt_length + length] = 1
            attention_mask = torch.ones_like(seq)
            # Generation and teacher-forced log probabilities use the same policy.
            with torch.no_grad():
                logits = model(seq, attention_mask=attention_mask).logits.float()
                old = torch.zeros(seq.shape, device="cuda")
                old[:, 1:] = (
                    logits[:, :-1]
                    .log_softmax(-1)
                    .gather(-1, seq[:, 1:, None])
                    .squeeze(-1)
                )
                del logits
            optimizer.zero_grad()
            logits = model(seq, attention_mask=attention_mask).logits
            data = BatchedDataDict(
                input_ids=seq,
                advantages=advantage[:, None].expand_as(old),
                prev_logprobs=old,
                generation_logprobs=old,
                reference_policy_logprobs=old,
                token_mask=token_mask,
                sample_mask=torch.ones(len(seq), device="cuda"),
            )
            loss, metrics = loss_fn(
                logits, data, torch.tensor(len(seq), device="cuda"), token_mask.sum()
            )
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if (
                not torch.isfinite(loss)
                or not math.isfinite(grad_norm)
                or grad_norm <= 0
            ):
                raise RuntimeError(
                    "No finite, nonzero policy gradient; increase rollout diversity"
                )
            optimizer.step()
            delta = (
                sum(
                    float((p.detach() - p0).float().square().sum())
                    for p, p0 in zip(params, initial)
                )
                ** 0.5
            )
            record = dict(
                step=step + 1,
                reward_mean=float(rewards.mean()),
                rewards=rewards.tolist(),
                responses=responses,
                gradient_norm=grad_norm,
                parameter_delta_l2=delta,
                loss=float(loss),
                cpu_environment=env.sandbox.sandbox_id,
            )
            history.append(record)
            print(json.dumps(record), flush=True)
            (output / "metrics.json").write_text(json.dumps(history, indent=2))
            del logits, loss, data
        model.save_pretrained(output / "adapter")
        tokenizer.save_pretrained(output / "adapter")
        summary = dict(
            backend="custom-driver-nemo-rl-loss",
            model=args.model,
            model_revision=args.revision,
            nemo_rl_revision="b1c86a816c5e2b4ca41ece193624a38dc62e6fdf",
            gpu=torch.cuda.get_device_name(),
            optimizer_steps=len(history),
            parameter_delta_l2=history[-1]["parameter_delta_l2"],
            reward_calls=env.calls,
            checkpoint="adapter",
            status="passed",
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
