"""Publish PEFT LoRA checkpoints and sample version-pinned serverless rollouts.

This module provides the checkpoint transport, not a NeMo RL trainer backend.
The trainer must consume returned token IDs and publish after each update.
"""

import argparse
import json
import math
import os
import re
import urllib.request
from pathlib import Path


def publish(
    directory: Path, *, entity: str, project: str, name: str, base_model: str
) -> str:
    """Upload a PEFT adapter and return its immutable inference model URI."""
    import wandb

    config = json.loads((directory / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError("Expected a PEFT LoRA adapter, not a full-model checkpoint")
    if config.get("base_model_name_or_path") != base_model:
        raise ValueError("Adapter base model must match the serving base model exactly")
    rank = config.get("r")
    if not isinstance(rank, int) or not 1 <= rank <= 16:
        raise ValueError("This recipe supports LoRA ranks 1 through 16")
    if any(int(r) > 16 for r in config.get("rank_pattern", {}).values()):
        raise ValueError("Per-layer LoRA ranks must not exceed 16")
    weights = directory / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("Export adapter_model.safetensors before publishing")

    # Pass credentials from the environment; never call wandb.login(), which
    # can persist the API key in the user's netrc file.
    with wandb.init(
        entity=entity,
        project=project,
        job_type="publish-lora",
        settings=wandb.Settings(disable_code=True),
    ) as run:
        artifact = wandb.Artifact(
            name,
            type="lora",
            metadata={"wandb.base_model": base_model},
            storage_region="coreweave-us",
        )
        # Upload only serving artifacts, not trainer logs or credential files.
        artifact.add_file(str(directory / "adapter_config.json"))
        artifact.add_file(str(weights))
        run.log_artifact(artifact)
        artifact.wait()
        version = artifact.version
    return f"wandb-artifact:///{entity}/{project}/{name}:{version}"


def sample(
    model_uri: str,
    messages: list[dict],
    *,
    entity: str,
    project: str,
    max_tokens: int = 128,
    prompt_token_ids: list[int] | None = None,
    base_model: str | None = None,
) -> dict:
    """Return exact sampled tokens and their log probabilities for one policy.

    Chat sampling requires vLLM's public ``return_token_ids`` extension.
    With explicit prompt IDs, use Completions and ``return_tokens_as_token_ids``.
    Fail if the serving model omits the metadata needed for training.
    """
    if not re.fullmatch(r"wandb-artifact:///[^/]+/[^/]+/[^/:]+:v\d+", model_uri):
        raise ValueError(
            "Pin an uploaded adapter version (:vN); aliases are unsafe within a batch"
        )
    body = {
        "model": model_uri,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": True,
        "return_token_ids": True,
    }
    endpoint = "chat/completions"
    if prompt_token_ids is not None:
        if not prompt_token_ids or not base_model:
            raise ValueError(
                "Explicit-token sampling requires prompt IDs and base model"
            )
        endpoint = "completions"
        body.pop("messages")
        body.pop("return_token_ids")
        body.update(
            prompt=prompt_token_ids,
            logprobs=1,
            return_tokens_as_token_ids=True,
            add_special_tokens=False,
        )
    request = urllib.request.Request(
        "https://api.inference.wandb.ai/v1/" + endpoint,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + os.environ["WANDB_API_KEY"],
            "OpenAI-Project": f"{entity}/{project}",
            "Content-Type": "application/json",
            "User-Agent": "cwsandbox-nemo-rl/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    # Some backends report the canonical base ID for adapter requests. The
    # immutable request URI selects the policy; reject any unrelated identity.
    allowed_models = {model_uri}
    if prompt_token_ids is not None:
        allowed_models.add(base_model)
    if result.get("model") not in allowed_models:
        raise ValueError("Endpoint returned an unexpected model identity")
    choice = result["choices"][0]
    if prompt_token_ids is not None:
        prompt_ids = prompt_token_ids
        logprobs = choice.get("logprobs") or {}
        tokens = logprobs.get("tokens") or []
        if not tokens or any(not re.fullmatch(r"token_id:\d+", t) for t in tokens):
            raise ValueError("Endpoint did not return exact completion token IDs")
        output_ids = [int(t.split(":", 1)[1]) for t in tokens]
        entries = [{"logprob": p} for p in logprobs.get("token_logprobs", [])]
        usage = result.get("usage") or {}
        if usage.get("prompt_tokens") != len(prompt_ids):
            raise ValueError("Endpoint changed the explicit prompt token count")
        if usage.get("completion_tokens") != len(output_ids):
            raise ValueError("Endpoint omitted completion tokens")
        output_text = choice["text"]
    else:
        prompt_ids = result.get("prompt_token_ids")
        output_ids = choice.get("token_ids")
        entries = (choice.get("logprobs") or {}).get("content")
        output_text = choice["message"].get("content") or ""
    if not prompt_ids or not output_ids or not entries:
        raise ValueError(
            "Endpoint did not return exact prompt/output tokens and logprobs"
        )
    if len(output_ids) != len(entries):
        raise ValueError("Output token IDs and logprobs do not align")
    logprobs = [entry["logprob"] for entry in entries]
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("Endpoint returned non-finite log probabilities")
    return {
        "policy_version": model_uri,
        "prompt_token_ids": prompt_ids,
        "output_token_ids": output_ids,
        "output_logprobs": logprobs,
        "text": output_text,
        "finish_reason": choice["finish_reason"],
        "request_id": result["id"],
        "served_model": result.get("model"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("publish")
    upload.add_argument("directory", type=Path)
    upload.add_argument("--name", required=True)
    upload.add_argument("--base-model", required=True)
    rollout = commands.add_parser("sample")
    rollout.add_argument("model_uri")
    rollout.add_argument("--prompt", required=True)
    rollout.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.command == "publish":
        print(
            publish(
                args.directory,
                entity=args.entity,
                project=args.project,
                name=args.name,
                base_model=args.base_model,
            )
        )
    else:
        print(
            json.dumps(
                sample(
                    args.model_uri,
                    [{"role": "user", "content": args.prompt}],
                    entity=args.entity,
                    project=args.project,
                    max_tokens=args.max_tokens,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
