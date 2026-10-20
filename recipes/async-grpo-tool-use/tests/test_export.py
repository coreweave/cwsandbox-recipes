"""The export worker's shard plan and files: each rank writes its own shard of the checkpoint."""

import json

import pytest

torch = pytest.importorskip("torch")

from cwtrain.hotload.snapshot import inspect_snapshot  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from async_grpo_tool_use import hotload  # noqa: E402
from async_grpo_tool_use.nemo_rl_ext import worker  # noqa: E402


def qwen25_7b_sizes():
    """Tensor bytes of Qwen2.5-Math-7B in bf16 (28 layers, untied embeddings)."""
    hidden, inter, vocab, kv = 3584, 18944, 152064, 512
    sizes = {"model.embed_tokens.weight": vocab * hidden * 2, "lm_head.weight": vocab * hidden * 2,
             "model.norm.weight": hidden * 2}  # fmt: skip
    for i in range(28):
        p = f"model.layers.{i}."
        sizes |= {p + "self_attn.q_proj.weight": hidden * hidden * 2, p + "self_attn.q_proj.bias": hidden * 2,
                  p + "self_attn.k_proj.weight": kv * hidden * 2, p + "self_attn.k_proj.bias": kv * 2,
                  p + "self_attn.v_proj.weight": kv * hidden * 2, p + "self_attn.v_proj.bias": kv * 2,
                  p + "self_attn.o_proj.weight": hidden * hidden * 2,
                  p + "mlp.gate_proj.weight": inter * hidden * 2, p + "mlp.up_proj.weight": inter * hidden * 2,
                  p + "mlp.down_proj.weight": inter * hidden * 2,
                  p + "input_layernorm.weight": hidden * 2, p + "post_attention_layernorm.weight": hidden * 2}  # fmt: skip
    return sizes


def test_plan_gives_every_rank_about_the_same_bytes():
    sizes = qwen25_7b_sizes()
    plan = worker.plan_shards(sizes, 8)
    loads = [sum(sizes[n] for n, shard in plan.items() if shard == rank) for rank in range(8)]
    assert sum(loads) / 2**30 == pytest.approx(14.2, abs=0.1)
    assert max(loads) / (sum(loads) / 8) < 1.05  # the slowest rank writes under 5% more than average
    assert worker.plan_shards(dict(reversed(list(sizes.items()))), 8) == plan  # same on every rank
    assert set(worker.plan_shards({"a": 1, "b": 1}, 8).values()) == {0, 1}  # no empty shards


def test_ranks_write_one_shard_each_and_cwtrain_reads_a_full_snapshot(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    tensors = {
        "a": torch.ones(4, dtype=torch.bfloat16),
        "b": torch.ones(2, 2),
        "c": torch.zeros(8, dtype=torch.bfloat16),
    }
    save_file(tensors, str(base / "model-00001-of-00001.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model-00001-of-00001.safetensors")})
    )
    (base / "config.json").write_text("{}")
    dtypes, sizes = hotload.checkpoint_dtypes(base), hotload.checkpoint_sizes(base)
    assert sizes == {"a": 8, "b": 16, "c": 16}

    # The policy holds fp32; each rank sees every tensor (the gather) and keeps its own.
    policy = [(name, t.float() + 1) for name, t in tensors.items()]
    final, tmp = tmp_path / "v000001", tmp_path / "v000001.tmp"
    results = [worker.write_checkpoint(iter(policy), tmp, dtypes, sizes, rank=r, world_size=2) for r in range(2)]
    assert [r["error"] for r in results] == [None, None]
    assert sum(r["tensors"] for r in results) == 3 and sum(r["bytes"] for r in results) == 40
    worker.finish_export(final, tmp, failed=False)
    hotload.add_model_files(base, final)

    assert sorted(p.name for p in final.iterdir()) == [
        "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ]  # fmt: skip
    loaded = {}
    for shard in final.glob("*.safetensors"):
        loaded |= load_file(str(shard))
    assert {n: t.dtype for n, t in loaded.items()} == {n: t.dtype for n, t in tensors.items()}
    assert torch.equal(loaded["b"], torch.full((2, 2), 2.0))
    info = inspect_snapshot(final)
    assert info.type.name == "FULL" and info.total_bytes > 40


def test_a_failed_rank_leaves_no_export(tmp_path):
    dtypes, sizes = {"a": "BF16", "b": "BF16"}, {"a": 2, "b": 2}
    tmp = tmp_path / "v.tmp"
    ok = worker.write_checkpoint(iter([("a", torch.ones(1)), ("b", torch.ones(1))]), tmp, dtypes, sizes, rank=0,
                                 world_size=2)  # fmt: skip
    bad = worker.write_checkpoint(iter([("a", torch.ones(1))]), tmp, dtypes, sizes, rank=1, world_size=2)
    assert ok["error"] is None and "missing 1 tensors" in bad["error"]
    worker.finish_export(tmp_path / "v", tmp, failed=True)
    assert not any(tmp_path.iterdir())
