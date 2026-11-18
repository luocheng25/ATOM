# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The FlyDSL QSA prefill path must hand anything it cannot compute to the native kernel."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from atom.model_ops.qwen4_exp import qsa_attention
from atom.model_ops.qwen4_exp.ops.qsa import FP8_DTYPE
from atom.model_ops.qwen4_exp.qsa_attention import Qwen4ExpAttention


def _layer(**overrides):
    layer = SimpleNamespace(
        indexer=SimpleNamespace(compress_ratio=4, token_topk=2048),
        k_scale_cache=None,
        num_kv_heads=1,
        num_heads=12,
        head_dim=256,
        scaling=0.0625,
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _selection(tokens, width):
    return torch.arange(tokens * width, dtype=torch.int32).view(tokens, width)


def _call(layer, selected=None, tokens=8):
    if selected is None:
        selected = _selection(tokens, 2051)
    query = torch.zeros(tokens, layer.num_heads, layer.head_dim, dtype=torch.bfloat16)
    key = torch.zeros(tokens, 1, layer.head_dim, dtype=torch.bfloat16)
    qsa = SimpleNamespace(host_lens=((tokens,), (tokens,)), context_slots=None)
    return Qwen4ExpAttention._flydsl_attention(layer, query, key, key, selected, qsa)


@pytest.fixture
def flydsl(monkeypatch):
    calls = []
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")
    qsa_attention.log_flydsl_fallback.cache_clear()
    qsa_attention._log_flydsl_active.cache_clear()
    monkeypatch.setattr(qsa_attention, "_flydsl_arch_supported", lambda: True)
    monkeypatch.setattr(qsa_attention, "_flydsl_takes_fp8", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        qsa_attention,
        "_flydsl_attention_op",
        lambda: lambda q, k, v, indices, **kw: calls.append(indices) or q,
    )
    return calls


def test_supported_layer_takes_flydsl(flydsl):
    assert _call(_layer()) is not None
    assert [indices.shape for indices in flydsl] == [(8, 2051)]


@pytest.mark.parametrize("budget", [1024, 2048, 3072, 4096])
def test_short_selection_is_padded_with_minus_one(flydsl, budget):
    selected = _selection(8, 11)
    layer = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=budget))
    assert _call(layer, selected) is not None
    (indices,) = flydsl
    assert indices.shape == (8, 1027)
    assert torch.equal(indices[:, :11], selected)
    assert bool((indices[:, 11:] == -1).all())


@pytest.mark.parametrize(
    "overrides",
    [
        {"indexer": SimpleNamespace(compress_ratio=4, token_topk=1536)},
        {"indexer": SimpleNamespace(compress_ratio=8, token_topk=2048)},
        {"k_scale_cache": torch.ones(1)},
    ],
    ids=["unsupported-budget", "ratio-8", "fp8-kv-without-fp8-aiter"],
)
def test_unsupported_layer_falls_back(flydsl, overrides):
    assert _call(_layer(**overrides), _selection(8, 1539)) is None
    assert flydsl == []


def _fp8_layer(pages=2, block=64):
    """A layer bound like the SGLang bridge binds an FP8 pool."""
    cache = torch.zeros(pages, block, 1, 256, dtype=FP8_DTYPE)
    scale = torch.ones(pages, block, 1)
    return _layer(
        k_cache=cache,
        v_cache=cache.clone(),
        k_scale_cache=scale,
        v_scale_cache=scale.clone(),
    )


@pytest.fixture
def flydsl_fp8(flydsl, monkeypatch):
    calls = []
    monkeypatch.setattr(qsa_attention, "_flydsl_takes_fp8", lambda: True)
    monkeypatch.setattr(
        qsa_attention,
        "_flydsl_attention_op",
        lambda: lambda q, k, v, indices, **kw: calls.append((k, v, kw)) or q,
    )
    return calls


@pytest.mark.parametrize("mfma,compute", [("1", "fp8"), ("0", "bf16")])
def test_fp8_layer_reads_the_pool_in_place(flydsl_fp8, monkeypatch, mfma, compute):
    """FP8 K/V go to aiter as the pool itself, its per-(token, KV head) scales
    and the context slots, also without a prefix."""
    monkeypatch.setenv("ATOM_QSA_FLYDSL_FP8_MFMA", mfma)
    layer, tokens = _fp8_layer(), 8
    query = torch.zeros(tokens, 12, 256, dtype=torch.bfloat16)
    key = torch.zeros(tokens, 1, 256, dtype=torch.bfloat16)
    slots = torch.arange(64, 64 + tokens, dtype=torch.int32)
    qsa = SimpleNamespace(host_lens=((tokens,), (tokens,)), context_slots=slots)
    selected = _selection(tokens, 2051)
    assert (
        Qwen4ExpAttention._flydsl_attention(layer, query, key, key, selected, qsa)
        is not None
    )
    ((k, v, kwargs),) = flydsl_fp8
    assert k.dtype == FP8_DTYPE and tuple(k.shape) == (128, 1, 256)
    assert k.data_ptr() == layer.k_cache.data_ptr()
    assert v.data_ptr() == layer.v_cache.data_ptr()
    assert kwargs["kv_slots"] is slots
    assert tuple(kwargs["k_scale"].shape) == (128, 1)
    assert kwargs["k_scale"].data_ptr() == layer.k_scale_cache.data_ptr()
    assert kwargs["v_scale"].data_ptr() == layer.v_scale_cache.data_ptr()
    assert kwargs["compute"] == compute
    assert kwargs["prefix_lens"] == (0,)


def test_fp8_layer_without_context_slots_falls_back(flydsl_fp8, caplog):
    with caplog.at_level("WARNING"):
        assert _call(_fp8_layer()) is None
    assert flydsl_fp8 == []
    assert any("context slots" in r.getMessage() for r in caplog.records)


def test_selection_over_aiter_byte_limit_falls_back(flydsl):
    # 131072 rows x 4099 int32 columns span 2,149,056,512 bytes (>= 2**31).
    tokens, width = 131072, 4099
    layer = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=4096))
    query = torch.zeros(1, 12, 256, dtype=torch.bfloat16).expand(tokens, -1, -1)
    selected = torch.zeros(1, width, dtype=torch.int32).expand(tokens, -1)
    qsa = SimpleNamespace(host_lens=((tokens,), (tokens,)), context_slots=None)
    assert (
        Qwen4ExpAttention._flydsl_attention(layer, query, query, query, selected, qsa)
        is None
    )
    assert flydsl == []


def test_unsupported_arch_falls_back(flydsl, monkeypatch):
    monkeypatch.setattr(qsa_attention, "_flydsl_arch_supported", lambda: False)
    assert _call(_layer()) is None
    assert flydsl == []


def test_flag_off_falls_back(flydsl, monkeypatch):
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "0")
    assert _call(_layer()) is None
    assert flydsl == []


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if "native kernel" in r.message]


def test_each_fallback_reason_is_logged_once(flydsl, monkeypatch, caplog):
    budget = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=1536))
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert _call(budget, _selection(8, 1539)) is None
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        for _ in range(3):
            assert _call(_layer()) is None
    reasons = _warnings(caplog)
    assert len(reasons) == 2
    assert "budget 1536" in reasons[0] and "graph capture" in reasons[1]


def test_unlaid_forward_and_flag_off_log_nothing(flydsl, monkeypatch, caplog):
    """A forward the bridge did not lay out (target verify, warmup) is not a
    fallback, and with the flag off nothing is."""
    no_layout = SimpleNamespace(host_lens=None, context_slots=None)
    query = torch.zeros(8, 12, 256, dtype=torch.bfloat16)
    with caplog.at_level("WARNING"):
        assert (
            Qwen4ExpAttention._flydsl_attention(
                _layer(), query, query, query, _selection(8, 2051), no_layout
            )
            is None
        )
        monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "0")
        assert (
            _call(_layer(indexer=SimpleNamespace(compress_ratio=8, token_topk=2048)))
            is None
        )
    assert _warnings(caplog) == [] and flydsl == []


def test_first_flydsl_call_is_logged_once(flydsl, caplog):
    with caplog.at_level("INFO"):
        for _ in range(3):
            assert _call(_layer()) is not None
    active = [r for r in caplog.records if "FlyDSL kernel" in r.message]
    assert len(active) == 1 and len(flydsl) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_mixed_prefix_batch_gathers_full_contexts(monkeypatch):
    """Prefixed and fresh requests with unequal lengths on non-consecutive
    pages, crossing page boundaries: the bridge's context rows must be every
    request's full context in request-major order, and the kernel must receive
    exactly those K/V rows plus the per-request query / prefix lengths."""
    from atom.plugin.sglang.qwen4_exp_bridge import _flydsl_prefill_layout

    device, block, pages = "cuda", 4, 12
    prefixes, extends = (5, 0, 2), (3, 6, 1)
    seqs = tuple(p + e for p, e in zip(prefixes, extends))
    tables = torch.tensor(
        [[7, 2, 11], [5, 9, 11], [0, 11, 11]], dtype=torch.int32, device=device
    )
    batch = SimpleNamespace(extend_seq_lens_cpu=list(extends), seq_lens_cpu=list(seqs))
    host_lens, context_slots = _flydsl_prefill_layout(
        batch, len(seqs), sum(extends), tables, block
    )
    assert host_lens == (extends, seqs)
    expected_slots = [
        int(tables[r, t // block]) * block + t % block
        for r, seq in enumerate(seqs)
        for t in range(seq)
    ]
    assert context_slots.tolist() == expected_slots

    # Each cache row holds its own flat index, so gathered rows name themselves.
    rows = torch.arange(pages * block, device=device, dtype=torch.bfloat16)
    k_cache = rows.view(pages, block, 1, 1).expand(-1, -1, 1, 256).contiguous()
    calls = []
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")
    monkeypatch.setattr(qsa_attention, "_flydsl_arch_supported", lambda: True)
    monkeypatch.setattr(
        qsa_attention,
        "_flydsl_attention_op",
        lambda: lambda q, k, v, indices, **kw: calls.append((k, v, kw)) or q,
    )
    layer = _layer(k_cache=k_cache, v_cache=-k_cache)
    tokens = sum(extends)
    query = torch.zeros(tokens, 12, 256, dtype=torch.bfloat16, device=device)
    qsa = SimpleNamespace(host_lens=host_lens, context_slots=context_slots)
    selected = _selection(tokens, 2051).to(device)
    assert (
        Qwen4ExpAttention._flydsl_attention(layer, query, query, query, selected, qsa)
        is not None
    )
    ((k, v, kwargs),) = calls
    assert k[:, 0, 0].tolist() == expected_slots
    assert torch.equal(v, -k)
    assert kwargs["query_lens"] == extends
    assert kwargs["prefix_lens"] == prefixes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "prefixes, extends",
    [
        ((0,), (7,)),
        ((0,), (300,)),
        ((0,), (512,)),
        ((200,), (300,)),
        ((0, 0, 100), (100, 37, 250)),
        ((0,), (1000,)),
    ],
    ids=["7", "300", "512", "prefix-200+300", "mixed-3-requests", "1000-unpadded"],
)
def test_short_context_matches_native_kernel(monkeypatch, prefixes, extends):
    """Short contexts shrink the selection (its width follows max_seq_len, not
    the budget), so it is padded with -1 to the next FlyDSL width, past the
    budget's own layout. Every row still lists all its complete blocks, then
    its tail, left-aligned, so the real kernel must match the native
    `qsa_sparse_paged_gqa` on the same selection."""
    from atom.model_ops.qwen4_exp.ops.qsa import (
        qsa_select_paged_tokens,
        qsa_sparse_paged_gqa,
    )
    from atom.plugin.sglang.qwen4_exp_bridge import _flydsl_prefill_layout

    if not qsa_attention._flydsl_arch_supported():
        pytest.skip("FlyDSL QSA attention needs gfx942")
    try:
        qsa_attention._flydsl_attention_op()
    except ImportError:
        pytest.skip("aiter without FlyDSL QSA attention (ROCm/aiter#6576)")
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")

    device, block, ratio, budget, heads, dim = "cuda", 64, 4, 2048, 6, 256
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=device, generator=gen)

    seqs = [p + e for p, e in zip(prefixes, extends)]
    width = max(-(-s // block) for s in seqs)
    pages = len(seqs) * width
    tables = (
        torch.randperm(pages, device=device, generator=gen)
        .view(len(seqs), width)
        .to(torch.int32)
    )
    positions = torch.cat(
        [torch.arange(p, p + e, device=device) for p, e in zip(prefixes, extends)]
    )
    token_to_req = torch.cat(
        [
            torch.full((e,), r, dtype=torch.int32, device=device)
            for r, e in enumerate(extends)
        ]
    )
    selected = qsa_select_paged_tokens(
        randn(positions.numel(), 4, 128),
        randn(pages, block // ratio, 1, 128),
        tables,
        token_to_req,
        positions,
        torch.tensor(seqs, dtype=torch.int32, device=device),
        budget,
        ratio,
        max_seq_len=max(seqs),
        allow_flydsl=True,
    )
    if max(seqs) <= 512:
        assert selected.shape[1] < 1027  # the padded case under review

    k_cache, v_cache = randn(pages, block, 1, dim), randn(pages, block, 1, dim)
    query = randn(positions.numel(), heads, dim)
    slots = (
        tables[token_to_req.long(), positions // block].long() * block
        + positions % block
    )
    batch = SimpleNamespace(extend_seq_lens_cpu=list(extends), seq_lens_cpu=seqs)
    host_lens, context_slots = _flydsl_prefill_layout(
        batch, len(seqs), positions.numel(), tables, block
    )
    layer = _layer(
        indexer=SimpleNamespace(compress_ratio=ratio, token_topk=budget),
        num_heads=heads,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    qsa = SimpleNamespace(host_lens=host_lens, context_slots=context_slots)
    flydsl = Qwen4ExpAttention._flydsl_attention(
        layer,
        query,
        k_cache.flatten(0, 1)[slots],
        v_cache.flatten(0, 1)[slots],
        selected,
        qsa,
    )
    assert flydsl is not None
    native = qsa_sparse_paged_gqa(
        query,
        k_cache,
        v_cache,
        selected,
        tables,
        token_to_req,
        softmax_scale=layer.scaling,
        num_decode_requests=None,
    )
    torch.testing.assert_close(flydsl, native, atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("compute", ["bf16", "fp8"])
@pytest.mark.parametrize(
    "prefixes, extends",
    [((0,), (300,)), ((200,), (300,)), ((0, 0, 100), (100, 37, 250)), ((0,), (3000,))],
    ids=["300", "prefix-200+300", "mixed-3-requests", "3000"],
)
def test_fp8_pool_matches_native_kernel(monkeypatch, compute, prefixes, extends):
    """An FP8 pool with per-(token, KV head) scales, as the SGLang bridge binds
    it, read in place through the context slots: aiter's BF16-compute mode
    must match the native FP8 kernel on the same selection, and its FP8 MFMA
    mode (FP8 Q and probabilities) stay within a few percent of it."""
    from atom.model_ops.qwen4_exp.ops.qsa import (
        qsa_select_paged_tokens,
        qsa_sparse_paged_gqa,
        qsa_store_quantized_rows,
    )
    from atom.plugin.sglang.qwen4_exp_bridge import _flydsl_prefill_layout

    if not qsa_attention._flydsl_arch_supported():
        pytest.skip("FlyDSL QSA attention needs gfx942")
    if not qsa_attention._flydsl_takes_fp8():
        pytest.skip("aiter qsa_sparse_prefill without FP8 K/V")
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")
    monkeypatch.setenv("ATOM_QSA_FLYDSL_FP8_MFMA", "1" if compute == "fp8" else "0")

    device, block, ratio, budget, heads, dim = "cuda", 64, 4, 2048, 12, 256
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=device, generator=gen)

    seqs = [p + e for p, e in zip(prefixes, extends)]
    width = max(-(-s // block) for s in seqs)
    pages = len(seqs) * width
    tables = (
        torch.randperm(pages, device=device, generator=gen)
        .view(len(seqs), width)
        .to(torch.int32)
    )

    def packed(lengths, starts):
        positions = torch.cat(
            [torch.arange(s, s + n, device=device) for s, n in zip(starts, lengths)]
        )
        requests = torch.cat(
            [
                torch.full((n,), r, dtype=torch.int32, device=device)
                for r, n in enumerate(lengths)
            ]
        )
        slots = (
            tables[requests.long(), positions // block].long() * block
            + positions % block
        )
        return positions, requests, slots

    # Every context token's K/V (prefix and extend) quantized into the pool.
    k_cache = torch.zeros(pages, block, 1, dim, dtype=FP8_DTYPE, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_scale = torch.zeros(pages, block, 1, device=device)
    v_scale = torch.zeros_like(k_scale)
    _, _, context = packed(seqs, [0] * len(seqs))
    qsa_store_quantized_rows(
        k_cache,
        k_scale,
        context,
        randn(context.numel(), 1, dim),
        v_cache,
        v_scale,
        randn(context.numel(), 1, dim),
    )
    positions, token_to_req, _ = packed(extends, prefixes)
    selected = qsa_select_paged_tokens(
        randn(positions.numel(), 4, 128),
        randn(pages, block // ratio, 1, 128),
        tables,
        token_to_req,
        positions,
        torch.tensor(seqs, dtype=torch.int32, device=device),
        budget,
        ratio,
        max_seq_len=max(seqs),
        allow_flydsl=True,
    )
    query = randn(positions.numel(), heads, dim)
    batch = SimpleNamespace(extend_seq_lens_cpu=list(extends), seq_lens_cpu=seqs)
    host_lens, context_slots = _flydsl_prefill_layout(
        batch, len(seqs), positions.numel(), tables, block, read_pool=True
    )
    layer = _layer(
        indexer=SimpleNamespace(compress_ratio=ratio, token_topk=budget),
        num_heads=heads,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
    )
    qsa = SimpleNamespace(host_lens=host_lens, context_slots=context_slots)
    flydsl = Qwen4ExpAttention._flydsl_attention(
        layer, query, None, None, selected, qsa
    )
    assert flydsl is not None
    native = qsa_sparse_paged_gqa(
        query,
        k_cache,
        v_cache,
        selected,
        tables,
        token_to_req,
        softmax_scale=layer.scaling,
        num_decode_requests=None,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if compute == "bf16":
        torch.testing.assert_close(flydsl, native, atol=2e-2, rtol=2e-2)
        return
    # Per-(row, head) norm-relative errors; norms below the median count as
    # the median, so rows whose values cancel keep their absolute error.
    norms = native.float().norm(dim=-1)
    floor = torch.maximum(norms, norms.median().clamp_min(1e-6))
    error = (flydsl.float() - native.float()).norm(dim=-1) / floor
    assert float(error.mean()) < 0.05 and float(error.max()) < 0.2, (
        float(error.mean()),
        float(error.max()),
    )
