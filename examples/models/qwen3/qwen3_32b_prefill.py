# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Qwen3 single-layer prefill forward (batch=16, max_seq=4096).

Each session in the batch can have a different input sequence length (up to
MAX_SEQ).  The ``seq_lens`` input tensor (shape [BATCH], INT32) carries the
per-session token count.  Tensors are padded to MAX_SEQ on the sequence axis;
the program only processes valid tokens per session.

Design goals:
- keep a decode-like structure and reuse the same primitive ops
- fuse work in three large auto_incore scopes per token-tile
- all pl.slice / pl.slice of GM tensors use 512-B-aligned shapes
  (full TOK_TILE rows even on the tail tile; padding rows are harmless)
- scope 2 (attention + KV cache write) iterates only over valid tokens
  to avoid writing garbage into the KV cache

Tiling (Ascend950 / ExpandMixedKernel): cross-core pipe slots must be one byte size per
mixed InCore kernel.  With TOK_TILE=4 and Q_OUT_CHUNK=64, use K_CHUNK=128 so
TOK_TILE×K_CHUNK×2 (BF16 LHS) equals TOK_TILE×64×4 (FP32 acc).  Use SEQ_TILE=64 so
FP32 score rows (SEQ_TILE×4) match BF16 q-vectors (HEAD_DIM×2) for attention.
hidden_states is reshaped to [BATCH*MAX_SEQ, HIDDEN] so token-tile GM slices are
[TOK_TILE, K] (even leading dim under UP_DOWN). RMS partial sums use [TOK_TILE, 1] so
stores stay 2D (PTO tile.store requires 2D valid_shape); avoid [1, TOK_TILE] (dim 0 = 1
under UP_DOWN) and avoid 1D [TOK_TILE] for the same reason.
RMS weights are shaped [TOK_TILE, HIDDEN] (repeated row) so γ GM tiles are [TOK_TILE, K_CHUNK];
norm uses pl.mul(..., gamma) instead of col_expand_mul(..., [1, K]) to avoid [1,·] tiles in split AIV.
Scope 2 mirrors qwen3_32b_decode_scope2: one KV incore (per-head RoPE + cache),
then grouped attention with Q_HEAD_BATCH=8, Q_HEAD_PAD=16, and ctx flattened to
[1, Q_HEAD_BATCH * head_dim] before assembling into attn_row.
"""

import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096
HIDDEN = 5120
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 25600
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS

EPS = 1e-6
ATTN_SCALE = 0.08838834764831845
HIDDEN_INV = 1.0 / HIDDEN

# Prefill tuning knobs (aligned with decode_scope2/3 where noted).
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
SEQ_TILE = 64
MLP_OUT_CHUNK = 64
TOK_TILE = 4


def build_qwen3_single_layer_prefill_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    NUM_HEADS_CFG = num_heads
    NUM_KV_HEADS_CFG = num_kv_heads
    HEAD_DIM_CFG = head_dim
    KV_HIDDEN_CFG = num_kv_heads * head_dim
    INTER_CFG = intermediate_size
    Q_PER_KV_CFG = num_heads // num_kv_heads
    # Match qwen3_32b_decode_scope2: batch Q heads per group so incore tiles avoid
    # leading dim 1 (SplitVectorKernel / UP_DOWN split halves dim 0).
    Q_HEAD_BATCH_CFG = 8
    Q_HEAD_PAD_CFG = 16
    q_groups_cfg = Q_PER_KV_CFG // Q_HEAD_BATCH_CFG
    total_q_groups_cfg = NUM_KV_HEADS_CFG * q_groups_cfg

    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    KV_OUT_BLOCKS = (KV_HIDDEN_CFG + KV_OUT_CHUNK - 1) // KV_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_CFG + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK
    CACHE_ROWS = BATCH_CFG * NUM_KV_HEADS_CFG * MAX_SEQ_CFG

    @pl.program
    class Qwen3SingleLayerPrefill:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_prefill_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ_CFG, HEAD_DIM_CFG], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ_CFG, HEAD_DIM_CFG], pl.FP32],
            k_cache: pl.Tensor[[CACHE_ROWS, HEAD_DIM_CFG], pl.BF16],
            v_cache: pl.Tensor[[CACHE_ROWS, HEAD_DIM_CFG], pl.BF16],
            input_rms_weight: pl.Tensor[[TOK_TILE, HIDDEN_CFG], pl.FP32],
            wq: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            wk: pl.Tensor[[HIDDEN_CFG, KV_HIDDEN_CFG], pl.BF16],
            wv: pl.Tensor[[HIDDEN_CFG, KV_HIDDEN_CFG], pl.BF16],
            wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[TOK_TILE, HIDDEN_CFG], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16]:
            for b in pl.parallel(0, BATCH_CFG, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                flat_hidden = pl.reshape(
                    hidden_states, [BATCH_CFG * MAX_SEQ_CFG, HIDDEN_CFG]
                )
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)
                    row_base = b * MAX_SEQ_CFG + p0
                    # Scope 1: RMSNorm + Q/K/V projections for a token tile.
                    # Uses full [TOK_TILE, ...] views from hidden_states even on the
                    # tail tile — padding rows map to allocated-but-unused MAX_SEQ
                    # slots, keeping every GM view >= 512 B aligned.
                    with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
                        sq_acc = pl.full([TOK_TILE, 1], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(
                                        flat_hidden,
                                        [TOK_TILE, K_CHUNK],
                                        [row_base, k0],
                                        valid_shape=[valid_tok, K_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            sq_acc = pl.add(
                                sq_acc,
                                pl.reshape(
                                    pl.row_sum(pl.mul(x_chunk, x_chunk)),
                                    [TOK_TILE, 1],
                                ),
                            )

                        inv_rms = pl.reshape(
                            pl.rsqrt(pl.add(pl.mul(sq_acc, HIDDEN_INV), EPS)),
                            [TOK_TILE, 1],
                        )
                        q_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.BF16)
                        k_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN_CFG], dtype=pl.BF16)
                        v_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN_CFG], dtype=pl.BF16)

                        for ob in pl.parallel(0, Q_OUT_BLOCKS, 1):
                            q0 = ob * Q_OUT_CHUNK
                            q_acc = pl.create_tensor([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                            q_acc = pl.mul(q_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                x_chunk = pl.reshape(
                                    pl.cast(
                                        pl.slice(
                                            flat_hidden,
                                            [TOK_TILE, K_CHUNK],
                                            [row_base, k0],
                                            valid_shape=[valid_tok, K_CHUNK],
                                        ),
                                        target_type=pl.FP32,
                                    ),
                                    [TOK_TILE, K_CHUNK],
                                )
                                gamma = pl.slice(input_rms_weight, [TOK_TILE, K_CHUNK], [0, k0])
                                normed = pl.mul(
                                    pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [TOK_TILE, 1])),
                                    gamma,
                                )
                                wq_chunk = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                                q_acc = pl.add(q_acc, pl.matmul(pl.cast(normed, target_type=pl.BF16), wq_chunk))
                            q_proj_tile = pl.assemble(q_proj_tile, pl.cast(q_acc, target_type=pl.BF16), [0, q0])

                        for ob in pl.parallel(0, KV_OUT_BLOCKS, 1):
                            kv0 = ob * KV_OUT_CHUNK
                            k_acc = pl.create_tensor([TOK_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                            v_acc = pl.create_tensor([TOK_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                            k_acc = pl.mul(k_acc, 0.0)
                            v_acc = pl.mul(v_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                x_chunk = pl.reshape(
                                    pl.cast(
                                        pl.slice(
                                            flat_hidden,
                                            [TOK_TILE, K_CHUNK],
                                            [row_base, k0],
                                            valid_shape=[valid_tok, K_CHUNK],
                                        ),
                                        target_type=pl.FP32,
                                    ),
                                    [TOK_TILE, K_CHUNK],
                                )
                                gamma = pl.slice(input_rms_weight, [TOK_TILE, K_CHUNK], [0, k0])
                                normed = pl.mul(
                                    pl.row_expand_mul(x_chunk, inv_rms),
                                    gamma,
                                )
                                normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                                wk_chunk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                wv_chunk = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                k_acc = pl.add(k_acc, pl.matmul(normed_bf16, wk_chunk))
                                v_acc = pl.add(v_acc, pl.matmul(normed_bf16, wv_chunk))
                            k_proj_tile = pl.assemble(k_proj_tile, pl.cast(k_acc, target_type=pl.BF16), [0, kv0])
                            v_proj_tile = pl.assemble(v_proj_tile, pl.cast(v_acc, target_type=pl.BF16), [0, kv0])

                    # Scope 2: RoPE + KV cache update + causal attention.
                    # Loop bound must be a static TOK_TILE (not seq_len) so chunk /
                    # SplitVector passes match decode_scope2; guard with ti < valid_tok.
                    attn_tile = pl.full([TOK_TILE, HIDDEN_CFG], dtype=pl.FP32, value=0.0)
                    for ti in pl.range(TOK_TILE):
                        if ti < valid_tok:
                            pos = p0 + ti
                            ctx_len = pos + 1
                            ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                            cos_row = pl.slice(rope_cos, [1, HEAD_DIM_CFG], [pos, 0])
                            sin_row = pl.slice(rope_sin, [1, HEAD_DIM_CFG], [pos, 0])
                            cos_lo = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                            cos_hi = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                            sin_lo = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                            sin_hi = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])

                            with pl.incore():
                                for ki in pl.range(NUM_KV_HEADS_CFG):
                                    kv_col = ki * HEAD_DIM_CFG
                                    k_lo = pl.slice(
                                        k_proj_tile, [1, HEAD_DIM_CFG // 2], [ti, kv_col]
                                    )
                                    k_hi = pl.slice(
                                        k_proj_tile,
                                        [1, HEAD_DIM_CFG // 2],
                                        [ti, kv_col + HEAD_DIM_CFG // 2],
                                    )
                                    k_lo_f = pl.cast(k_lo, target_type=pl.FP32)
                                    k_hi_f = pl.cast(k_hi, target_type=pl.FP32)
                                    rot_lo = pl.sub(
                                        pl.col_expand_mul(k_lo_f, cos_lo),
                                        pl.col_expand_mul(k_hi_f, sin_lo),
                                    )
                                    rot_hi = pl.add(
                                        pl.col_expand_mul(k_hi_f, cos_hi),
                                        pl.col_expand_mul(k_lo_f, sin_hi),
                                    )
                                    cache_row = (
                                        b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + ki * MAX_SEQ_CFG + pos
                                    )
                                    k_cache = pl.assemble(
                                        k_cache,
                                        pl.cast(rot_lo, target_type=pl.BF16),
                                        [cache_row, 0],
                                    )
                                    k_cache = pl.assemble(
                                        k_cache,
                                        pl.cast(rot_hi, target_type=pl.BF16),
                                        [cache_row, HEAD_DIM_CFG // 2],
                                    )
                                    v_cache = pl.assemble(
                                        v_cache,
                                        pl.cast(
                                            pl.slice(
                                                v_proj_tile,
                                                [1, HEAD_DIM_CFG],
                                                [ti, ki * HEAD_DIM_CFG],
                                            ),
                                            target_type=pl.BF16,
                                        ),
                                        [cache_row, 0],
                                    )

                            attn_row = pl.full([1, HIDDEN_CFG], dtype=pl.BF16, value=0.0)

                            for gi in pl.parallel(0, total_q_groups_cfg, 1):
                                kvh = gi // q_groups_cfg
                                qg = gi - kvh * q_groups_cfg
                                q_base = kvh * Q_PER_KV_CFG + qg * Q_HEAD_BATCH_CFG

                                q_padded = pl.create_tensor([Q_HEAD_PAD_CFG, HEAD_DIM_CFG], dtype=pl.BF16)
                                with pl.incore():
                                    for qi in pl.range(Q_HEAD_BATCH_CFG):
                                        q_col = (q_base + qi) * HEAD_DIM_CFG
                                        q_lo = pl.slice(
                                            q_proj_tile, [1, HEAD_DIM_CFG // 2], [ti, q_col]
                                        )
                                        q_hi = pl.slice(
                                            q_proj_tile,
                                            [1, HEAD_DIM_CFG // 2],
                                            [ti, q_col + HEAD_DIM_CFG // 2],
                                        )
                                        q_lo_f = pl.cast(q_lo, target_type=pl.FP32)
                                        q_hi_f = pl.cast(q_hi, target_type=pl.FP32)
                                        rot_lo_bf16 = pl.cast(
                                            pl.sub(
                                                pl.col_expand_mul(q_lo_f, cos_lo),
                                                pl.col_expand_mul(q_hi_f, sin_lo),
                                            ),
                                            target_type=pl.BF16,
                                        )
                                        rot_hi_bf16 = pl.cast(
                                            pl.add(
                                                pl.col_expand_mul(q_hi_f, cos_hi),
                                                pl.col_expand_mul(q_lo_f, sin_hi),
                                            ),
                                            target_type=pl.BF16,
                                        )
                                        q_padded = pl.assemble(q_padded, rot_lo_bf16, [qi, 0])
                                        q_padded = pl.assemble(q_padded, rot_hi_bf16, [qi, HEAD_DIM_CFG // 2])

                                    oi = pl.full(
                                        [Q_HEAD_BATCH_CFG, HEAD_DIM_CFG],
                                        dtype=pl.FP32,
                                        value=0.0,
                                    )
                                    li = pl.full(
                                        [Q_HEAD_BATCH_CFG, 1], dtype=pl.FP32, value=0.0
                                    )
                                    mi = pl.full(
                                        [Q_HEAD_BATCH_CFG, 1], dtype=pl.FP32, value=0.0
                                    )

                                for sb in pl.range(ctx_blocks):
                                    s0 = sb * SEQ_TILE
                                    valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                    cache_row0 = b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + kvh * MAX_SEQ_CFG + s0

                                    raw_scores_pad = pl.create_tensor(
                                        [Q_HEAD_PAD_CFG, SEQ_TILE], dtype=pl.FP32
                                    )
                                    with pl.incore():
                                        k_tile = pl.slice(
                                            k_cache,
                                            [SEQ_TILE, HEAD_DIM_CFG],
                                            [cache_row0, 0],
                                        )
                                        raw_scores_pad = pl.matmul(
                                            q_padded, k_tile, b_trans=True, out_dtype=pl.FP32
                                        )

                                    exp_padded = pl.create_tensor(
                                        [Q_HEAD_PAD_CFG, SEQ_TILE], dtype=pl.BF16
                                    )
                                    with pl.incore():
                                        scores_valid = pl.slice(
                                            raw_scores_pad,
                                            [Q_HEAD_BATCH_CFG, SEQ_TILE],
                                            [0, 0],
                                            valid_shape=[Q_HEAD_BATCH_CFG, valid_len],
                                        )
                                        scores_padded = pl.fillpad(
                                            scores_valid, pad_value=pl.PadValue.min
                                        )
                                        scores = pl.mul(scores_padded, ATTN_SCALE)
                                        cur_mi = pl.row_max(scores)
                                        exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                        exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                        exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                                        cur_li = pl.row_sum(exp_scores_fp32)
                                        exp_padded = pl.assemble(exp_padded, exp_scores_bf16, [0, 0])

                                    oi_tmp_pad = pl.create_tensor(
                                        [Q_HEAD_PAD_CFG, HEAD_DIM_CFG], dtype=pl.FP32
                                    )
                                    with pl.incore():
                                        v_tile = pl.slice(
                                            v_cache,
                                            [SEQ_TILE, HEAD_DIM_CFG],
                                            [cache_row0, 0],
                                        )
                                        oi_tmp_pad = pl.matmul(
                                            exp_padded, v_tile, out_dtype=pl.FP32
                                        )

                                    with pl.incore():
                                        oi_tmp = pl.slice(
                                            oi_tmp_pad,
                                            [Q_HEAD_BATCH_CFG, HEAD_DIM_CFG],
                                            [0, 0],
                                        )
                                        if sb == 0:
                                            oi = oi_tmp
                                            li = cur_li
                                            mi = cur_mi
                                        else:
                                            mi_new = pl.maximum(mi, cur_mi)
                                            alpha = pl.exp(pl.sub(mi, mi_new))
                                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                                            li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                                            oi = pl.add(
                                                pl.row_expand_mul(oi, alpha),
                                                pl.row_expand_mul(oi_tmp, beta),
                                            )
                                            mi = mi_new
                                        # Same incore as merge so oi/li stay in scope (SSA); only last sb is final.
                                        if sb + 1 == ctx_blocks:
                                            ctx = pl.row_expand_div(oi, li)
                                            ctx_flat = pl.reshape(
                                                pl.cast(ctx, target_type=pl.BF16),
                                                [1, Q_HEAD_BATCH_CFG * HEAD_DIM_CFG],
                                            )
                                            attn_row = pl.assemble(
                                                attn_row,
                                                ctx_flat,
                                                [0, q_base * HEAD_DIM_CFG],
                                            )

                            attn_tile = pl.assemble(
                                attn_tile,
                                pl.cast(attn_row, target_type=pl.FP32),
                                [ti, 0],
                            )

                    # Scope 3: WO + resid in auto_incore; post-RMS reduce in plain incore; then
                    # post-norm + MLP in a second auto_incore.  Fusing [TOK_TILE,1] RMS acc with a
                    # mixed UP_DOWN kernel makes the compiler emit Tile[[1, TOK_TILE]] temporaries;
                    # SplitVector halves dim 0 and fails when that size is 1 (Ascend950).
                    with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
                        resid1_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.FP32)
                        for ob in pl.parallel(0, Q_OUT_BLOCKS):
                            o0 = ob * Q_OUT_CHUNK
                            zero_resid1 = pl.full([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                            resid1_tile = pl.assemble(resid1_tile, zero_resid1, [0, o0])

                        for ob in pl.parallel(0, Q_OUT_BLOCKS, 1):
                            o0 = ob * Q_OUT_CHUNK
                            o_acc = pl.full([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                a_chunk = pl.cast(
                                    pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0]),
                                    target_type=pl.BF16,
                                )
                                w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                                o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                            resid = pl.reshape(
                                pl.cast(
                                    pl.slice(
                                        flat_hidden,
                                        [TOK_TILE, Q_OUT_CHUNK],
                                        [row_base, o0],
                                        valid_shape=[valid_tok, Q_OUT_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_CHUNK],
                            )
                            resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                    with pl.incore():
                        sq_acc_post = pl.full([TOK_TILE, 1], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            sq_acc_post = pl.add(
                                sq_acc_post,
                                pl.reshape(
                                    pl.row_sum(pl.mul(x_chunk, x_chunk)),
                                    [TOK_TILE, 1],
                                ),
                            )
                        inv_rms_post = pl.reshape(
                            pl.rsqrt(pl.add(pl.mul(sq_acc_post, HIDDEN_INV), EPS)),
                            [TOK_TILE, 1],
                        )

                    with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
                        post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.BF16)
                        down_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.FP32)
                        for zi in pl.range(HIDDEN_BLOCKS):
                            z0 = zi * K_CHUNK
                            down_zero_chunk = pl.full([TOK_TILE, K_CHUNK], dtype=pl.FP32, value=0.0)
                            down_proj_tile = pl.assemble(down_proj_tile, down_zero_chunk, [0, z0])

                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            gamma = pl.slice(post_rms_weight, [TOK_TILE, K_CHUNK], [0, k0])
                            normed = pl.mul(
                                pl.row_expand_mul(x_chunk, inv_rms_post),
                                gamma,
                            )
                            post_norm_tile = pl.assemble(
                                post_norm_tile,
                                pl.cast(normed, target_type=pl.BF16),
                                [0, k0],
                            )

                        for ob in pl.range(MLP_OUT_BLOCKS):
                            o0 = ob * MLP_OUT_CHUNK
                            gate_acc = pl.full([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                            up_acc = pl.full([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)

                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                post_chunk = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                                up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                            for dob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                                d0 = dob * K_CHUNK
                                down_prev = pl.slice(down_proj_tile, [TOK_TILE, K_CHUNK], [0, d0])
                                w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                                down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                                down_proj_tile = pl.assemble(down_proj_tile, down_next, [0, d0])

                        for ob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                            o0 = ob * K_CHUNK
                            down_acc = pl.add(
                                pl.slice(down_proj_tile, [TOK_TILE, K_CHUNK], [0, o0]),
                                pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, o0]),
                            )
                            out = pl.assemble(
                                out,
                                pl.cast(down_acc, target_type=pl.BF16),
                                [b, p0, o0],
                            )

            return out

    return Qwen3SingleLayerPrefill


# ---------------------------------------------------------------------------
# Build / run helpers
# ---------------------------------------------------------------------------


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
):
    import torch  # type: ignore[import]
    from pypto.runtime import TensorSpec

    kv_hidden = num_kv_heads * head_dim
    cache_rows = batch * num_kv_heads * max_seq_len

    seq_lens_data = torch.randint(1, max_seq_len + 1, (batch,), dtype=torch.int32)

    rms_w = torch.randn(1, hidden_size, dtype=torch.float32).expand(TOK_TILE, hidden_size).clone()

    return [
        TensorSpec("hidden_states", [batch, max_seq_len, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens_data),
        TensorSpec("rope_cos", [max_seq_len, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("rope_sin", [max_seq_len, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16, init_value=torch.randn),
        TensorSpec("input_rms_weight", [TOK_TILE, hidden_size], torch.float32, init_value=rms_w),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("post_rms_weight", [TOK_TILE, hidden_size], torch.float32, init_value=rms_w.clone()),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("out", [batch, max_seq_len, hidden_size], torch.bfloat16, is_output=True),
    ]


def compile_and_run(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    enable_profiling: bool = False,
    output_dir: str | None = None,
):
    """Compile the prefill program to Ascend950 (ptoas). Full on-device run needs a golden.

    Args:
        output_dir: If set, compilation artefacts are written here (directory is created).
            If ``None``, uses ``build_output/<program_name>_YYYYMMDD_HHMMSS/`` under the CWD.

    Returns:
        ``(RunResult, work_dir)`` where ``work_dir`` is the resolved output path.
    """
    import time
    from datetime import datetime
    from pathlib import Path

    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime.runner import RunResult, compile_program

    _ = (platform, device_id, enable_profiling)  # reserved for future device run

    program = build_qwen3_single_layer_prefill_program(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )
    print(f"=== {program.name} ===")
    print(program.as_python())
    print()
    
    if output_dir is not None:
        work_dir = Path(output_dir).expanduser().resolve()
    else:
        work_dir = Path("build_output") / f"{program.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    compile_program(
        program,
        work_dir,
        strategy=OptimizationStrategy.Default,
        backend_type=BackendType.Ascend950,
        dump_passes=dump_passes,
    )
    return RunResult(passed=True, execution_time=time.time() - t0), work_dir


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Write compilation output to DIR (created if missing). "
        "Default: build_output/<program>_<timestamp>/",
    )
    args = parser.parse_args()

    result, work_dir = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        enable_profiling=args.enable_profiling,
        output_dir=args.output_dir,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
    print(f"Build output: {work_dir.resolve()}")
