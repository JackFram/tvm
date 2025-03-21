# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import Dict, List, Tuple, Union

import numpy as np
import pytest
import scipy.special
import nvtx
import argparse

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm import relax
from tvm.contrib import utils
from tvm.relax.frontend.nn.llm.kv_cache import (
    AttnKind,
    RopeMode,
    _compact_kv_copy,
    _copy_single_page,
    _kv_cache_debug_get_kv,
    _kv_cache_transpose_append,
    _merge_state_inplace,
    llama_rope_with_position_map,
)
from tvm.runtime import ShapeTuple
from tvm import relax, te, tir

reserved_nseq = 1
maximum_total_seq_length = 12800 * 8
prefill_chunk_size = 12800 * 8
token_budget = 512
tidal_layer_indices = [2,]
page_size = 1
num_layers = 8
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
sm_scale = head_dim ** (-0.5)
rope_scale = 1.0
rope_theta = 1e4
dtype = "float16"
device = tvm.cuda()

fclear = None
fadd_sequence = None
fremove_sequence = None
ffork_sequence = None
fpopn = None
fbegin_forward = None
fend_forward = None
fattention = None
fattention_with_fuse_qkv = None
ftopk_attention_with_fuse_qkv = None
fsparse_attention_with_fuse_qkv = None
fdebug_get_kv = None

fattention_prefill_topk = None
fattention_prefill = None
fattention_decode = None
fattention_prefill_ragged = None
fattention_prefill_plan = None
fattention_decode_plan = None
fattention_prefill_ragged_plan = None
fattention_merge_state = None

ftranspose_append = None
fsplit_rotary = None
fcopy_single_page = None
fcopy_cache = None
fcompact_copy = None

# Tidal Function
fset_tidal = None
fupdate_tidal = None
fargtopk = None

def _attach_argtopk_func(token_budget=token_budget):
    bb = relax.BlockBuilder()
    batch_size = tir.SizeVar("batch_size", "int64")
    seq_len = tir.SizeVar("seq_len", "int64")
    num_qo_head = tir.SizeVar("num_qo_head", "int64")
    qk_product = relax.Var("qk_product", relax.TensorStructInfo((seq_len, batch_size, num_qo_head), dtype))
    with bb.function("argtopk_qk_product", [qk_product]):
        with bb.dataflow():
            topk_values, topk_indices = bb.emit(relax.op.topk(qk_product, k=token_budget, axis=0, dtype="int32"))
            output = bb.emit_output((topk_values, topk_indices))
        gv = bb.emit_func_output(output)
    return bb.finalize()


def set_global_func():
    global fclear, fadd_sequence, fremove_sequence, ffork_sequence, fpopn
    global fbegin_forward, fend_forward, fattention, fattention_with_fuse_qkv, ftopk_attention_with_fuse_qkv, fdebug_get_kv
    global fattention_prefill, fattention_prefill_topk, fattention_decode, fattention_prefill_ragged
    global fattention_prefill_plan, fattention_decode_plan, fattention_prefill_ragged_plan
    global fattention_merge_state, fsplit_rotary, fcopy_single_page
    global ftranspose_append, fcopy_cache, fcompact_copy
    global fsparse_attention_with_fuse_qkv, fset_tidal, fupdate_tidal, fargtopk

    fclear = tvm.get_global_func("vm.builtin.kv_state_clear")
    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fremove_sequence = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    ffork_sequence = tvm.get_global_func("vm.builtin.kv_state_fork_sequence")
    fpopn = tvm.get_global_func("vm.builtin.kv_state_popn")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fset_tidal = tvm.get_global_func("vm.builtin.attention_kv_cache_attention_set_tidal")
    fupdate_tidal = tvm.get_global_func("vm.builtin.attention_kv_cache_attention_update_tidal")
    fattention_with_fuse_qkv = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_attention_with_fused_qkv"
    )
    ftopk_attention_with_fuse_qkv = tvm.get_global_func(
        "vm.builtin.topk_attention_kv_cache_attention_with_fused_qkv"
    )
    fsparse_attention_with_fuse_qkv = tvm.get_global_func(
        "vm.builtin.sparse_attention_kv_cache_attention_with_fused_qkv"
    )
    fdebug_get_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv")

    def load_module(name: str, static_modules: List[tvm.runtime.Module]):
        assert len(static_modules) > 0
        if len(static_modules) == 1:
            return static_modules[0]
        static_mod = static_modules[0]
        for mod in static_modules[1:]:
            static_mod.import_module(mod)
        temp = utils.tempdir()
        mod_path = temp.relpath(f"{name}.so")
        static_mod.export_library(mod_path)
        return tvm.runtime.load_module(mod_path)

    target = tvm.target.Target.from_device(device)
    # Enable thrust for CUDA
    target_dict = dict(target.export())
    target_dict["libs"] = (
        (target_dict["libs"] + ["thrust"]) if "libs" in target_dict else ["thrust"]
    )
    target = tvm.target.Target(target_dict)
    flashinfer_prefill_mod = load_module(
        "flashinfer_prefill",
        relax.backend.cuda.flashinfer.gen_flashinfer_prefill_module(
            dtype_q=dtype,
            dtype_kv=dtype,
            dtype_o=dtype,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            target=target,
        ),
    )
    flashinfer_decode_mod = load_module(
        "flashinfer_decode",
        relax.backend.cuda.flashinfer.gen_flashinfer_decode_module(
            dtype_q=dtype,
            dtype_kv=dtype,
            dtype_o=dtype,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            target=target,
        ),
    )
    
    mod = _attach_argtopk_func(token_budget = token_budget)
    executable = relax.build(
        mod, target=target, pipeline=relax.backend.cuda.get_default_pipeline(target)
    )
    vm = relax.VirtualMachine(executable, device)
    fargtopk = vm["argtopk_qk_product"]

    fattention_prefill = flashinfer_prefill_mod["batch_prefill_with_paged_kv_cache_run"]
    fattention_prefill_topk = flashinfer_prefill_mod["topk_batch_prefill_with_paged_kv_cache_run"]
    fattention_prefill_plan = flashinfer_prefill_mod["batch_prefill_with_kv_cache_plan"]
    fattention_prefill_ragged = flashinfer_prefill_mod["batch_prefill_with_ragged_kv_cache_run"]
    fattention_prefill_ragged_plan = flashinfer_prefill_mod["batch_prefill_with_kv_cache_plan"]
    fattention_decode = flashinfer_decode_mod["batch_decode_with_paged_kv_cache_run"]
    fattention_decode_plan = flashinfer_decode_mod["batch_decode_with_paged_kv_cache_plan"]

    builts = []
    for tir_func in [
        _kv_cache_transpose_append(num_kv_heads, head_dim, dtype),
        _merge_state_inplace(num_qo_heads, head_dim, dtype, target),
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype, {}
        ),
        _copy_single_page(num_kv_heads, page_size, head_dim, dtype, target),
        _kv_cache_debug_get_kv(num_layers, num_kv_heads, head_dim, dtype),
        _compact_kv_copy(num_kv_heads, head_dim, dtype, target),
    ]:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.build(mod["main"], target=target)
        builts.append(f.entry_func)

    (
        ftranspose_append,
        fattention_merge_state,
        fsplit_rotary,
        fcopy_single_page,
        fcopy_cache,
        fcompact_copy,
    ) = builts


def create_kv_cache(rope_mode):
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    support_sliding_window = 0
    cache = fcreate(
        tvm.runtime.ShapeTuple(
            [
                reserved_nseq,
                maximum_total_seq_length,
                prefill_chunk_size,
                page_size,
                support_sliding_window,
            ]
        ),
        tvm.runtime.ShapeTuple([0, num_layers]),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,  # v_head_dim
        tvm.runtime.ShapeTuple([int(AttnKind.MHA) for _ in range(num_layers)]),
        False,  # enable_kv_transfer
        rope_mode,
        rope_scale,
        rope_theta,
        None,  # rope_ext_factors
        tvm.nd.empty((), dtype, device=device),
        ftranspose_append,
        None,  # f_transpose_append_mla
        ["flashinfer", fattention_prefill_ragged, fattention_prefill_ragged_plan],
        ["flashinfer", fattention_prefill, fattention_prefill_plan],
        ["flashinfer", fattention_prefill_topk, fattention_prefill_plan],
        ["flashinfer", fattention_decode, fattention_decode_plan],
        [],  # fattn_prefill_sliding_window
        [],  # fattn_decode_sliding_window
        [],  # fattn_prefill_with_tree_mask_paged_kv_cache
        [],  # fattn_prefill_with_tree_mask
        [],  # f_mla_prefill
        [fattention_merge_state],
        fsplit_rotary,
        fcopy_single_page,
        fcopy_cache,
        fcompact_copy,
    )
    with nvtx.annotate("setup tidal decode token budget"):
        fset_tidal(cache, token_budget)
    return cache


@pytest.fixture(params=[RopeMode.NONE, RopeMode.NORMAL, RopeMode.INLINE])
def kv_cache_and_rope_mode(request):
    set_global_func()
    return create_kv_cache(request.param), request.param


def verify_cached_kv(kv_cache, seq_ids, expected_k, expected_v):
    for seq_id in seq_ids:
        keys_expected = expected_k[seq_id]
        values_expected = expected_v[seq_id]
        assert keys_expected.shape == values_expected.shape
        seq_length = expected_k[seq_id].shape[1]
        keys = tvm.nd.empty(keys_expected.shape, dtype=dtype, device=device)
        values = tvm.nd.empty(values_expected.shape, dtype=dtype, device=device)
        fdebug_get_kv(kv_cache, seq_id, 0, seq_length, keys, values)
        tvm.testing.assert_allclose(keys.numpy(), keys_expected, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(values.numpy(), values_expected, rtol=1e-3, atol=1e-3)


def f_apply_rotary(x, offset, scale, theta):
    # x: (N, H, D)
    assert len(x.shape) == 3
    nfeat = x.shape[-1]
    nfeat_half = x.shape[-1] // 2
    x = x.astype("float32")
    y = np.concatenate([-x[:, :, nfeat_half:], x[:, :, :nfeat_half]], axis=-1)

    inv_freq = scale / (theta ** (np.arange(0, nfeat, 2).astype("float32") / nfeat))
    t = np.arange(offset, offset + x.shape[0], dtype=inv_freq.dtype)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos_values = np.cos(emb)
    sin_values = np.sin(emb)

    return np.einsum("ij,ikj->ikj", cos_values, x) + np.einsum("ij,ikj->ikj", sin_values, y)


def apply_attention(
    kv_cache,
    rope_mode: RopeMode,
    batch: List[Tuple[Union[int, Tuple[int, int, int]], int]],
    cached_k: Dict[int, np.ndarray],
    cached_v: Dict[int, np.ndarray],
    prompt_len: int,
    decode_len: int,
    batch_size: int,
) -> None:
    seq_ids = []
    append_lengths = []
    decode = True
    # print(batch)
    for i, (seq_id, append_length) in enumerate(batch):
        if append_length > 1:
            decode = False
        fork_parent_id = None
        if isinstance(seq_id, tuple):
            # Fork sequence
            seq_id, fork_parent_id, fork_pos = seq_id
            batch[i] = (seq_id, append_length)
        seq_ids.append(seq_id)
        append_lengths.append(append_length)
        if fork_parent_id is not None:
            assert fork_parent_id in cached_k
            assert seq_id not in cached_k
            ffork_sequence(kv_cache, fork_parent_id, seq_id, fork_pos)
            if fork_pos == -1:
                cached_k[seq_id] = cached_k[fork_parent_id]
                cached_v[seq_id] = cached_v[fork_parent_id]
            else:
                cached_k[seq_id] = cached_k[fork_parent_id][::, :fork_pos]
                cached_v[seq_id] = cached_v[fork_parent_id][::, :fork_pos]
        elif seq_id not in cached_k:
            fadd_sequence(kv_cache, seq_id)
            cached_k[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
            cached_v[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)

    with nvtx.annotate("BeginForward"):
        fbegin_forward(kv_cache, ShapeTuple(seq_ids), ShapeTuple(append_lengths))

    global_new_q = np.zeros((num_layers, 0, num_qo_heads, head_dim), dtype)
    global_new_k = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
    global_new_v = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)

    q_array = []
    for seq_id, append_length in batch:
        new_q = np.random.rand(num_layers, append_length, num_qo_heads, head_dim).astype(dtype)
        new_k = np.random.rand(num_layers, append_length, num_kv_heads, head_dim).astype(dtype)
        new_v = np.random.rand(num_layers, append_length, num_kv_heads, head_dim).astype(dtype)
        q_array.append(new_q)

        cached_k[seq_id] = np.concatenate(
            [
                cached_k[seq_id],
                np.stack(
                    [
                        (
                            new_k[l]
                            if rope_mode != RopeMode.NORMAL
                            else f_apply_rotary(
                                new_k[l], cached_k[seq_id].shape[1], rope_scale, rope_theta
                            )
                        )
                        for l in range(num_layers)
                    ],
                    axis=0,
                ),
            ],
            axis=1,
        )
        cached_v[seq_id] = np.concatenate([cached_v[seq_id], new_v], axis=1)
        global_new_q = np.concatenate([global_new_q, new_q], axis=1)
        global_new_k = np.concatenate([global_new_k, new_k], axis=1)
        global_new_v = np.concatenate([global_new_v, new_v], axis=1)

    for layer_id in range(num_layers):
        queries_np = global_new_q[layer_id]
        keys_np = global_new_k[layer_id]
        values_np = global_new_v[layer_id]
        qkv = tvm.nd.array(np.concatenate([queries_np, keys_np, values_np], axis=1), device)
        outputs = tvm.nd.empty(queries_np.shape, dtype, device=device)
        # Here depend on specific layers we will use different attention func calls
        if decode:
            if layer_id < tidal_layer_indices[0]:
                # Initial full attention layers
                with nvtx.annotate("decode-full-attention"):
                    fattention_with_fuse_qkv(kv_cache, layer_id, sm_scale, qkv, outputs)
            elif layer_id in tidal_layer_indices:
                # Token re-selection layers
                with nvtx.annotate("qk-inner-product-attention"):
                    qk_len = max(prompt_len+decode_len, token_budget)
                    qk_inner_product_data = tvm.nd.array(np.full((qk_len, queries_np.shape[0], num_qo_heads), -1000, dtype), device=device)
                    ftopk_attention_with_fuse_qkv(kv_cache, layer_id, sm_scale, qkv, outputs, qk_inner_product_data)
                with nvtx.annotate("topk"):
                    _, top_k_indices = fargtopk(qk_inner_product_data) # Shape (token_budget, batch_size, num_qo_heads)
                    top_k_indices = np.argsort(qk_inner_product_data.numpy(), axis=0)[-token_budget:]
                    global_tidal_indices = top_k_indices[:, :, 0].T.flatten().tolist() # Temporarily use the first head
                with nvtx.annotate("update-topk-indices"):
                    fupdate_tidal(kv_cache, ShapeTuple(seq_ids), ShapeTuple(global_tidal_indices))
            else:
                # Tidal sparse attention
                with nvtx.annotate("sparse-attention"):
                    fsparse_attention_with_fuse_qkv(kv_cache, layer_id, sm_scale, qkv, outputs)
        else:
            # Prefill full attention
            with nvtx.annotate("prefill-full-attention"):
                fattention_with_fuse_qkv(kv_cache, layer_id, sm_scale, qkv, outputs)

    fend_forward(kv_cache)


def test_paged_attention_kv_cache_prefill_and_decode(kv_cache_and_rope_mode, prompt_len=512, decode_len=8, batch_size=1):
    kv_cache, rope_mode = kv_cache_and_rope_mode
    fclear(kv_cache)

    # # Prefill.
    # operation_seq = [[(0, 6)], [(1, 8)], [(2, 11)], [(3, 16)], [(4, 19), (5, 20)]]
    # operation_seq += [[(6, 21), (7, 24)], [(2, 5), (4, 7), (8, 24)]]
    # operation_seq += [[(6, 13)], [(8, 19)], [(0, 1)], [(1, 3), (3, 8), (5, 12), (7, 11)]]
    # # Decode
    # operation_seq += [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]
    # operation_seq += [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]
    # operation_seq += [[(0, 1), (2, 1), (4, 1), (6, 1), (8, 1)]]
    # operation_seq += [[(4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]

    print(f"prompt_len: {prompt_len}, decode_len: {decode_len}, batch_size: {batch_size}")

    # Prefill.
    operation_seq = [[(i, prompt_len) for i in range(batch_size)], ]
    # Decode
    operation_seq += [[(i, 1) for i in range(batch_size)], ] * decode_len

    cached_k = {}
    cached_v = {}
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v, prompt_len=prompt_len, decode_len=decode_len, batch_size=batch_size)


if __name__ == "__main__":
    set_global_func()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt-len", type=int, default=512)
    parser.add_argument("-d", "--decode-len", type=int, default=8)
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    args = parser.parse_args()
    prompt_len = args.prompt_len
    decode_len = args.decode_len
    batch_size = args.batch_size
    for rope_mode in [RopeMode.NONE, RopeMode.NORMAL]:
        cache = create_kv_cache(rope_mode)
        test_paged_attention_kv_cache_prefill_and_decode((cache, rope_mode), prompt_len=prompt_len, decode_len=decode_len, batch_size=batch_size)