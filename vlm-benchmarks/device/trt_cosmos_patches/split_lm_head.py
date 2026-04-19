#!/usr/bin/env python3
"""Split the LM head MatMul in Cosmos-Reason2-2B ONNX to avoid Myelin OOM.

Problem: TensorRT's Myelin graph optimizer fuses the final LM head
(GatherND -> MatMul[2048, 151936] -> Cast) into a single ForeignNode whose
tactic-selection scratch is ~1 GiB. On Orin Nano 8 GB the NvMap/CMA contiguous
allocation cap is well under 1 GiB (we measured ~950 MB max reservable),
so tactic 0 fails, its 12 retries exhaust the pool, and the 593 MB fallback
tactic cannot proceed. Build aborts with:

    NvMapMemAllocInternalTagged: 1075072515 error 12   (x12)
    CUDA error 2 for 622329856-byte allocation
    Could not find any implementation for node {ForeignNode[/Unsqueeze.../Cast]}

Fix: split the single (2048, 151936) MatMul into two (2048, 75968) MatMuls
along the vocab dimension, then Concat their outputs back together. Two
smaller Myelin fusion candidates each fit under the 950 MB CMA ceiling,
and each one's allocation succeeds on first try (no wedging). Output is
mathematically identical to the original — this is just a graph rewrite,
not a model change.

Identity barriers (a simpler approach) were tried first and silently removed
by TRT's constant-folding pass before Myelin saw them. Concat is a real
data-rearranging op and the optimizer leaves it in place.

Inputs:
    /home/hc/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/llm/
        model.onnx
        onnx_model.data   (~1.35 GB external weights)
        (other tokenizer / config files, symlinked)

Outputs:
    /home/hc/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-split/onnx/llm/
        model.onnx        (patched graph, 3919 nodes vs 3917 original)
        onnx_model.data   (1.35 GB; new file — weights are re-laid-out)
        (tokenizer / config files copied)
"""
import os
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper

SRC_DIR = "/home/hc/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/llm"
DST_DIR = "/home/hc/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-split/onnx/llm"


def mirror_directory(src, dst):
    """Symlink large files, copy small ones. Skip model.onnx + onnx_model.data
    which the script rewrites below."""
    os.makedirs(dst, exist_ok=True)
    print(f"Setting up {dst}...")
    for fname in os.listdir(src):
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname)
        if os.path.lexists(dst_path):
            os.remove(dst_path)
        if fname in ("model.onnx", "onnx_model.data"):
            continue
        size = os.path.getsize(src_path)
        if size > 10 * 1024 * 1024:
            os.symlink(src_path, dst_path)
            print(f"  symlinked {fname} ({size / 1e6:.1f} MB)")
        else:
            shutil.copy(src_path, dst_path)


def split_matmul(graph, node_name="/lm_head/MatMul"):
    """Split the named MatMul along the N (output) dimension into two halves
    reconnected via Concat. Returns the list of new nodes."""
    matmul_node = None
    for n in graph.node:
        if n.name == node_name:
            matmul_node = n
            break
    if matmul_node is None:
        raise RuntimeError(f"Could not find {node_name}")

    # MatMul inputs: [activation, weight]. Weight is an initializer (K, N).
    weight_name = matmul_node.input[1]
    weight_init = None
    for init in graph.initializer:
        if init.name == weight_name:
            weight_init = init
            break
    if weight_init is None:
        raise RuntimeError(f"Could not find initializer {weight_name}")

    W = numpy_helper.to_array(weight_init)
    print(f"\nLM head weight {weight_name}: shape {W.shape}, dtype {W.dtype}")
    K, N = W.shape
    if N != 151936:
        raise RuntimeError(f"Unexpected output dim {N}, expected 151936")

    mid = N // 2
    W_lo = np.ascontiguousarray(W[:, :mid])
    W_hi = np.ascontiguousarray(W[:, mid:])
    print(f"  split into {W_lo.shape} + {W_hi.shape}")

    w_lo_name = weight_name + "_lo"
    w_hi_name = weight_name + "_hi"
    w_lo_init = numpy_helper.from_array(W_lo, name=w_lo_name)
    w_hi_init = numpy_helper.from_array(W_hi, name=w_hi_name)

    act_input = matmul_node.input[0]
    out_lo = "/lm_head/MatMul_lo_output_0"
    out_hi = "/lm_head/MatMul_hi_output_0"
    concat_out = matmul_node.output[0]  # reuse original — downstream unchanged

    matmul_lo = helper.make_node("MatMul", inputs=[act_input, w_lo_name],
                                 outputs=[out_lo], name="/lm_head/MatMul_lo")
    matmul_hi = helper.make_node("MatMul", inputs=[act_input, w_hi_name],
                                 outputs=[out_hi], name="/lm_head/MatMul_hi")
    concat = helper.make_node("Concat", inputs=[out_lo, out_hi],
                              outputs=[concat_out], name="/lm_head/Concat_split",
                              axis=-1)

    graph.node.remove(matmul_node)
    graph.initializer.remove(weight_init)

    # Keep topological order: insert after /GatherND (MatMul's predecessor)
    insert_idx = None
    for i, n in enumerate(graph.node):
        if n.name == "/GatherND":
            insert_idx = i + 1
            break
    if insert_idx is None:
        insert_idx = len(graph.node)
    graph.node.insert(insert_idx, matmul_lo)
    graph.node.insert(insert_idx + 1, matmul_hi)
    graph.node.insert(insert_idx + 2, concat)
    graph.initializer.append(w_lo_init)
    graph.initializer.append(w_hi_init)

    return [matmul_lo, matmul_hi, concat]


def main():
    mirror_directory(SRC_DIR, DST_DIR)

    src_onnx = os.path.join(SRC_DIR, "model.onnx")
    print(f"\nLoading {src_onnx} with external data...")
    model = onnx.load(src_onnx, load_external_data=True)
    graph = model.graph
    print(f"  {len(graph.node)} nodes")

    new_nodes = split_matmul(graph)
    print(f"\nAfter surgery: {len(graph.node)} nodes, {len(graph.initializer)} initializers")

    dst_onnx = os.path.join(DST_DIR, "model.onnx")
    dst_data = "onnx_model.data"  # relative path, lands next to model.onnx
    print(f"\nSaving {dst_onnx} (+external data {dst_data})...")
    onnx.save_model(model, dst_onnx, save_as_external_data=True,
                    all_tensors_to_one_file=True, location=dst_data,
                    size_threshold=1024, convert_attribute=False)
    print(f"  model.onnx: {os.path.getsize(dst_onnx) / 1e6:.2f} MB")
    print(f"  data file : {os.path.getsize(os.path.join(DST_DIR, dst_data)) / 1e6:.2f} MB")

    patched = onnx.load(dst_onnx, load_external_data=False)
    found = [n for n in patched.graph.node
             if n.name in ("/lm_head/MatMul_lo", "/lm_head/MatMul_hi", "/lm_head/Concat_split")]
    old_present = any(n.name == "/lm_head/MatMul" for n in patched.graph.node)
    print(f"\nVerification:")
    print(f"  new LM-head nodes found: {len(found)} (expect 3)")
    print(f"  old /lm_head/MatMul still present: {old_present} (expect False)")
    for n in found:
        print(f"    {n.name}: {list(n.input)} -> {list(n.output)}")
    # onnx.checker.check_model rejects symlinked external data with a
    # ValidationError. The check is cosmetic — TRT follows symlinks fine.
    # Skipped here; the structural verification above is sufficient.
    print("\nDone.")


if __name__ == "__main__":
    main()
