import numpy as np
import tvm
import tvm.testing
from tvm import relax, te, tir

token_budget = 100

def _attach_argtopk_func(token_budget=100):
    bb = relax.BlockBuilder()
    batch_size = tir.SizeVar("batch_size", "int64")
    seq_len = tir.SizeVar("seq_len", "int64")
    num_qo_head = tir.SizeVar("num_qo_head", "int64")
    qk_product = relax.Var("qk_product", relax.TensorStructInfo((seq_len, batch_size, num_qo_head), "float32"))
    with bb.function("argtopk_qk_product", [qk_product]):
        with bb.dataflow():
            topk_values, topk_indices = bb.emit(relax.op.topk(qk_product, k=token_budget, axis=0, dtype="int32"))
            output = bb.emit_output((topk_values, topk_indices))
        gv = bb.emit_func_output(output)
    return bb.finalize()


def main():
    mod = _attach_argtopk_func(token_budget = token_budget)
    # print(mod.script())

    device = tvm.cuda()
    target = tvm.target.Target.from_device(device)
    executable = relax.build(
        mod, target="cuda", pipeline=relax.backend.cuda.get_default_pipeline(target)
    )

    vm = relax.VirtualMachine(executable, device)
    func = vm["argtopk_qk_product"]  # <=======
    batch_size = 10
    seq_len = 10000
    num_qo_head = 12
    # rng = np.random.default_rng(0)
    qk_product = np.random.randn(seq_len, batch_size, num_qo_head).astype("float32")
    qk_product_tvm = tvm.nd.array(qk_product, device=device)
    topk_values, topk_indices = func(qk_product_tvm)
    # print(topk_values.numpy()[0, 0, 0])
    # print(topk_indices.numpy().shape)
    actual_topk_ind = np.argsort(qk_product, axis=0)[::-1][:token_budget]
    # print(actual_topk_ind.shape)
    tvm.testing.assert_allclose(
                topk_indices.numpy(),
                actual_topk_ind,
                rtol=1e-3,
                atol=1e-3,
            )


if __name__ == "__main__":
    main()