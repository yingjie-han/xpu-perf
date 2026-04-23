from core.ops.vector_norm_ops import SoftmaxOp


OP_MAPPING = {}


class INTELSoftmaxTorchOp(SoftmaxOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._provider = "torch"


# Keep both aliases for compatibility with provider filters.
OP_MAPPING["torch"] = INTELSoftmaxTorchOp
