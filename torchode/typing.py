from jaxtyping import Bool, Float, Int
from torch import Tensor


def same_dtype(*tensors: Tensor) -> bool:
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:], strict=True):
        if a.dtype != b.dtype:
            return False
    return True


def same_device(*tensors: Tensor) -> bool:
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:], strict=True):
        if a.device != b.device:
            return False
    return True


def same_shape(*tensors: Tensor, dim: int | None = None) -> bool:
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:], strict=True):
        if dim is None:
            if a.shape != b.shape:
                return False
        else:
            if a.shape[dim] != b.shape[dim]:
                return False
    return True


################
# Tensor Types #
################

# Shape- and dtype-annotated aliases for ``torch.Tensor`` built with jaxtyping. They
# document the expected layout of tensors in function signatures. The annotations are
# not enforced at runtime unless a jaxtyping-aware type checker is enabled.

DataTensor = Float[Tensor, "batch feature"]
NormTensor = Float[Tensor, "batch"]
SolutionDataTensor = Float[Tensor, "batch time feature"]
TimeTensor = Float[Tensor, "batch"]
EvaluationTimesTensor = Float[Tensor, "batch time"]
AcceptTensor = Bool[Tensor, "batch"]
StatusTensor = Int[Tensor, "batch"]
InterpTimeTensor = Float[Tensor, "interp_points"]
InterpDataTensor = Float[Tensor, "interp_points feature"]
SampleIndexTensor = Int[Tensor, "interp_points"]
