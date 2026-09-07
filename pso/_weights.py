import collections
import torch
import torch.nn as nn


class ParameterCodec:
    """
    Private parameter encoder/decoder for flattening and reconstructing PyTorch nn.Module parameters.
    """

    def __init__(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")

        params = list(model.named_parameters())
        if not params:
            raise ValueError("model contains zero trainable parameters")

        self.names: list[str] = []
        self.shapes: list[tuple[int, ...]] = []
        self.numels: list[int] = []
        dtypes: set[torch.dtype] = set()

        total_size = 0
        for name, p in params:
            if not p.dtype.is_floating_point:
                raise ValueError(
                    f"Parameter '{name}' has non-floating dtype {p.dtype}. Only floating-point parameters are supported."
                )
            self.names.append(name)
            self.shapes.append(tuple(p.shape))
            n = p.numel()
            self.numels.append(n)
            total_size += n
            dtypes.add(p.dtype)

        if len(dtypes) > 1:
            raise ValueError(f"Mixed parameter dtypes found in model: {dtypes}")

        self.dtype: torch.dtype = next(iter(dtypes))
        self.size: int = total_size

    def encode(self, model: nn.Module) -> torch.Tensor:
        """
        Flattens model parameters into a single detached 1D torch.Tensor.
        """
        params = [p for _, p in model.named_parameters()]
        return nn.utils.parameters_to_vector(params).detach()

    def apply_vector(self, vector: torch.Tensor, model: nn.Module) -> None:
        """
        Applies a validated 1D parameter vector to model.parameters() in-place without storage aliasing.
        """
        if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
            raise ValueError("vector must be a 1D torch.Tensor")
        if vector.numel() != self.size:
            raise ValueError(
                f"Vector size {vector.numel()} does not match codec size {self.size}"
            )
        if vector.dtype != self.dtype:
            raise ValueError(
                f"Vector dtype {vector.dtype} does not match codec dtype {self.dtype}"
            )

        params = [p for _, p in model.named_parameters()]
        if params:
            target_device = params[0].device
            if vector.device != target_device:
                vector = vector.to(target_device)

        with torch.no_grad():
            offset = 0
            for p, shape, numel in zip(params, self.shapes, self.numels):
                p.copy_(vector[offset : offset + numel].reshape(shape))
                offset += numel

    def to_state_dict(
        self, vector: torch.Tensor, eval_model: nn.Module
    ) -> collections.OrderedDict[str, torch.Tensor]:
        """
        Applies vector to eval_model and returns a defensive CPU-cloned ordered state dict including buffers.
        """
        self.apply_vector(vector, eval_model)
        state_dict = eval_model.state_dict()
        res = collections.OrderedDict()
        for k, v in state_dict.items():
            res[k] = v.detach().cpu().clone()
        return res
