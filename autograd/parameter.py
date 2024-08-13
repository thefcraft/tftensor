from .tensor import Tensor, BaseTensor

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = BaseTensor.randn(shape)
        super().__init__(data, required_grad=True)
    