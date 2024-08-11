from .basetensor import tensor as BaseTensor
from typing import List, NewType, Union, Any, Optional, Tuple, NamedTuple, Callable, Type

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[['BaseTensor'], 'BaseTensor']

Tensorable = Union[int, float, list, BaseTensor]

def ensure_tensor(tensorable: Tensorable) -> BaseTensor:
    if isinstance(tensorable, BaseTensor): return tensorable
    elif isinstance(tensorable, list):
        return BaseTensor.from_list(tensorable)
    elif isinstance(tensorable, float):
        return BaseTensor.full([1], tensorable)
    elif isinstance(tensorable, int):
        return BaseTensor.full([1], tensorable)

class Tensor:
    def __init__(self,
                 data: Tensorable,
                 required_grad: bool = False,
                 depends_on: List[Dependency] = None)->None:
        self.data: BaseTensor = ensure_tensor(data)
        self.required_grad = required_grad
        self.depends_on = depends_on or []
        
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        
        self.grad:Optional['Tensor'] = None
        if self.required_grad: 
            self.zero_grad()
    def zero_grad(self):
        self.grad = Tensor(BaseTensor.zeros_like(self.data))
        
    def __repr__(self) -> str:
        return f"Tensor({self.data.reprstr(spacing_size=7)}, required_grad={self.required_grad})"
    
    def backward(self, grad: Optional['Tensor'] = None)->None:
        assert self.required_grad, "called backward on non-required-grad tensor"
        if grad is None:
            if self.ndim == 1:
                grad = Tensor(1)
            else: 
                raise RuntimeError("grad must be specified for non-0-tensor")
                
            
        self.grad.data += grad.data
        
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))
    
    def sum(self) -> 'Tensor':
        return tensor_sum(self)
    

def tensor_sum(t: Tensor) -> Tensor:
    """
    takes a tensor and return 0-tensor sum
    """
    data = t.data.sum()
    required_grad = t.required_grad
    if required_grad:
        def grad_fn(grad: BaseTensor)->BaseTensor:
            """
            grad is nessarily a 0-tensor, so each element contribute that much
            """
            return grad * BaseTensor.ones_like(t.data)
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data,
                  required_grad,
                  depends_on)