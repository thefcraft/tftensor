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
        
        self.shape = tuple(self.data.shape)
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
                grad = Tensor(BaseTensor.ones([1], dtype=self.data.dtype))
            else: 
                raise RuntimeError("grad must be specified for non-0-tensor")
                
        assert grad.data is not None, f"grad.data is none {self}"
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

def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    required_grad = t1.required_grad or t2.required_grad
    depends_on: List[Dependency] = []
    if t1.required_grad:
        def grad_fn1(grad: BaseTensor)->BaseTensor:
            # [1, 2, 3] + [4, 5, 6] => [5, 7, 9]
            # Sum out added dims i.e (3) => (1, 3)
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(dim=0)
            # Sum across broadcasted (but non added dims) i.e (1, 3) => (2, 3)
            for i, (dim, dimg) in enumerate(zip(t1.shape, grad.shape)):
                if dim == 1 and dimg != 1:
                    grad = grad.sum(dim=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t1, grad_fn1))
    if t2.required_grad:
        def grad_fn2(grad: BaseTensor)->BaseTensor:
            # handle broadcasting properly
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(dim=0)
            for i, (dim, dimg) in enumerate(zip(t2.shape, grad.shape)):
                if dim == 1 and dimg != 1:
                    grad = grad.sum(dim=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t2, grad_fn2))
        
    return Tensor(data,
                  required_grad,
                  depends_on)
