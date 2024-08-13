from .basetensor import  float32, float64, tensor as BaseTensor, typetensor as TypeBaseTensor
from typing import List, NewType, Union, Any, Optional, Tuple, NamedTuple, Callable, Type

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[['BaseTensor'], 'BaseTensor']

baseTensorable = Union[int, float, list, TypeBaseTensor]

def ensure_basetensor(tensorable: baseTensorable, dtype) -> BaseTensor:
    if isinstance(tensorable, BaseTensor): return tensorable
    elif isinstance(tensorable, list):
        return BaseTensor.from_list(tensorable, dtype)
    elif isinstance(tensorable, float):
        return BaseTensor.full([1], tensorable, dtype)
    elif isinstance(tensorable, int):
        return BaseTensor.full([1], tensorable, dtype)

Tensorable = Union['Tensor', int, float, list, TypeBaseTensor]

def ensure_tensor(tensorable: Tensorable, dtype) -> "Tensor":
    if isinstance(tensorable, Tensor): 
        return tensorable
    return Tensor(
        data= ensure_basetensor(tensorable, dtype),
        required_grad=False
    )
    
class Tensor:
    def __init__(self,
                 data: baseTensorable,
                 required_grad: bool = False,
                 depends_on: List[Dependency] = None,
                 dtype=None)->None:
        self.__data: BaseTensor = ensure_basetensor(data, dtype or float32)
        self.required_grad = required_grad
        self.depends_on = depends_on or []
        
        self.shape = tuple(self.__data.shape)
        self.ndim = self.__data.ndim
        self.dtype = self.__data.dtype
        self.grad:Optional['Tensor'] = None
        if self.required_grad: 
            self.zero_grad()
    def zero_grad(self):
        self.grad = Tensor(BaseTensor.zeros_like(self.__data))
        
    def __repr__(self) -> str:
        return f"Tensor({self.__data.reprstr(spacing_size=7)}, required_grad={self.required_grad})"
    
    @property
    def data(self)->BaseTensor:
        return self.__data
    @data.setter
    def data(self, new_data:BaseTensor)->None:
        self.__data = new_data
        # Invalidate the gradients you can't use this grad anymore...
        self.grad = None
    def __matmul__(self, other)->"Tensor":
        return _matmul(self, other)
    def __add__(self, other:Tensorable)->"Tensor":
        return _add(self, ensure_tensor(other, dtype=self.dtype))
    def __radd__(self, other:Tensorable)->"Tensor":
        return _add(ensure_tensor(other, dtype=self.dtype), self)
    def __iadd__(self, other:Tensorable)->"Tensor":
        self.__data += ensure_tensor(other, dtype=self.dtype).__data
        # Invalidate the gradients you can't use this grad anymore...
        self.grad = None
        return self
    def __sub__(self, other:Tensorable)->"Tensor":
        return _sub(self, ensure_tensor(other, dtype=self.dtype))
    def __rsub__(self, other:Tensorable)->"Tensor":
        return _sub(ensure_tensor(other, dtype=self.dtype), self)
    def __isub__(self, other:Tensorable)->"Tensor":
        self.__data -= ensure_tensor(other, dtype=self.dtype).__data
        # Invalidate the gradients you can't use this grad anymore...
        self.grad = None
        return self
    def __mul__(self, other:Tensorable)->"Tensor":
        return _mul(self, ensure_tensor(other, dtype=self.dtype))
    def __rmul__(self, other:Tensorable)->"Tensor":
        return _mul(ensure_tensor(other, dtype=self.dtype), self)
    def __imul__(self, other:Tensorable)->"Tensor":
        self.__data *= ensure_tensor(other, dtype=self.dtype).__data
        # Invalidate the gradients you can't use this grad anymore...
        self.grad = None
        return self
    def __neg__(self)->"Tensor":
        return _neg(self)
    def __getitem__(self, index): 
        return _slice(self, index)
    def backward(self, grad: Optional['Tensor'] = None)->None:
        assert self.required_grad, "called backward on non-required-grad tensor"
        if grad is None:
            if self.ndim == 1:
                grad = Tensor(BaseTensor.ones([1], dtype=self.__data.dtype))
            else: 
                raise RuntimeError("grad must be specified for non-0-tensor")
                
        assert grad.__data is not None, f"grad.data is none {self}"
        self.grad.__data += grad.__data
        
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.__data)
            dependency.tensor.backward(Tensor(backward_grad))
    
    def sum(self) -> 'Tensor':
        return tensor_sum(self)
    
from .tensor_ops import _add, _sub, _mul, _neg, tensor_sum, _matmul, _slice