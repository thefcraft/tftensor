from .tensor import Tensor, BaseTensor, Dependency
from typing import List, NewType, Union, Any, Optional, TypeVar, Type

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

def _add(t1: Tensor, t2: Tensor) -> Tensor:
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

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    y = a * b
    fn = dL/dy 
    dL/da = dL/dy * (dy/da = b [partial diffn])
    """
    data = t1.data * t2.data
    required_grad = t1.required_grad or t2.required_grad
    depends_on: List[Dependency] = []
    if t1.required_grad:
        def grad_fn1(grad: BaseTensor)->BaseTensor:
            grad = grad * t2.data
            
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
            grad = grad * t1.data
            
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
    
def _neg(t: Tensor)->Tensor:
    data = -t.data
    required_grad = t.required_grad
    if t.required_grad:
        depends_on =[Dependency(t, lambda x: -x)]
    else:
        depends_on = []
    return Tensor(data,
                  required_grad,
                  depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    # return t1 + -t2 one line
    # return _add(t1, neg(t2)) this is simply one line
    data = t1.data - t2.data
    required_grad = t1.required_grad or t2.required_grad
    depends_on: List[Dependency] = []
    if t1.required_grad:
        def grad_fn1(grad: BaseTensor)->BaseTensor:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(dim=0)
            for i, (dim, dimg) in enumerate(zip(t1.shape, grad.shape)):
                if dim == 1 and dimg != 1:
                    grad = grad.sum(dim=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t1, grad_fn1))
    if t2.required_grad:
        def grad_fn2(grad: BaseTensor)->BaseTensor:
            grad *= BaseTensor.full([1], fill_value=-1, dtype=grad.dtype)
            
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

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    t1 = (n1, m1) t2 = (n2, m2) then t1 @ t2 = (n1, m2)
    so grad3 is (n1, m2)
    
    # grad1 (n1, m2) @ (m2, n2) => (n1, n2)
    # grad1 (n1, m1) => (n2, n2) todo
    
    if t3 = t1 @ t2 and grad3 is the gradient of some fn with respect to t3, then
        grad1 = grad @ t2.T
        grad2 = t1.T @ grad
    """
    data = t1.data @ t2.data
    required_grad = t1.required_grad or t2.required_grad
    depends_on: List[Dependency] = []
    if t1.required_grad:
        def grad_fn1(grad: BaseTensor)->BaseTensor:
            return grad @ t2.data.T
            # grad = grad * t2.data
            
            # # Sum out added dims i.e (3) => (1, 3)
            # ndims_added = grad.ndim - t1.data.ndim
            # for _ in range(ndims_added):
            #     grad = grad.sum(dim=0)
            # # Sum across broadcasted (but non added dims) i.e (1, 3) => (2, 3)
            # for i, (dim, dimg) in enumerate(zip(t1.shape, grad.shape)):
            #     if dim == 1 and dimg != 1:
            #         grad = grad.sum(dim=i, keepdims=True)
            # return grad
        depends_on.append(Dependency(t1, grad_fn1))
    if t2.required_grad:
        def grad_fn2(grad: BaseTensor)->BaseTensor:
            return t1.data.T @ grad
            # grad = grad * t1.data
            
            # # handle broadcasting properly
            # ndims_added = grad.ndim - t2.data.ndim
            # for _ in range(ndims_added):
            #     grad = grad.sum(dim=0)
            # for i, (dim, dimg) in enumerate(zip(t2.shape, grad.shape)):
            #     if dim == 1 and dimg != 1:
            #         grad = grad.sum(dim=i, keepdims=True)
            # return grad
        depends_on.append(Dependency(t2, grad_fn2))
        
    return Tensor(data,
                  required_grad,
                  depends_on)
    

def _slice(t: Tensor, idxs)->Tensor:
    data = t.data.__getitem__(idxs)
    required_grad = t.required_grad
    if required_grad:
        def grad_fn(grad: BaseTensor)->BaseTensor:
            bigger_grad = BaseTensor.zeros_like(data)
            bigger_grad.__setitem__(idxs, grad)
            return bigger_grad
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data, required_grad, depends_on)
        
        