from . import Tensor
from .tensor import Dependency
from . import basetensor as tf
from .basetensor import tensor as BaseTensor

def tanh(tensor: Tensor) -> Tensor:
    data = BaseTensor.tanh(tensor.data)  # TODO: tanh
    requires_grad = tensor.required_grad
    if requires_grad:
        def grad_fn(grad: BaseTensor)->BaseTensor:
            return grad * (1 - data * data)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data, requires_grad, depends_on)

def relu(tensor: Tensor) -> Tensor:
    # Compute the forward pass
    data = tf.maximum(0, tensor.data)
    # Determine if we need to compute gradients
    requires_grad = tensor.required_grad
    if requires_grad:
        def grad_fn(grad: BaseTensor) -> BaseTensor:
            # Gradient of ReLU is 1 where data > 0, else 0
            return grad * (tensor.data > 0)
        
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data, requires_grad, depends_on)