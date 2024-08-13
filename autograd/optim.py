from . import Module, Parameter
from typing import List


class SGD:
    def __init__(self, params, lr:float = 0.001) -> None:
        self.lr = lr
        self.params: List[Parameter] = list(params)
    def step(self)->None:
        for parameter in self.params:
            if parameter.required_grad:
                parameter -= parameter.grad * self.lr
            
    def zero_grad(self)->None:
        for parameter in self.params:
            if parameter.required_grad:
                parameter.zero_grad()