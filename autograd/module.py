from . import Parameter, Tensor
from typing import Iterator
import inspect
class Module:
    def parameters(self)->Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
    def zero_grad(self):
        for parameter in self.parameters():
            if parameter.required_grad:
                parameter.zero_grad()
    def forward(self, *args, **kwargs): raise NotImplementedError("Forward method not implemented yet")
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)