import unittest
from autograd.tensor import Tensor

class TestTensorMatmul(unittest.TestCase):
    def test_simple_matmul(self):
        # (3, 2)
        t1 = Tensor([[1, 2], [2, 3], [5, 6]], required_grad=True)
        # (2, 1)
        t2 = Tensor([[10], [20]], required_grad=True)
        
        t3 = t1 @ t2
        assert t3.data.to_list() == [[50], [80], [170]]
        grad = Tensor([[-1],[-2],[-3]])
        t3.backward(grad)
        
        assert t1.grad.data.to_list() == (grad.data @ t2.data.T).to_list()
        assert t2.grad.data.to_list() == (t1.data.T @ grad.data).to_list()
        
        
        # t1 *= 2
        # assert t1.grad is None
        # assert t1.data.to_list() == [2, 4, 6]
    