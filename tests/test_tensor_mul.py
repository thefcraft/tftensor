import unittest
from autograd.tensor import Tensor

class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor([1, 2, 3], required_grad=True)
        t2 = Tensor([4, 5, 6], required_grad=True)
        
        t3 = t1 * t2
        assert t3.data.to_list() == [4, 10, 18]
        t3.backward(Tensor([-1,-2,-3]))
        assert t1.grad.data.to_list() == [-4,-10,-18]
        assert t2.grad.data.to_list() == [-1,-4,-9]
        
        
        t1 *= 2
        assert t1.grad is None
        assert t1.data.to_list() == [2, 4, 6]
    
    def test_broadcast_mul(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], required_grad=True) # (2, 3)
        t2 = Tensor([7,8,9], required_grad=True) # (3)
        
        t3 = t1 * t2 #(2, 3)
        assert t3.data.to_list() == [[7, 16, 27], [28, 40, 54]]
        t3.backward(Tensor([[1,1,1], [1,1,1]]))
        assert t1.grad.data.to_list() == [[7,8,9], [7,8,9]]
        assert t2.grad.data.to_list() == [5, 7, 9] # [1, 2, 3] * [1,1,1] + [4,5,6] * [1,1,1]
    
    def test_broadcast_mul2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], required_grad=True) # (2, 3)
        t2 = Tensor([[7,8,9]], required_grad=True) # (1, 3)
        
        t3 = t2 * t1 #(2, 3)
        assert t3.data.to_list() == [[7, 16, 27], [28, 40, 54]]
        t3.backward(Tensor([[1,1,1], [1,1,1]]))
        assert t1.grad.data.to_list() == [[7,8,9], [7,8,9]]
        assert t2.grad.data.to_list() == [[5,7,9]]