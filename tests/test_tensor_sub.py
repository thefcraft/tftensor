import unittest
from autograd.tensor import Tensor

class TestTensorSub(unittest.TestCase):
    def test_simple_neg(self):
        t1 = Tensor([8, 2, 3], required_grad=True)
        t3 = -t1
        t3.backward(Tensor([-1,-2,-3]))
        assert t1.grad.data.to_list() == [1,2,3]
        
    def test_simple_sub(self):
        t1 = Tensor([1, 2, 3], required_grad=True)
        t2 = Tensor([4, 5, 6], required_grad=True)
        
        t3 = t1-t2
        assert t3.data.to_list() == [-3,-3,-3]
        t3.backward(Tensor([-1,-2,-3]))
        assert t1.grad.data.to_list() == [-1,-2,-3]
        assert t2.grad.data.to_list() == [1,2,3]
        
        t1 -= 1
        assert t1.grad is None
        assert t1.data.to_list() == [0, 1, 2]
    
    def test_broadcast_sub(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], required_grad=True) # (2, 3)
        t2 = Tensor([7,8,9], required_grad=True) # (3)
        
        t3 = t1-t2 #(2, 3)
        assert t3.data.to_list() == [[-6, -6,-6], [-3,-3,-3]]
        t3.backward(Tensor([[1,1,1], [1,1,1]]))
        assert t1.grad.data.to_list() == [[1,1,1], [1,1,1]]
        assert t2.grad.data.to_list() == [-2, -2, -2]
    
    
    def test_broadcast_sub2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], required_grad=True) # (2, 3)
        t2 = Tensor([[7,8,9]], required_grad=True) # (1, 3)
        
        t3 = t1-t2 #(2, 3)
        assert t3.data.to_list() == [[-6, -6,-6], [-3,-3,-3]]
        t3.backward(Tensor([[1,1,1], [1,1,1]]))
        assert t1.grad.data.to_list() == [[1,1,1], [1,1,1]]
        assert t2.grad.data.to_list() == [[-2, -2, -2]]