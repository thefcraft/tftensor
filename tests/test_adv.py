import unittest
from autograd.tensor import Tensor, float64

class TestTensorAdv(unittest.TestCase):
    def test_simple_gradient_descent(self):

        x = Tensor([10, -10, 10, -5, 6, 3, 1], required_grad=True, dtype=float64)

        for i in range(100):
            x.zero_grad()
            sum_of_squares = (x * x).sum()
            sum_of_squares.backward()

            delta_x = 0.1 * x.grad
            x -= delta_x # set them to None as we use inplace operation
            assert x is not None
            # x = Tensor(x.data - delta_x.data, required_grad=True)
    

        assert(sum_of_squares.data.to_list()[0] == 2.4054223063357045e-17)