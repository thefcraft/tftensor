import tensor as tf
from tensor import tensor as Tensor

if __name__ == '__main__':
    a = Tensor.randn([3, 4, 5], dtype=tf.float32)#, required_grad=True)
    b = Tensor.randn([3, 1, 5], dtype=tf.float32)#, required_grad=True)
    c = a+b
    # c.backward()
    print(c)
    
    for _ in range(2):
        c = c.sum(dim=0)
    print(c)