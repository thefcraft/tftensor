from autograd.tensor import Tensor, tensor_sum
from autograd import basetensor as tf

x_data = Tensor(tf.tensor.randn([100, 3], seed=42, dtype=tf.float32))
coef = Tensor(tf.tensor.from_list([[-1], [+3], [-2]], dtype=tf.float32))

y_data = (x_data @ coef) + 5 + tf.tensor.randn([100, 1], dtype=tf.float32)


w = Tensor(tf.tensor.randn([3, 1], seed=6, dtype=tf.float32), required_grad=True)
b = Tensor(tf.tensor.randn([1], seed=7, dtype=tf.float32), required_grad=True)
lr = 0.002
for epoch in range(100):
    w.zero_grad()
    b.zero_grad() 
    predicted = x_data @ w + b
    errors = predicted - y_data
    loss = (errors * errors).sum()
    loss.backward()
    
    w-=w.grad*lr
    b-=b.grad*lr
    print(epoch, loss)

