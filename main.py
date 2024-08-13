from autograd import Tensor, Parameter, Module, SGD
from autograd.tensor import tensor_sum
from autograd import basetensor as tf
from autograd.function import tanh

class Model(Module):
    def __init__(self) -> None:
        self.w = Parameter(3, 1)
        self.b = Parameter(1)
    def predict(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b


x_data = Tensor(tf.tensor.randn([100, 3], seed=42, dtype=tf.float32))
coef = Tensor(tf.tensor.from_list([[-1], [+3], [-2]], dtype=tf.float32))

y_data = (x_data @ coef) + 5 + tf.tensor.randn([100, 1], seed=1, dtype=tf.float32)


# w = Tensor(tf.tensor.randn([3, 1], seed=6, dtype=tf.float32), required_grad=True)
# b = Tensor(tf.tensor.randn([1], seed=7, dtype=tf.float32), required_grad=True)
w = Parameter(3, 1)
b = Parameter(1)
lr = 0.002
batch_size = 20
model = Model()
optimizer = SGD(model.parameters(), lr) # ,
for epoch in range(100):
    # w.zero_grad() # 34.09299850463867 why a little difference here ?
    # b.zero_grad() 
    # predicted = x_data @ w + b
    # errors = predicted - y_data
    # loss = (errors * errors).sum()
    # loss.backward()
    # w-=w.grad*lr
    # b-=b.grad*lr
    # print(epoch, loss.data[0])


    epoch_loss = 0.0
    for start in range(0, 100, batch_size): # 34.29061841964722
        end = start + batch_size
        assert end <= x_data.shape[0]
        model.zero_grad()
        
        # w.zero_grad()
        # b.zero_grad() 
        inputs = x_data[start:end]
        # predicted = inputs @ w + b
        predicted = model.predict(inputs)
        errors = predicted - y_data[start:end]
        loss = (errors * errors).sum()
        loss.backward()
        
        epoch_loss += loss.data[0]
        optimizer.step()
        # w-=w.grad*lr
        # b-=b.grad*lr
        # model.w -= model.w.grad * lr
        # model.b -= model.b.grad * lr
    print(epoch, epoch_loss)

