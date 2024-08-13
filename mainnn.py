from autograd import Tensor, Parameter, Module, SGD, Linear
from autograd.tensor import tensor_sum
from autograd import basetensor as tf
from autograd.function import relu

class Model(Module):
    def __init__(self) -> None:
        self.fc1 = Linear(3, 20)
        self.fc2 = Linear(20, 3)
        self.fc3 = Linear(3, 1)
    def forward(self, x: Tensor) -> Tensor:
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return relu(self.fc3(x))


x_data = Tensor(tf.tensor.randn([100, 3], seed=42, dtype=tf.float32))
coef = Tensor(tf.tensor.from_list([[-1], [+3], [-2]], dtype=tf.float32))

y_data = (x_data @ coef) + 5 + tf.tensor.randn([100, 1], seed=1, dtype=tf.float32)

lr = 0.0002
batch_size = 20
model = Model()
optm = SGD(model.parameters(), lr) # ,
for epoch in range(100):
    optm.zero_grad()
    predicted = model(x_data)
    errors = predicted - y_data
    loss = (errors * errors).sum()
    loss.backward()
    optm.step()
    print(epoch, loss.data[0])