from . import Parameter, Module
class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        # Initialize weights and biases
        self.in_features = in_features
        self.out_features = out_features
        
        # Weights and biases are initialized here
        self.weight = Parameter(in_features, out_features) # out_features x in_features
        self.bias = Parameter(out_features) # out_features

    def forward(self, x):
        # Linear transformation: y = x * W^T + b
        return x@self.weight+self.bias