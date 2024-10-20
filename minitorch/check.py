from project.run_tensor import Network, Linear
from minitorch.tensor import Tensor, TensorData, tensor

nn = Network(2)
# Sample data
input_data = tensor([1, 2])

# Perform a forward pass
output = nn.forward(input_data)

# Print the output
print("Output of the forward pass:", output)
