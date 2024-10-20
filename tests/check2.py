
from typing import Any
from minitorch import MathTestVariable, Tensor, grad_check, tensor, grad_central_difference
import random
import numpy as np




t1 = tensor([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]])

t1.permute( 1,0).sum().backward()

dims = Tensor.make(list(dims), (len(dims),), backend=t1.backend)

[int(dim) for dim in dims.tuple()[0]]