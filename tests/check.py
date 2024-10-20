

from typing import Any
from minitorch import MathTestVariable, Tensor, grad_check, tensor, grad_central_difference
import random
import numpy as np

fn = MathTestVariable.

tensor_fn = fn
t1 = tensor([[1.0, -2.0, 3.0]])
t2 = tensor([1.0, -2.0, 3.0])
out = tensor_fn(t1)
out2=tensor_fn(t2)
out.sum().backward()
out2.sum().backward()

collated = out.sum()

grad_check(tensor_fn, t1)   

def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, x, arg=i, ind=ind)
        #assert x.grad is not None, "No gradient computed"
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )



#grad_check(tensor_fn, t1)
