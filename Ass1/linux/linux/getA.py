from CMO_A1 import f4 as f_4
import numpy as np
def f4(x):
    return f_4(20466, x)


# Numerical gradient approximation using finite differences
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_step_forward = np.copy(x)
        x_step_backward = np.copy(x)
        x_step_forward[i] += h
        x_step_backward[i] -= h
        
        # Use only the function value, not the gradient
        fx_step_forward, _ = f(x_step_forward)
        fx_step_backward, _ = f(x_step_backward)
        
        grad[i] = (fx_step_forward - fx_step_backward) / (2 * h)
    return grad

# Numerical Hessian approximation using finite differences
def numerical_hessian(f, x, h=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))
    grad = numerical_gradient(f, x)
    for i in range(n):
        for j in range(n):
            if i == j:
                x_step_forward = np.copy(x)
                x_step_forward[i] += h
                grad_step_forward = numerical_gradient(f, x_step_forward)
                hessian[i, j] = (grad_step_forward[i] - grad[i]) / h
            else:
                x_step_forward_i = np.copy(x)
                x_step_forward_j = np.copy(x)
                x_step_forward_i[i] += h
                x_step_forward_j[j] += h
                grad_step_forward_i = numerical_gradient(f, x_step_forward_i)
                grad_step_forward_j = numerical_gradient(f, x_step_forward_j)
                hessian[i, j] = (grad_step_forward_i[j] - grad[j]) / h
    return hessian
x = [0.0, 0.0, 0.0, 0.0, 0.0]
print("Second Derivative of f4 at x=0:", numerical_hessian(f4, x))

