import numpy as np
from CMO_A1 import f4 as f_4
def f4(SrNo, x):
    return f_4(SrNo, x)

# Gradient Descent with diminishing step size
def DiminishingGradientDescent(InitialAlpha, initialx, max_iters=10000):
    x = initialx
    for k in range(max_iters):
        fx, gradfx = f4(20466, x)  # Assuming Sr.No = 20466
        pk = -gradfx  # Descent direction is the negative of the gradient
        alpha_k = InitialAlpha / (k + 1)  # Diminishing step size
        
        x_next = x + alpha_k * pk  # Update rule

        x = x_next  # Move to the next point

    fx_T, _ = f4(20466, x)  # Final function value after T iterations
    return x, fx_T

# Initialization
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Set initial step size alpha0
InitialAlpha = 1e-3

# Run diminishing gradient descent
optimized_x_T, fx_T = DiminishingGradientDescent(InitialAlpha, x0)

# Output the optimized x value and function value at T
print("Optimized x at iteration T=10000:", optimized_x_T)
print("Function value f(xT) at iteration T=10000:", fx_T)
