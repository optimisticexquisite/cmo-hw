import numpy as np
from CMO_A1 import f4 as f_4
def f4(SrNo, x):
    return f_4(SrNo, x)

# Gradient Descent with constant step size
def ConstantGradientDescent(alpha, initialx, tolerance=1e-6, max_iters=1000):
    x = initialx
    for i in range(max_iters):
        fx, gradfx = f4(20466, x)  # Assuming Sr.No = 20466
        pk = -gradfx  # Descent direction is the negative of the gradient
        x_next = x + alpha * pk  # Update rule

        # Check for convergence (stop if the gradient is smaller than tolerance)
        if np.linalg.norm(gradfx) < tolerance:
            print(f"Convergence reached after {i+1} iterations")
            break
        
        x = x_next  # Move to the next point

    return x

# Initialization
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Set step size alpha
alpha = 1e-5

# Run gradient descent
optimized_x = ConstantGradientDescent(alpha, x0)

# Output the optimized x value
print("Optimized x:", optimized_x)
