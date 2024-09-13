import numpy as np
from CMO_A1 import f4 as f_4
def f4(SrNo, x):
    return f_4(SrNo, x)

# Inexact Line Search with Wolfe Conditions
def InExactLineSearch(c1, c2, gamma, initialx, max_iters=1000):
    x = initialx
    for k in range(max_iters):
        fx, gradfx = f4(20466, x)  # Assuming Sr.No = 20466
        pk = -gradfx  # Descent direction is the negative of the gradient
        
        # Initialize step size
        alpha_k = 1.0
        
        # Define Armijo condition and Curvature condition checks
        def armijo_condition(alpha):
            return f4(20466, x + alpha * pk)[0] <= fx + c1 * alpha * np.dot(pk.T, gradfx)
        
        def curvature_condition(alpha):
            return -np.dot(pk.T, f4(20466, x + alpha * pk)[1]) <= -c2 * np.dot(pk.T, gradfx)
        
        # Wolfe conditions loop to adjust alpha
        while not (armijo_condition(alpha_k) and curvature_condition(alpha_k)):
            alpha_k *= gamma  # Reduce alpha by a factor of gamma
        
        # Update x using the adjusted alpha
        x_next = x + alpha_k * pk
        x = x_next  # Move to the next point

    # Final function value after iterations
    fx_star, _ = f4(20466, x)
    return x, fx_star

# Initialization
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Set parameters for Wolfe conditions
c1 = 0.1
c2 = 0.9  # c1 + c2 = 1
gamma = 0.5  # Alpha reduction factor

# Run the inexact line search
optimized_x_star, fx_star = InExactLineSearch(c1, c2, gamma, x0)

# Output the optimized x value and function value at the final iteration
print("Optimized x*:", optimized_x_star)
print("Function value f(x*):", fx_star)
