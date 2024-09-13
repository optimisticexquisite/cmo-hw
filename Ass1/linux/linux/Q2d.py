import numpy as np
import matplotlib.pyplot as plt
from CMO_A1 import f4 as f_4
def f4(SrNo, x):
    A = np.array([[20466, 0, 0, 0, 0], 
                  [0, 2000, 0, 0, 0], 
                  [0, 0, 1000, 0, 0], 
                  [0, 0, 0, 500, 0], 
                  [0, 0, 0, 0, 100]]) 
    fx, gradfx = f_4(SrNo, x)
    return fx, gradfx, A


# Exact Line Search method
def ExactLineSearch(initialx, tolerance=1e-6, max_iters=1000):
    x = initialx
    gradients_norm = []
    f_diff = []
    f_ratio = []
    x_diff = []
    x_ratio = []

    x_history = [x]
    fx_history = []

    for k in range(max_iters):
        fx, gradfx, A = f4(20466, x)
        pk = -gradfx  # Descent direction is the negative of the gradient
        
        # Calculate the exact alpha using the quadratic form
        alpha_k = -np.dot(gradfx.T, pk) / np.dot(pk.T, np.dot(A, pk))
        
        # Update x
        x_next = x + alpha_k * pk
        
        # Store x and f(x)
        x_history.append(x_next)
        fx_history.append(fx)
        
        # Calculate norms and differences for plotting
        gradients_norm.append(np.linalg.norm(gradfx)**2)
        
        if k > 0:
            f_diff.append(fx_history[k-1] - fx)
            f_ratio.append(f_diff[-1] / f_diff[-2] if len(f_diff) > 1 else None)
            x_diff.append(np.linalg.norm(x_history[k] - x_history[-1])**2)
            x_ratio.append(x_diff[-1] / x_diff[-2] if len(x_diff) > 1 else None)

        # Stop if the gradient norm is below the tolerance level
        if np.linalg.norm(gradfx) < tolerance:
            print(f"Convergence reached after {k+1} iterations.")
            break

        x = x_next  # Update for the next iteration
    
    # Final optimized values
    x_star = x_history[-1]
    fx_star = fx_history[-1]
    T = len(x_history) - 1  # Number of iterations

    # Generate the plots
    plot_results(gradients_norm, f_diff, f_ratio, x_diff, x_ratio, T)
    
    return x_star, fx_star, T

# Helper function for plotting
def plot_results(gradients_norm, f_diff, f_ratio, x_diff, x_ratio, T):
    iterations = np.arange(1, T+1)
    gradients_norm.append(np.float64('nan'))
    f_diff.append(np.float64('nan'))
    f_ratio.append(np.float64('nan'))
    x_diff.append(np.float64('nan'))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    # print(iterations.shape)
    plt.plot(iterations, gradients_norm[:T], label='||∇f(xk)||^2')
    plt.xlabel('Iteration')
    plt.ylabel('||∇f(xk)||^2')
    plt.title('Gradient Norm ||∇f(xk)||^2')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(iterations, f_diff[:T], label='f(xk) - f(xT)')
    plt.xlabel('Iteration')
    plt.ylabel('f(xk) - f(xT)')
    plt.title('Function Difference f(xk) - f(xT)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(iterations, f_ratio[:T], label='(f(xk) - f(xT)) / (f(xk-1) - f(xT))')
    plt.xlabel('Iteration')
    plt.ylabel('Ratio f(xk)/f(xk-1)')
    plt.title('Function Ratio')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(iterations, x_diff[:T], label='||xk - xT||^2')
    plt.xlabel('Iteration')
    plt.ylabel('||xk - xT||^2')
    plt.title('||xk - xT||^2 Difference')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Q2d_plots.png")

# Initialization
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Run Exact Line Search
x_star, fx_star, T = ExactLineSearch(x0)

# Output results
print("Optimized x*:", x_star)
print("Function value f(x*):", fx_star)
print("Number of iterations (T):", T)
