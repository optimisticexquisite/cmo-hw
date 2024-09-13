import numpy as np
from scipy.optimize import fsolve
from CMO_A1 import f3 as f_3

# Function to calculate the first derivative using finite differences
def first_derivative(fx, x, h=1e-5):
    return (fx(x + h) - fx(x - h)) / (2 * h)

# Function to calculate the second derivative using finite differences
def second_derivative(fx, x, h=1e-5):
    return (fx(x + h) - 2 * fx(x) + fx(x - h)) / (h ** 2)

# Function to check if a function is coercive
def isCoercive(fx):
    x_large_positive = 1e6
    x_large_negative = -1e6
    
    # Check the values of the function at large positive and large negative values of x
    fx_positive = fx(x_large_positive)
    fx_negative = fx(x_large_negative)
    
    # If both go to infinity, the function is coercive
    return fx_positive > 0 and fx_negative > 0

# Function to find stationary points
def FindStationaryPoints(fx):
    # Generate an array of values from -10 to 10 to estimate stationary points
    x_values = np.linspace(-10, 10, 1000)
    stationary_points = []
    
    # First derivative function
    def f_prime(x):
        return first_derivative(fx, x)
    
    # Find roots of f_prime to get stationary points
    for x0 in np.linspace(-10, 10, 5):  # Initial guesses for root finding
        root, = fsolve(f_prime, x0)
        if root not in stationary_points and -10 <= root <= 10:
            stationary_points.append(root)
    
    # Classify stationary points
    result = {
        "Roots": [],
        "Minima": [],
        "LocalMaxima": []
    }
    
    # Check if they are minima, maxima, or saddle points using the second derivative
    for point in stationary_points:
        fpp = second_derivative(fx, point)
        if fpp > 0:
            result["Minima"].append(point)
        elif fpp < 0:
            result["LocalMaxima"].append(point)
    
    # Find roots where the function itself is zero
    def f_root(x):
        return fx(x)
    
    roots = []
    for x0 in np.linspace(-10, 10, 5):  # Initial guesses for root finding
        root, = fsolve(f_root, x0)
        if root not in roots and -10 <= root <= 10:
            roots.append(root)
    
    result["Roots"] = roots
    return result

# Example: Define a quartic function f3
f3 = lambda x: f_3(20466, x)

# Check coercivity of f3
coercivity_f3 = isCoercive(f3)
print("Is f3 coercive?", coercivity_f3)

# Find stationary points of f3
stationary_points_f3 = FindStationaryPoints(f3)
print("Stationary points of f3:", stationary_points_f3)
