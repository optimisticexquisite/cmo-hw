import numpy as np
from CMO_A1 import f1, f2
# Function to calculate the second derivative using finite differences
def second_derivative(fx, x, h=1e-5):
    return (fx(x + h) - 2 * fx(x) + fx(x - h)) / (h ** 2)

# Function to check convexity
def isConvex(fx, SrNo, interval):
    x_values = np.linspace(interval[0], interval[1], 100)
    second_derivatives = [second_derivative(fx, SrNo, x) for x in x_values]
    for i in range(1, len(second_derivatives), 5):
        print(x_values[i], fx(x_values[i]), second_derivatives[i])
    # Check if all second derivatives are non-negative (convex)
    is_convex = all(d >= 0 for d in second_derivatives)
    is_strictly_convex = all(d > 0 for d in second_derivatives)
    
    if is_strictly_convex:
        return "Strictly convex"
    elif is_convex:
        return "Convex"
    else:
        return "Not convex"

srno = 20466
f_1 = lambda x: f1(srno, x)
f_2 = lambda x: f2(srno, x)

# Testing convexity over the interval [-2, 2]
interval = [-2, 2]

# Check f1 and f2
result_f1 = isConvex(f_1, 1, interval)
result_f2 = isConvex(f_2, 2, interval)

# Output the results
print("f1 is:", result_f1)
print("f2 is:", result_f2)
