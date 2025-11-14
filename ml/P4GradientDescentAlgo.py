import numpy as np

# Function
def f(x):
    return (x + 3)**2

# Derivative of function
def df(x):
    return 2 * (x + 3)

# Gradient Descent
x = 2                 # starting point
learning_rate = 0.1
epochs = 50

print("Starting Gradient Descent...\n")

for i in range(epochs):
    gradient = df(x)
    x = x - learning_rate * gradient
    print(f"Iteration {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}")

print("\nLocal minimum occurs at x =", round(x, 4))
