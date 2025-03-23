import numpy as np
import matplotlib.pyplot as plt

"""
*************************************
* Govher Sapardurdyyeva             *
* Math 374, Spring 2025             *
* Project 2                         *
* Jung - Han Kimn                   *
* Secant method to find the roots   *
* of the function f                 *
*************************************
"""

# Define the functions
def f1(x):
    return x ** 2 - 4 * np.sin(x)


def f2(x):
    return x ** 2 - 1


def f3(x):
    return x ** 3 - 3 * x ** 2 + 3 * x - 1


# Secant Method Implementation with Zero Division Handling
def secant_method(f, a, b, nmax, delta1, delta2):
    iterations = []
    fa = f(a)
    fb = f(b)

    # Swap a and b
    if abs(fb) < abs(fa):
        a, b = b, a
        fa, fb = fb, fa

    iterations.append((a, fa))
    iterations.append((b, fb))

    for n in range(2, nmax + 1):
        # Check for zero division to avoid division by zero
        if abs(fb - fa) < 1e-12:
            print(f"Division by zero detected at iteration {n}. Undefined")
            return a, iterations

        # Compute the next approximation
        d = (b - a) / (fb - fa)
        b = a
        fb = fa
        d = d * fa
        a = a - d
        fa = f(a)

        iterations.append((a, fa))

        # Check stopping criteria
        if abs(d) < delta1 or abs(fa) < delta2:
            print("Converged!")
            return a, iterations

    return a, iterations


# Plotting function for Secant Method
def plot_secant_method(f, x_range, root, iterations, title):
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.vectorize(f)(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f'{title}')
    plt.axhline(0, color='black', linestyle='--')

    # Plot iterations
    iter_x = [point[0] for point in iterations]
    iter_y = [point[1] for point in iterations]
    plt.scatter(iter_x, iter_y, color='red', label='Iterations')

    # Plot the root
    plt.scatter(root, f(root), color='green', marker='x', s=100, label='Root')

    plt.legend()
    plt.title(f"Secant Method: {title}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()


# Test Secant Method on f1, f2, and f3
functions = [f1, f2, f3]
titles = ["f1(x) = x^2 - 4sin(x)", "f2(x) = x^2 - 1", "f3(x) = x^3 - 3x^2 + 3x - 1"]

for f, title in zip(functions, titles):
    a, b = -2, 2
    root, iterations = secant_method(f, a, b, nmax=100, delta1=1e-6, delta2=1e-6)

    print(f"\nRoot found for {title}: {root}")
    print(f"Iterations: {len(iterations)}")
    for i, (xi, fxi) in enumerate(iterations):
        print(f"Iteration {i}: x = {xi}, f(x) = {fxi}")

    # Plot the function and iterations
    plot_secant_method(f, [-2, 2], root, iterations, title)
