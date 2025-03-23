import numpy as np
import matplotlib.pyplot as plt
"""
*************************************
* Govher Sapardurdyyeva             *
* Math 374, Spring 2025             *
* Project 2                         *
* Jung - Han Kimn                   *
* Newton method to find the roots   *
* of the function f                 *
*************************************
"""

# Define the functions and their derivatives
def f1(x):
    return x ** 2 - 4 * np.sin(x)


def f1_prime(x):
    return 2 * x - 4 * np.cos(x)


def f2(x):
    return x ** 2 - 1


def f2_prime(x):
    return 2 * x


def f3(x):
    return x ** 3 - 3 * x ** 2 + 3 * x - 1


def f3_prime(x):
    return 3 * x ** 2 - 6 * x + 3


# Newton's Method
def newton_method(f, f_prime, x0, nmax, delta1, delta2, epsilon):
    iterations = []
    x = x0
    for n in range(1, nmax + 1):
        fx = f(x)
        fpx = f_prime(x)

        # Check if the derivative is too small to avoid division with 0
        if abs(fpx) < epsilon:
            print("Small derivative. Undefined.")
            return x, iterations

        # Calculate the Newton's step
        d = fx / fpx
        x = x - d
        iterations.append((x, fx))

        # Check if convergence criteria are met
        if abs(d) < delta1 or abs(fx) < delta2:
            print("The function converges.")
            return x, iterations

    return x, iterations


# Plotting function
def plot_newton_method(f, x_range, root, iterations, title):
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
    plt.title(f"Newton's Method: {title}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()


    """
def monte_carlo_initial_guess(start, end, num_samples=10):
    initial_guesses = [random.uniform(start, end) for _ in range(num_samples)]
    return initial_guesses
"""
# Test Newton's Method on f1, f2, and f3
functions = [(f1, f1_prime), (f2, f2_prime), (f3, f3_prime)]
titles = ["f1(x) = x^2 - 4sin(x)", "f2(x) = x^2 - 1", "f3(x) = x^3 - 3x^2 + 3x - 1"]

for (f, f_prime), title in zip(functions, titles):
    x0 = 1.0  # Initial guess
    root, iterations = newton_method(f, f_prime, x0, nmax=100, delta1=1e-6, delta2=1e-6, epsilon=1e-6)

    print(f"\nRoot found for {title}: {root}")
    print(f"Iterations: {len(iterations)}")
    for i, (xi, fxi) in enumerate(iterations):
        print(f"Iteration {i}: x = {xi}, f(x) = {fxi}")

    # Plot the function and iterations
    plot_newton_method(f, [-2, 2], root, iterations, title)
