import math
import numpy as np
import matplotlib.pyplot as plt
import random

"""
*************************************
* Govher Sapardurdyyeva             *
* Math 374, Spring 2025             *
* Project 2                         *
* Jung - Han Kimn                   *
* Bisection method for finding a    *
* root of function f in the interval*
* [a, b]                            *
*************************************
"""

def bisection_method(f, a, b, M, delta, epsilon):
    """""
****************************************************
* Parameters:                                      *
*   f: function.                                   *
*   a, b: float - Interval (will be using [-2 , 2].*
*   M: int - Maximum number of iterations.         *
*   delta: float - Interval tolerance.             *
*   epsilon: float - Function value tolerance.     *
* Return value:                                    *
*   float - Approximate roots of function f.       *
****************************************************
"""
    u = f(a)
    v = f(b)
    e = b - a

    if u * v > 0:
        print("Function has the same sign at a and b. Bisection method cannot proceed.")
        return None, 0

    iter_points = []
    for k in range(1, M + 1):
        e /= 2
        c = a + e
        w = f(c)
        iter_points.append((c, w))

        print(f"Iteration {k}: c = {c}, w = {w}, e = {e}")

        if abs(e) < delta or abs(w) < epsilon:
            print("Stopping criteria met.")
            return c, k, iter_points  # Return the number of iterations as well

        if w * u < 0:
            b = c
            v = w
        else:
            a = c
            u = w

    return c, M, iter_points  # Return maximum iterations if no early stopping


# Functions
def f1(x):
    return x ** 2 - 4 * math.sin(x)


def f2(x):
    return x ** 2 - 1


def f3(x):
    return x ** 3 - 3 * x ** 2 + 3 * x - 1


# Monte Carlo method to find a valid interval for bisection method
def find_valid_interval_monte_carlo(f, start=-2, end=2, num_trials=1000):
    for _ in range(num_trials):
        x1 = random.uniform(start, end)
        x2 = random.uniform(start, end)
        if x1 > x2:
            x1, x2 = x2, x1  # Make sure x1 < x2

        if f(x1) * f(x2) < 0:
            return x1, x2
    return None, None


# Function to plot the function and iterations
def plot_function(f, a, b, root, iter_points, title):
    x = np.linspace(a, b, 400)
    y = np.vectorize(f)(x)

    plt.figure()
    plt.plot(x, y, label=f'{title}')
    plt.axhline(0, color='black', linestyle='--')
    plt.scatter(*zip(*iter_points), color='red', label='Iterations')
    plt.scatter([root], [f(root)], color='green', marker='x', s=100, label='Root')
    plt.legend()
    plt.title(f"Bisection Method: {title}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()
    plt.show()


# Running bisection method with Monte Carlo for interval finding and plotting results
for f, title in zip([f1, f2, f3], ["f1(x) = x^2 - 4sin(x)", "f2(x) = x^2 - 1", "f3(x) = x^3 - 3x^2 + 3x - 1"]):
    a, b = find_valid_interval_monte_carlo(f, -2, 2)
    if a is not None:
        root, iterations, iter_points = bisection_method(f, a, b, 100, 1e-6, 1e-6)
        print(f"Root for {title}: {root} found in {iterations} iterations.")
        plot_function(f, a, b, root, iter_points, title)
    else:
        print(f"No valid interval found for {title} using Monte Carlo in [-2, 2].")
