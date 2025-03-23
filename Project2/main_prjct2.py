# main.py

# Import  methods from each script
from bisection import bisection_method, f1, f2, f3, find_valid_interval_monte_carlo
from newton import newton_method, f1_prime, f2_prime, f3_prime
from secant import secant_method, f1 as f1_secant, f2 as f2_secant, f3 as f3_secant

def display_iterations():
    # Define the functions
    functions = [
        (f1, f1_prime, "f1(x) = x^2 - 4sin(x)"),
        (f2, f2_prime, "f2(x) = x^2 - 1"),
        (f3, f3_prime, "f3(x) = x^3 - 3x^2 + 3x - 1")
    ]

    # Run Bisection Method
    for f, title in functions:
        a, b = find_valid_interval_monte_carlo(f, -2, 2)
        if a is not None:
            root, iterations, _ = bisection_method(f, a, b, 100, 1e-6, 1e-6)
            print(f"Bisection method for {title}: Root found in {iterations} iterations.")
        else:
            print(f"Bisection method for {title}: No valid interval found.")

    # Run Newton's Method
    for f, f_prime, title in functions:
        root, iterations = newton_method(f, f_prime, x0=1.0, nmax=100, delta1=1e-6, delta2=1e-6, epsilon=1e-6)
        print(f"Newton method for {title}: Root found in {len(iterations)} iterations.")

    # Run Secant Method
    for f, title in zip([f1_secant, f2_secant, f3_secant], ["f1(x) = x^2 - 4sin(x)", "f2(x) = x^2 - 1", "f3(x) = x^3 - 3x^2 + 3x - 1"]):
        root, iterations = secant_method(f, a=-2, b=2, nmax=100, delta1=1e-6, delta2=1e-6)
        print(f"Secant method for {title}: Root found in {len(iterations)} iterations.")

if __name__ == "__main_prjct2__":
    display_iterations()
