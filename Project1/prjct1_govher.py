# ***********************************
# Govher Sapardurdyyeva             *
# Math 374, Spring 2025             *
# Project 1                         *
# Jung - Han Kimn                   *
# This program calculates the       *
# truncation and rounding errors    *
# and displays the grpahs and the   *
# table with values                 *
# ***********************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given function and its derivative
def f(x):
    return np.sin(x)
def df_exact(x):
    return np.cos(x)

# Given formulas
def df_1(x, h):
    return (f(x + h) - f(x)) / h
def df_2(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Error calculation for truncation error
def truncation_error_1(h):
    return h / 2  # Leading term in Taylor series

def truncation_error_2(h):
    return h**2 / 6  # Leading term in Taylor series

# Machine epsilon for rounding error
eps = np.finfo(float).eps
def rounding_error(h):
    return eps / h

# Define h values
# h = 1 * 10^-k where k ranges from 1 to 16
k_values = np.arange(1, 17, dtype=float)
h_values = 10.0**-k_values  # Ensure floating point division
x_point = 1

# Error calculations
err_for1 = np.abs(df_1(x_point, h_values) - df_exact(x_point))
err_for2 = np.abs(df_2(x_point, h_values) - df_exact(x_point))
trunc_1 = truncation_error_1(h_values)
trunc_2 = truncation_error_2(h_values)
rounding_errs = rounding_error(h_values)
total_err1 = trunc_1 + rounding_errs
total_err2 = trunc_2 + rounding_errs

# DataFrame to store error values
df_errors = pd.DataFrame({
    "h": h_values,
    "Formula 1": err_for1,
    "Formula 2": err_for2,
    "Truncation Error (1)": trunc_1,
    "Truncation Error (2)": trunc_2,
    "Rounding Error": rounding_errs,
    "Total Error (1)": total_err1,
    "Total Error (2)": total_err2
})

# Table with error values fixed to 6 decimal places
pd.set_option("display.float_format", "{:.2e}".format)
print(df_errors.to_string(index=False))

# Data visuals
# Plot the values using formula 1
plt.figure(figsize=(10, 6))
plt.loglog(h_values, trunc_1, label="Truncation Error (1)", linestyle='dashed', color='yellow')
plt.loglog(h_values, rounding_errs, label="Rounding Error (1)", linestyle='dashed', color='blue')
plt.loglog(h_values, total_err1, label="Total Error (1)", linestyle='solid', color='black')
plt.xlabel(r'$\log_{10} h$')
plt.ylabel(r'$\log_{10} |error|$')
plt.legend()
plt.title("Truncation, Rounding, and Total Error for formula 1")
plt.grid(True, which="both", linestyle="--")
plt.show()

# Plot the values for formual 2
plt.figure(figsize=(10, 6))
plt.loglog(h_values, trunc_2, label="Truncation Error (2)", linestyle='dashed', color='yellow')
plt.loglog(h_values, rounding_errs, label="Rounding Error (2)", linestyle='dashed', color='blue')
plt.loglog(h_values, total_err2, label="Total Error (2)", linestyle='solid', color='black')
plt.xlabel(r'$\log_{10} h$')
plt.ylabel(r'$\log_{10} |error|$')
plt.legend()
plt.title("Truncation, Rounding, and Total Error for formula 2")
plt.grid(True, which="both", linestyle="--")
plt.show()

