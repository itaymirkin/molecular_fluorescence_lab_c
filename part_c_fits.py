import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Define the upper and lower polariton functions
def upper_polariton(x, a0, a1, a2, a3, a4):
    return a2 - a0 * np.sqrt((x - a4) ** 2 + a1 ** 2) + np.sqrt(
        4 * a3 ** 2 + (a2 - a0 * np.sqrt((x - a4) ** 2 + a1 ** 2)) ** 2) / 2

def lower_polariton(x, a0, a1, a2, a3, a4):
    return a2 - a0 * np.sqrt((x - a4) ** 2 + a1 ** 2) - np.sqrt(
        4 * a3 ** 2 + (a2 - a0 * np.sqrt((x - a4) ** 2 + a1 ** 2)) ** 2) / 2

# Define the residual function for least squares fitting
def residual_func(params, x, y, yerr, polariton_func):
    return (polariton_func(x, *params) - y)

# Perform the least squares fit
def perform_least_squares_fit(x, y, yerr, initial_guess, polariton_func):
    bounds = ([-10, -25, -1.5, -0.5, -0.5],  # lower bounds
              [10, 100, 2.5, 0.5, 0.5])

    result = least_squares(residual_func, initial_guess, args=(x, y, yerr, polariton_func),
                           bounds=bounds, method='trf', loss='linear',
                           f_scale=0.1, verbose=1)

    return result

# Load data from Excel file
df_upper = pd.read_excel('data_partc_with_errors.xlsx', sheet_name='Upper Values')
df_lower = pd.read_excel('data_partc_with_errors.xlsx', sheet_name='Lower Values')

# Upper polariton data
k_values_upper = df_upper['k'].values * 1000
energy_values_upper = df_upper['Energy'].values
energy_errors_upper = df_upper['Energy_error'].values
k_errors_upper = abs(df_upper['k_error'].values * 1000)  # Assuming k_error is in the same units as k

# Lower polariton data
k_values_lower = df_lower['k'].values * 1000
energy_values_lower = df_lower['Energy'].values
energy_errors_lower = df_lower['Energy_error'].values
k_errors_lower = abs(df_lower['k_error'].values * 1000)  # Assuming k_error is in the same units as k

# Initial guess as specified
initial_guess = [0.132, 20.94, 2.23, 0.1, 0.1]

# Perform the fits
result_upper = perform_least_squares_fit(k_values_upper, energy_values_upper, energy_errors_upper, initial_guess, upper_polariton)
result_lower = perform_least_squares_fit(k_values_lower, energy_values_lower, energy_errors_lower, initial_guess, lower_polariton)

# Extract the optimized parameters
best_params_upper = result_upper.x
best_params_lower = result_lower.x

# Calculate the Jacobian at the solution
J_upper = result_upper.jac
J_lower = result_lower.jac

# Calculate the covariance matrix
try:
    cov_upper = np.linalg.inv(J_upper.T @ J_upper)
    param_errors_upper = np.sqrt(np.diag(cov_upper))
except np.linalg.LinAlgError:
    print("Warning: Singular matrix encountered. Parameter errors may not be reliable.")
    param_errors_upper = np.full(5, np.nan)

try:
    cov_lower = np.linalg.inv(J_lower.T @ J_lower)
    param_errors_lower = np.sqrt(np.diag(cov_lower))
except np.linalg.LinAlgError:
    print("Warning: Singular matrix encountered. Parameter errors may not be reliable.")
    param_errors_lower = np.full(5, np.nan)

# Calculate R-squared
residuals_upper = result_upper.fun * energy_errors_upper  # Unweight the residuals
ss_res_upper = np.sum(residuals_upper ** 2)
ss_tot_upper = np.sum((energy_values_upper - np.mean(energy_values_upper)) ** 2)
r_squared_upper = 1 - (ss_res_upper / ss_tot_upper)

residuals_lower = result_lower.fun * energy_errors_lower  # Unweight the residuals
ss_res_lower = np.sum(residuals_lower ** 2)
ss_tot_lower = np.sum((energy_values_lower - np.mean(energy_values_lower)) ** 2)
r_squared_lower = 1 - (ss_res_lower / ss_tot_lower)

# Print the results
print("\nFinal fitted parameters and their errors - Upper:")
for i, (param, error) in enumerate(zip(best_params_upper, param_errors_upper)):
    print(f"a{i} = {param:.6f} ± {error:.6f}")
print(f"Final cost: {result_upper.cost}")
print(f"R-squared: {r_squared_upper:.4f}")

print("\nFinal fitted parameters and their errors - Lower:")
for i, (param, error) in enumerate(zip(best_params_lower, param_errors_lower)):
    print(f"a{i} = {param:.6f} ± {error:.6f}")
print(f"Final cost: {result_lower.cost}")
print(f"R-squared: {r_squared_lower:.4f}")

# Plotting
k_fit_upper = np.linspace(min(k_values_upper), max(k_values_upper), 1000)
fit_values_upper = upper_polariton(k_fit_upper, *best_params_upper)

k_fit_lower = np.linspace(min(k_values_lower), max(k_values_lower), 1000)
fit_values_lower = lower_polariton(k_fit_lower, *best_params_lower)

residuals_upper = (energy_values_upper - upper_polariton(k_values_upper, *best_params_upper)) / energy_errors_upper
residuals_lower = (energy_values_lower - lower_polariton(k_values_lower, *best_params_lower)) / energy_errors_lower

# Main plot upper
plt.figure(figsize=(10, 6))
plt.errorbar(k_values_upper, energy_values_upper, xerr=k_errors_upper, yerr=energy_errors_upper, fmt='o', label='Data')
plt.plot(k_fit_upper, fit_values_upper, 'r-', label='Fitted curve')
plt.xlabel(r'$k \ (\frac{1}{\mu m})$')
plt.ylabel('Energy (eV)')
plt.title('Upper Polariton: Energy vs. Wave Number k')
plt.legend()
plt.show()

# Residuals plot upper
plt.figure(figsize=(10, 6))
plt.errorbar(k_values_upper, residuals_upper, xerr=k_errors_upper, yerr=energy_errors_upper * 200, fmt='o')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel(r'$k \ (\frac{1}{\mu m})$')
plt.ylabel('Residuals')
plt.title('Residuals of the Upper Polariton Fit')
plt.show()

# Main plot lower
plt.figure(figsize=(10, 6))
plt.errorbar(k_values_lower, energy_values_lower, xerr=k_errors_lower, yerr=energy_errors_lower, fmt='o', label='Data')
plt.plot(k_fit_lower, fit_values_lower, 'r-', label='Fitted curve')
plt.xlabel(r'$k \ (\frac{1}{\mu m})$')
plt.ylabel('Energy (eV)')
plt.title('Lower Polariton: Energy vs. Wave Number k')
plt.legend()
plt.show()

# Residuals plot lower
plt.figure(figsize=(10, 6))
plt.errorbar(k_values_lower, residuals_lower, xerr=k_errors_lower, yerr=energy_errors_lower*200, fmt='o')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel(r'$k \ (\frac{1}{\mu m})$')
plt.ylabel('Residuals')
plt.title('Residuals of the Lower Polariton Fit')
plt.show()
