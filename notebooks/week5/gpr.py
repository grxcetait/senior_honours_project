import numpy as np
from scipy.special import gamma, kv

def matern_kernel(x, y, sigma, l, eta):
    x = np.atleast_1d(x).ravel()[:,None]
    y = np.atleast_1d(y).ravel()[None,:]
    kernel = (
        sigma**2 
        * 2 ** (1 - eta)
        / gamma(eta)
        * (np.sqrt(2*eta) * np.abs(x-y)/l) ** eta
        * kv(eta, np.sqrt(2*eta) * np.abs(x-y)/l)
    )
    kernel[x-y==0] = sigma**2
    return kernel

def rbf_kernel(x, y, sigma, l, sp_indx=0.0):
    rescale = (x/x.ravel()[0])**sp_indx
    rescale = rescale.ravel()
    rescale = rescale[None,:] * rescale[:,None]
    x = np.atleast_1d(x).ravel()[:,None]
    y = np.atleast_1d(y).ravel()[None,:]
    return np.nan_to_num(sigma**2 * np.exp(-(x-y)**2 / (2*l**2)) * rescale)

def rbf_kernel_no_sp_indx(x, y, sigma, l):
    x = np.atleast_1d(x).ravel()[:,None]
    y = np.atleast_1d(y).ravel()[None,:]
    return np.nan_to_num(sigma**2 * np.exp(-(x-y)**2 / (2*l**2)))

def exponential_kernel(x, y, sigma, l):
    x = np.atleast_1d(x).ravel()[:,None]
    y = np.atleast_1d(y).ravel()[None,:]
    return np.nan_to_num(sigma**2 * np.exp(-np.abs(x-y)/l))

def log_marginal_likelihood(cov_data, kernel):
    chi2 = -0.5 * np.trace(cov_data @ np.linalg.inv(kernel))
    sign, log_determinant = np.linalg.slogdet(kernel)
    if sign <= 0:
        log_determinant = np.inf
    return chi2 - 0.5 * log_determinant

# untouched function
#def function_to_optimise(params, xarr = None, cov_tot = None, minimise=False):
#    sigma_hi, sigma_fg, l_hi, l_fg, sp_indx_fg = params
#    k_hi = exponential_kernel(xarr,xarr,sigma_hi,l_hi)
#    k_fg = rbf_kernel(xarr,xarr,sigma_fg,l_fg,sp_indx_fg)
#    k_tot = k_hi + k_fg
#    if minimise:
#        return -log_marginal_likelihood(cov_tot, k_tot)
#    else:
#        return log_marginal_likelihood(cov_tot, k_tot)

# changed function
def function_to_optimise(params, xarr = None, cov_tot = None, minimise = True):
    
    sigma_hi, sigma_fg, l_hi, l_fg, sp_indx_fg = params

    # Generate kernals
    k_hi = exponential_kernel(xarr, xarr, sigma_hi, l_hi)
    k_fg = rbf_kernel(xarr, xarr, sigma_fg, l_fg, sp_indx_fg)

    # Create total kernal (HI signal + foreground)
    k_tot = k_hi + k_fg

    # Return the optimised function
    if minimise:
        return -log_marginal_likelihood(cov_tot, k_tot)
    else:
        return log_marginal_likelihood(cov_tot, k_tot)

# changed function
def function_to_optimise_no_sp_indx(params, xarr = None, cov_tot = None, minimise = True):
    
    sigma_hi, sigma_fg, l_hi, l_fg = params

    # Generate kernals
    k_hi = exponential_kernel(xarr, xarr, sigma_hi, l_hi)
    k_fg = rbf_kernel(xarr, xarr, sigma_fg, l_fg)

    # Create total kernal (HI signal + foreground)
    k_tot = k_hi + k_fg

    # Return the optimised function
    if minimise:
        return -log_marginal_likelihood(cov_tot, k_tot)
    else:
        return log_marginal_likelihood(cov_tot, k_tot)

# changed function
def function_to_optimise_matern(params, xarr=None, cov_tot=None, minimise=True):
    sigma_hi, sigma_fg, l_hi, l_fg, eta_fg = params
    
    k_hi = exponential_kernel(xarr, xarr, sigma_hi, l_hi)
    k_fg = matern_kernel(xarr, xarr, sigma_fg, l_fg, eta_fg)
    
    k_tot = k_hi + k_fg
    
    if minimise:
        return -log_marginal_likelihood(cov_tot, k_tot)
    else:
        return log_marginal_likelihood(cov_tot, k_tot)

# changed function
def function_to_optimise_matern_5_2(params, xarr=None, cov_tot=None, minimise=True):
    sigma_hi, sigma_fg, l_hi, l_fg = params
    
    k_hi = exponential_kernel(xarr, xarr, sigma_hi, l_hi)
    k_fg = matern_kernel(xarr, xarr, sigma_fg, l_fg, 5/2)
    
    k_tot = k_hi + k_fg
    
    if minimise:
        return -log_marginal_likelihood(cov_tot, k_tot)
    else:
        return log_marginal_likelihood(cov_tot, k_tot)

# changed function
def function_to_optimise_matern_3_2(params, xarr=None, cov_tot=None, minimise=True):
    sigma_hi, sigma_fg, l_hi, l_fg = params
    
    k_hi = exponential_kernel(xarr, xarr, sigma_hi, l_hi)
    k_fg = matern_kernel(xarr, xarr, sigma_fg, l_fg, 3/2)
    
    k_tot = k_hi + k_fg
    
    if minimise:
        return -log_marginal_likelihood(cov_tot, k_tot)
    else:
        return log_marginal_likelihood(cov_tot, k_tot)

# changed function
def function_to_optimise_l_fg_fixed_no_sp_indx_fg(params, l_fg = None, xarr = None, cov_tot = None, minimise = True):
    
    sigma_hi, sigma_fg, l_hi = params

    # Generate kernals
    k_hi = exponential_kernel(xarr, xarr, sigma_hi, l_hi)
    k_fg = rbf_kernel_no_sp_indx(xarr, xarr, sigma_fg, l_fg)

    # Create total kernal (HI signal + foreground)
    k_tot = k_hi + k_fg

    # Return the optimised function
    if minimise:
        return -log_marginal_likelihood(cov_tot, k_tot)
    else:
        return log_marginal_likelihood(cov_tot, k_tot)

# changed function
def function_to_optimise_l_fg_fixed(params, l_fg = None, xarr = None, cov_tot = None, minimise = True):
    
    sigma_hi, sigma_fg, l_hi, sp_indx_fg = params

    # Generate kernals
    k_hi = exponential_kernel(xarr, xarr, sigma_hi, l_hi)
    k_fg = rbf_kernel(xarr, xarr, sigma_fg, l_fg, sp_indx_fg)

    # Create total kernal (HI signal + foreground)
    k_tot = k_hi + k_fg

    # Return the optimised function
    if minimise:
        return -log_marginal_likelihood(cov_tot, k_tot)
    else:
        return log_marginal_likelihood(cov_tot, k_tot)