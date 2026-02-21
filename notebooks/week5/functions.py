import numpy as np
import matplotlib.pyplot as plt
from meer21cm import MockSimulation
from meer21cm.plot import plot_map
from meer21cm.fg import ForegroundSimulation
from meer21cm.util import pca_clean
from gpr import *
from nautilus import Sampler, Prior
from scipy import optimize
from meer21cm.util import redshift_to_freq
from joblib import Parallel, delayed # this is for the multiprocessing but works with Jupyter notebooks
from tqdm import tqdm # this is to create a progress bar when running code 
import time
import os

# function created in notebook 1
# altered for inputs of k_perppara_min and k_perppara_max
# this is called by generate_power_spectrum_data_100_realisations() which runs this function 100 times with different seeds
def generate_power_spectrum_data(seed, fg_map, k_perppara_min, k_perppara_max, map_type = "hi"):

    # check if seed is an integer
    if not isinstance(seed, (int, np.integer)):
        
        print("Error: The seed must be an integer.")
        return

    # generate mock data according to seed
    mock = MockSimulation(
        survey='meerklass_2021',
        band='L',
        ra_range=(334, 357),
        dec_range=(-35, -26.5),
        flat_sky=True,
        seed=seed,
        omega_hi=5e-4,
        mean_amp_1='average_hi_temp',
        tracer_bias_1=1.0,
    )

    # didn't include foreground simulation since we only need to include it once?

    # create map from generated data
    hi_map = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    #fg_map = fgsim.fg_wcs_cube(mock.nu) # don't need to load this in?

    # create noise map
    sigma_noise = 1.21 * 1e-3
    noise_map = np.random.default_rng(mock.seed).normal(0, sigma_noise, hi_map.shape) * mock.w_HI

    if map_type == "hi":
        map = hi_map
    elif map_type == "fg":
        map = fg_map
    elif map_type == "noise":
        map = noise_map
    else: 
        print("Invalid map type")
        return 

    # create total map
    tot_map = hi_map + fg_map
    #tot_map = hi_map + fg_map + noise_map # but this does not include noise

    # remove first 3 modes
    res_map = pca_clean(tot_map,3,weights=mock.w_HI,return_analysis=False,mean_center=True)

    # bins
    mock.kparabins = np.linspace(0.01, 1, 11)
    mock.kperpbins = np.linspace(0.01, 0.5, 21)
    mock.k1dbins = np.linspace(0, 1.2, 21)
    
    # no foreground removal
    mock.data = map
    mock.grid_data_to_field();
    power_cy_map,_ = mock.get_cy_power(mock.auto_power_3d_1)
    power_1d_map,k_1d,nmodes_1d = mock.get_1d_power(mock.auto_power_3d_1)

    # with foreground removal
    mock.data = res_map            # this is the hi_map + fg_map with the 3 modes removed
    mock.grid_data_to_field();
    power_cy_res,_ = mock.get_cy_power(mock.auto_power_3d_1)
    power_1d_res,k_1d,nmodes_1d = mock.get_1d_power(mock.auto_power_3d_1, 
                                                    k_perppara_min = k_perppara_min, 
                                                    k_perppara_max = k_perppara_max) # add in inputs here

    return k_1d, power_1d_map, power_1d_res, power_cy_map, power_cy_res


# function to call above function and run it with multiprocessing
# also gives progress bar
# altered for inputs of k_perppara_min and k_perppara_max
# where if there is no input, put the default as None (which is the default in the documentation)
def generate_power_spectrum_data_100_realisations(seeds, fg_map, k_perppara_min, k_perppara_max):

    # determine number of processes
    num_processes = os.cpu_count()
    
    # start timer
    start_time = time.time()
    
    # run in parallel with progress bar
    results = Parallel(n_jobs=num_processes)(
    delayed(generate_power_spectrum_data)(seed, fg_map, k_perppara_min, k_perppara_max)
    for seed in tqdm(seeds, desc="Processing seeds", leave=True, position=0))
    
    # calculate time passed and print to update the progress bar
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    
    # unpack results
    k_1d_list = [result[0] for result in results]
    power_1d_hi_list = [result[1] for result in results]
    power_1d_res_list = [result[2] for result in results]
    power_cy_hi_list = [result[3] for result in results]
    power_cy_res_list = [result[4] for result in results]
    
    # take the means of the 100 realisations
    #k_1d_mean = np.mean(k_1d, axis = 0)
    #power_1d_hi_mean = np.mean(power_1d_hi, axis = 0)
    #power_1d_res_mean = np.mean(power_1d_res, axis = 0)
    #power_cy_hi_mean = np.mean(power_cy_hi, axis = 0)
    #power_cy_res_mean = np.mean(power_cy_res, axis = 0)
    
    # take the standard deviations of the 100 realisations
    #k_1d_std = np.std(k_1d, axis = 0)
    #power_1d_hi_std = np.std(power_1d_hi, axis = 0)
    #power_1d_res_std = np.std(power_1d_res, axis = 0)
    #power_cy_hi_std = np.std(power_cy_hi, axis = 0)
    #power_cy_res_std = np.std(power_cy_res, axis = 0)

    return k_1d_list, power_1d_hi_list, power_1d_res_list, power_cy_hi_list, power_cy_res_list


def plot_1d_log_power_spectrum(k_1d_list, power_1d_hi_list, power_1d_res_list, k_perppara_min, k_perppara_max, errors = True, map_type = "hi", save_plot = False, week_number = None):
    
    # calculate log values
    log_power_1d_hi_list = np.log10(power_1d_hi_list)
    log_power_1d_res_list = np.log10(power_1d_res_list)

    # calculate means
    k_1d_mean = np.mean(k_1d_list, axis = 0)
    log_power_1d_hi_mean = np.mean(log_power_1d_hi_list, axis = 0)
    log_power_1d_res_mean = np.mean(log_power_1d_res_list, axis = 0)

    # calculate standard deviations
    k_1d_std = np.std(k_1d_list, axis = 0)
    log_power_1d_hi_std = np.abs(np.std(log_power_1d_hi_list, axis = 0))
    log_power_1d_res_std = np.abs(np.std(log_power_1d_res_list, axis = 0))

    # plot the log 1D power spectrum
    plt.plot(k_1d_mean, log_power_1d_hi_mean, label='no foreground removal', color = 'blue', linewidth = 1)
    plt.plot(k_1d_mean, log_power_1d_res_mean, label='after foreground removal',ls='--', color = 'red', linewidth = 1)

    if errors == True:
        plt.errorbar(k_1d_mean, log_power_1d_hi_mean, yerr = log_power_1d_hi_std, xerr = k_1d_std, fmt = 'o', capsize = 3, color = 'blue', markersize = 1, linewidth = 1)
        plt.errorbar(k_1d_mean, log_power_1d_res_mean, yerr = log_power_1d_hi_std, xerr = k_1d_std, fmt = 'None', capsize = 3, color = 'red', markersize = 1, linewidth = 1)

    # add title and labels
    plt.xlabel('k [Mpc$^{-1}$]')
    plt.ylabel(r'log$_{10}$ P(k) [${\rm Mpc}^{3}K^2]$)')
    plt.title('1D Power Spectrum')
    plt.legend()

    # save plot
    if save_plot == True:
        save_project_plot(plt.gcf(), week_number = {week_number}, filename = f"1d_log_power_spectrum_{k_perppara_min}_{k_perppara_max}_{map_type}.pdf")

    # show plot
    plt.show()

# function to plot the 1d
def plot_1d_power_spectrum(k_1d_list, power_1d_hi_list, power_1d_res_list, k_perppara_min, k_perppara_max, errors = True, map_type = "hi"):

    # convert to numpy array
    k_1d_list = np.array(k_1d_list)
    power_1d_hi_list = np.array(power_1d_hi_list)
    power_1d_res_list = np.array(power_1d_res_list)

    # calculate y axis values
    #y_axis_before_list = (power_1d_hi_list.T) * k_1d_list**(3/2)
    #y_axis_after_list = (power_1d_res_list.T) * k_1d_list**(3/2)
    y_axis_before_list = (power_1d_hi_list) * k_1d_list**(3/2)
    y_axis_after_list = (power_1d_res_list) * k_1d_list**(3/2)

    # calculate means
    k_1d_mean = np.mean(k_1d_list, axis = 0)
    y_axis_before_mean = np.mean(y_axis_before_list, axis = 0)
    y_axis_after_mean = np.mean(y_axis_after_list, axis = 0)

    # calculate standard deviations
    k_1d_std = np.std(k_1d_list, axis = 0)
    y_axis_before_std = np.std(y_axis_before_list, axis = 0)
    y_axis_after_std = np.std(y_axis_after_list, axis = 0)

    # plot the comparison of the 1d power spectrum 
    plt.plot(k_1d_mean, y_axis_before_mean,label='no foreground removal')
    plt.plot(k_1d_mean, y_axis_after_mean,label='after foreground removal',ls='--')

    # plot errors if True
    if errors == True:
        plt.errorbar(k_1d_mean, y_axis_before_mean, yerr = y_axis_before_std, xerr = k_1d_std, fmt = 'o', capsize = 3, color = 'blue', markersize = 1, linewidth = 1)
        plt.errorbar(k_1d_mean, y_axis_after_mean, yerr = y_axis_after_std, xerr = k_1d_std, fmt = 'None', capsize = 3, color = 'red', markersize = 1, linewidth = 1)

    # add titles and labels
    plt.xlabel('k [Mpc$^{-1}$]')
    plt.ylabel(r'P(k)$k^{3/2}$ [${\rm Mpc}^{3/2}K^2]$)')
    plt.title('1D Power Spectrum')
    plt.legend()

    # save plot?
    #save_project_plot(plt.gcf(), week_number = 5, filename = f"1d_power_spectrum_{k_perppara_min}_{k_perppara_max}_{map_type}.pdf")
    
    plt.show()

def plot_cylindrical_power_spectrum(mock, power_cy_hi_list, power_cy_res_list, k_perppara_min, k_perppara_max, map_type = "hi"):
    
    # bins
    mock.kparabins = np.linspace(0, 1, 11)
    mock.kperpbins = np.linspace(0, 0.5, 21)

    # convert to arrays
    power_cy_hi_list = np.array(power_cy_hi_list)
    power_cy_res_list = np.array(power_cy_res_list)

    # calculate means
    power_cy_hi_mean = np.mean(power_cy_hi_list, axis = 0)
    power_cy_res_mean = np.mean(power_cy_res_list, axis = 0)

    # calculate log values
    log_power_cy_hi_mean = np.log10(power_cy_hi_mean.T)
    log_power_cy_res_mean = np.log10(power_cy_res_mean.T)

    # find minimum and maximum values
    vmin = min(log_power_cy_hi_mean.min(), log_power_cy_res_mean.min())
    vmax = max(log_power_cy_hi_mean.max(), log_power_cy_res_mean.max())
    #vmin = np.log10([power_cy_hi.min(),power_cy_res.min()]).min()
    #vmax = np.log10([power_cy_hi.max(),power_cy_res.max()]).max()   

    # create empty figures
    fig,axes=plt.subplots(1,3,figsize=(15,5))

    # plot one
    axes[0].pcolormesh(mock.kperpbins, mock.kparabins, log_power_cy_hi_mean, vmin = vmin, vmax = vmax)
    axes[0].set_xlabel('k$_\perp$ [Mpc$^{-1}$]')
    axes[0].set_ylabel('k$_\parallel$ [Mpc$^{-1}$]')
    axes[0].set_title('HI power spectrum')

    # plot two
    im = axes[1].pcolormesh(mock.kperpbins, mock.kparabins, log_power_cy_res_mean, vmin = vmin, vmax = vmax)
    axes[1].set_xlabel('k$_\perp$ [Mpc$^{-1}$]')
    axes[1].set_ylabel('k$_\parallel$ [Mpc$^{-1}$]')
    axes[1].set_title('After foreground removal')

    # colour bar
    cbar = plt.colorbar(im,ax=axes[:2],location='top',aspect=50,pad=0.1)
    cbar.set_label(r'log$_{10}$ P(k$_\perp$, k$_\parallel$) [${\rm Mpc}^{3}K^2]$)')

    # plot three
    im = axes[2].pcolormesh(mock.kperpbins, mock.kparabins, (power_cy_res_mean.T/power_cy_hi_mean.T),cmap='bwr')
    axes[2].set_xlabel('k$_\perp$ [Mpc$^{-1}$]')
    axes[2].set_ylabel('k$_\parallel$ [Mpc$^{-1}$]')
    axes[2].set_title('Ratio')
    cbar = plt.colorbar(im,ax=axes[2],location='top',aspect=50,pad=0.1)

    # save plot?
    #save_project_plot(plt.gcf(), week_number = 5, filename = f"cylindrical_power_spectrum_{k_perppara_min}_{k_perppara_max}_{map_type}.pdf")

    # show plot
    plt.show()

def plot_HI_signal_kernal_comparison(mock, hi_map, cov_noise, cov_fg, fitted_pars):

    # generate gaussian noise and extract its covariance
    sigma_noise = 1.21 * 1e-3
    noise_map = np.random.default_rng(mock.seed).normal(0,sigma_noise,hi_map.shape) * mock.w_HI
    cov_noise,_,_,_ = pca_clean(noise_map,1,weights=mock.w_HI,return_analysis=True,mean_center=True)

    # create hi signal kernal using exponential kernal and best fit parameters
    fitted_hi_kern = exponential_kernel(mock.nu / 1e6, mock.nu / 1e6, fitted_pars[0], fitted_pars[2])

    # plot the best fitted kernal
    fig,axes = plt.subplots(1,2,figsize=(10,5))
    vmin = np.min([fitted_hi_kern.min(), cov_noise.min()])
    vmax = np.max([fitted_hi_kern.max(), cov_noise.max()])
    im = axes[0].imshow(fitted_hi_kern,origin='lower',extent=[mock.nu[0]/1e6,mock.nu[-1]/1e6,mock.nu[0]/1e6,mock.nu[-1]/1e6],vmin=vmin,vmax=vmax)
    plt.colorbar(im, location='top',pad=0.1)
    axes[0].set_xlabel('Frequency [MHz]')
    axes[0].set_ylabel('Frequency [MHz]')
    axes[0].set_title(r'Fitted HI + noise kernel')

    # plot the noise covariance
    im = axes[1].imshow(cov_noise,origin='lower',extent=[mock.nu[0]/1e6,mock.nu[-1]/1e6,mock.nu[0]/1e6,mock.nu[-1]/1e6],vmin=vmin,vmax=vmax)
    plt.colorbar(im, location='top',pad=0.1)
    axes[1].set_xlabel('Frequency [MHz]')
    axes[1].set_ylabel('Frequency [MHz]')
    axes[1].set_title(r'Noise covariance')

def plot_foreground_signal_kernal_comparison(mock, cov_fg, fitted_pars):

    # create the foreground kernal using the rbg kernal and best fit parameters
    fitted_fg_kern = rbf_kernel(mock.nu / 1e6, mock.nu / 1e6, fitted_pars[1] , fitted_pars[3], fitted_pars[4])

    # plot the best fitted kernal
    fig,axes = plt.subplots(1,2,figsize=(10,5))
    vmin = np.min([fitted_fg_kern.min(), cov_fg.min()])
    vmax = np.max([fitted_fg_kern.max(), cov_fg.max()])
    im = axes[0].imshow(fitted_fg_kern,origin='lower',extent=[mock.nu[0]/1e6,mock.nu[-1]/1e6,mock.nu[0]/1e6,mock.nu[-1]/1e6],vmin=vmin,vmax=vmax)
    plt.colorbar(im, location='top',pad=0.1)
    axes[0].set_xlabel('Frequency [MHz]')
    axes[0].set_ylabel('Frequency [MHz]')
    axes[0].set_title(r'Best fit foreground kernel')

    # plot the foreground covariance
    im = axes[1].imshow(cov_fg,origin='lower',extent=[mock.nu[0]/1e6,mock.nu[-1]/1e6,mock.nu[0]/1e6,mock.nu[-1]/1e6],vmin=vmin,vmax=vmax)
    plt.colorbar(im, location='top',pad=0.1)
    axes[1].set_xlabel('Frequency [MHz]')
    axes[1].set_ylabel('Frequency [MHz]')
    axes[1].set_title(r'Foreground covariance')

def foreground_kernal_removal_plots(mock, fg_map, noise_map, tot_map, fitted_pars):

    fitted_hi_kern = exponential_kernel(mock.nu / 1e6, mock.nu / 1e6, fitted_pars[0], fitted_pars[2])
    fitted_fg_kern = rbf_kernel(mock.nu / 1e6, mock.nu / 1e6, fitted_pars[1] , fitted_pars[3], fitted_pars[4])

    fg_remov = np.einsum(
    'ij, abj->abi',
    fitted_fg_kern @ np.linalg.inv(fitted_fg_kern + fitted_hi_kern),
    tot_map
    )
    res_map = tot_map - fg_remov

    noise_residual = np.einsum(
    'ij, abj->abi',
    np.eye(mock.nu.size) - fitted_fg_kern @ np.linalg.inv(fitted_fg_kern+fitted_hi_kern),
    noise_map
    )

    plot_map(fg_remov,mock.wproj,W=mock.W_HI, title='Removed foreground')

    plot_map(fg_map,mock.wproj,W=mock.W_HI, title='Actual foreground')