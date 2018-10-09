import numpy as np
import h5py
import time as timemodule
import pycbc
from pycbc.waveform import utils
import lal

"""Includes a method to append echoes to a given waveform 
and a tapering function used for the echo's form 
relative to the original waveform."""

def truncfunc(t, t0, t_merger, omega_of_t):

    """A tapering function used to smoothly introduce the echo waveforms
    from the original waveform."""

    theta = 0.5 * (1 + np.tanh(0.5 * omega_of_t * (t - t_merger - t0)))
    return theta


def add_echoes(hp, hc, omega, t0trunc, t_echo, del_t_echo, n_echoes, amplitude,
               gamma, inclination=0., t_merger=None, sampletimesarray=None,
               include_imr=False):
    """Takes waveform timeseries' of plus and cross polarisation, 
    produces echoes of the waveform and returns the original 
    waveform timeseries' with the echoes appended. 
    The original starting time is lost, however.

    Parameters
    ----------
    hp, hc : timeseries
        Plus and cross polarisation timeseries' of the original waveform.
    t0trunc : float
        Truncation time parameter for the echo form, time difference taken 
        with respect to time of merger. Thought to be negative, in seconds.
    t_echo : float
        Time difference between original waveform and first echo, in seconds.
    del_t_echo : float
        Time difference between subsequent echoes, in seconds.
    n_echoes : integer
        Number of echoes to be appended. 
    amplitude : float
        Strain amplitude of first echo with respect to original wave amplitude.
    gamma : float
        Dampening factor between two successive echoes. (n+1)st echo has 
        amplitude of (n)th echo multiplied with gamma.
    inclination : float
        The inclination of the signal.
    
    Returns
    -------
    hp, hc: timeseries
        Waveform timeseries' for plus and cross polarisation with echoes 
        appended.    
    """
    timestep = hp.delta_t
    if t_merger is None:
        t_merger = float((hp**2 + hc**2).numpy().argmax() * hp.delta_t +
                         hp.start_time)
    if sampletimesarray is None:
        sampletimesarray = hp.sample_times.numpy()
    
    # Counting leading zeros. Calculate angular frequency for trimmed waveform.
    # For leading/trailing zeros, use first/last frequency found. Use
    # np.nonzero instead?
    
#    omega = 2. * np.pi * utils.frequency_from_polarizations(hp.trim_zeros(), 
#                                                        hc.trim_zeros())
#    
#    omega_temp = np.zeros(len(template))
#    first_zero_index_hp = 0
#    first_zero_index_hc = 0
#    
#    if hp[0] == 0 and hc[0] == 0:
#        print('Active')
#        
#        while hp[first_zero_index_hp] == 0:
#            first_zero_index_hp += 1
#        
#        while hc[first_zero_index_hc] == 0:
#            first_zero_index_hc += 1
#        
#        if first_zero_index_hp != first_zero_index_hc:
#            print('Polarisations have unequal number of leading zeroes.')
#        
#        omega_temp[:max(first_zero_index_hp,first_zero_index_hc)] = \
#            omega[max(first_zero_index_hp,first_zero_index_hc)]
#        omega = omega_temp
#    
#    omega.resize(len(template))
#    
    #Producing the tapered waveform from the original one for the echoes:
    length = len(hp)
    tapercoeff = truncfunc(sampletimesarray, t0trunc, t_merger, omega.numpy())
    threshold = 0.01
    first_idx = np.nonzero(tapercoeff > threshold)[0][0]
    
    hp_numpy = (hp.numpy() * tapercoeff)
    hc_numpy = (hc.numpy() * tapercoeff)

    last_idx = max(np.nonzero(hp_numpy > threshold * hp_numpy.max())[0][-1],
                   np.nonzero(hc_numpy > threshold * hc_numpy.max())[0][-1])
    last_idx += 1
    hp_numpy = hp_numpy[first_idx:last_idx]
    hc_numpy = hc_numpy[first_idx:last_idx]

    #Appending first echo after t_echo.
    pad = int(np.ceil((t_echo + n_echoes * del_t_echo) * 1.0/timestep))
    hparray = np.zeros(len(hp) + pad)
    hcarray = np.zeros(len(hc) + pad)

    t_echo_steps = int(round(t_echo * 1.0/timestep)) + first_idx
    hparray[t_echo_steps:t_echo_steps+hp_numpy.size] = hp_numpy * -amplitude
    hcarray[t_echo_steps:t_echo_steps+hc_numpy.size] = hc_numpy * -amplitude

    #Appending further echoes. 
    for j in range(1, int(n_echoes)):
        del_t_echo_steps = \
            int(round((t_echo + del_t_echo * j) * 1.0/timestep)) + first_idx
        damping_factor = amplitude * gamma**(j) * ((-1.0)**(j+1))
        # apply to hp
        echo_slice = slice(del_t_echo_steps, del_t_echo_steps + hp_numpy.size)
        hparray[echo_slice] += hp_numpy
        hparray[echo_slice] *= damping_factor
        # apply to hc
        echo_slice = slice(del_t_echo_steps, del_t_echo_steps + hc_numpy.size)
        hcarray[echo_slice] += hc_numpy
        hcarray[echo_slice] *= damping_factor

    # add the original waveform, if desired
    if include_imr:
        hparray[:length] += hp.numpy()
        hcarray[:length] += hc.numpy()

    hp = pycbc.types.TimeSeries(hparray, delta_t=timestep, epoch=hp.start_time)
    hc = pycbc.types.TimeSeries(hcarray, delta_t=timestep, epoch=hc.start_time)

    # apply the inclination angle: since we assume this was generated with
    # zero inclination, we have to remove that
    yp0, _ = utils.spher_harms(2, 2, 0.)
    yp, yc = utils.spher_harms(2, 2, inclination)
    hp *= yp / yp0
    hc *= yc / yp0

    return hp, hc
