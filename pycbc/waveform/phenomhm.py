#!/usr/bin/env python

# Copyright (C) 2018 Collin Capano, Sebastian Khan
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Temporary module for calling PhenomHM mode-by-mode, directly. This should
be removed once a standard method has been established in lalsimulation.
"""

import numpy
import lal
import lalsimulation as lalsim
from pycbc.types import FrequencySeries

all_modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
n_modes = len(all_modes)

def get_phenomhm(**input_params):
    """Generates an PhenomHM waveform. Requires modes list and phi_ref to be
    provided.
    
    Parameters
    ----------
    modes : list of tuples
        List of tuples giving (l,m) modes to generate. If not provided, or
        None, will use all available modes.
    phi_ref : float
        Default is 0.
    \**kwargs :
        The rest of the keyword args should be the same as given to
        ``get_fd_waveform``.
    """
    m1 = input_params['mass1'] * lal.MSUN_SI
    m2 = input_params['mass2'] * lal.MSUN_SI
    spin1x = input_params['spin1x']
    spin1y = input_params['spin1y']
    spin1z = input_params['spin1z']
    spin2x = input_params['spin2x']
    spin2y = input_params['spin2y']
    spin2z = input_params['spin2z']
    f_ref = input_params['f_ref']
    f_lower = input_params['f_lower']
    f_final = input_params['f_final']
    delta_f = input_params['delta_f']
    distance = input_params['distance'] * 1e6 * lal.PC_SI
    inclination = input_params['inclination']
    # check for precessing spins (not supported)
    if numpy.nonzero([spin1x, spin1y, spin2x, spin2y])[0].any():
        raise ValueError("non-zero x/y spins provided, but this is "
                         "aligned-spin approximant")
    try:
        phi_ref = input_params['phi_ref']
    except KeyError:
        phi_ref = 0.
    try:
        modes = input_params['modes']
        if not isinstance(modes, list):
            modes = [modes]
        modes = list(set(modes))
    except KeyError:
        modes = None
    # create the modes
    if modes is not None:
        ma = lalsim.SimInspiralCreateModeArray()
        for l,m in modes:
            lalsim.SimInspiralModeArrayActivateMode(ma, l, m)
        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, ma)
    else:
        params = None
        modes = all_modes
    # generate the hlms
    freqs = lal.CreateREAL8Sequence(2)
    freqs.data = [f_lower, f_final]
    hlms = lalsim.SimIMRPhenomHMGethlmModes(freqs, m1, m2, spin1z, spin2z,
                                            phi_ref, delta_f, f_ref, params)
    # convert the mode frequency series to pycbc FrequencySeries
    hplus = None
    hcross = None
    for l,m in modes:
        h = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
        ylm = lal.SpinWeightedSphericalHarmonic(inclination, 0., -2, l, m)
        ylnm = numpy.conj(lal.SpinWeightedSphericalHarmonic(
            inclination, 0., -2, l, -m))
        yplus = 0.5 * (ylm + (-1)**l * ylnm)
        ycross = 0.5j * (ylm - (-1)**l * ylnm)
        hp = FrequencySeries(yplus * h.data.data, delta_f=h.deltaF)
        hc = FrequencySeries(ycross * h.data.data, delta_f=h.deltaF)
        if hplus is None:
            hplus = hp
        else:
            hplus += hp
        if hcross is None:
            hcross = hc
        else:
            hcross += hc
    amp0 = lalsim.SimPhenomUtilsFDamp0((m1+m2)/lal.MSUN_SI, distance)
    return amp0*hplus, amp0*hcross


phenomhm_approximants = {"IMRPhenomHM_modes": get_phenomhm}
