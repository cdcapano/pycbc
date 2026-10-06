# Copyright (C) 2016 Collin Capano
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
""" Functions for applying gates to data.
"""

import numpy
from scipy import linalg
from pycbc.types import FrequencySeries
from . import strain


def _gates_from_cli(opts, gate_opt):
    """Parses the given `gate_opt` into something understandable by
    `strain.gate_data`.
    """
    gates = {}
    if getattr(opts, gate_opt) is None:
        return gates
    for gate in getattr(opts, gate_opt):
        try:
            ifo, central_time, half_dur, taper_dur = gate.split(':')
            central_time = float(central_time)
            half_dur = float(half_dur)
            taper_dur = float(taper_dur)
        except ValueError:
            raise ValueError("--gate {} not formatted correctly; ".format(
                gate) + "see help")
        try:
            gates[ifo].append((central_time, half_dur, taper_dur))
        except KeyError:
            gates[ifo] = [(central_time, half_dur, taper_dur)]
    return gates


def gates_from_cli(opts):
    """Parses the --gate option into something understandable by
    `strain.gate_data`.
    """
    return _gates_from_cli(opts, 'gate')


def psd_gates_from_cli(opts):
    """Parses the --psd-gate option into something understandable by
    `strain.gate_data`.
    """
    return _gates_from_cli(opts, 'psd_gate')


def apply_gates_to_td(strain_dict, gates):
    """Applies the given dictionary of gates to the given dictionary of
    strain.

    Parameters
    ----------
    strain_dict : dict
        Dictionary of time-domain strain, keyed by the ifos.
    gates : dict
        Dictionary of gates. Keys should be the ifo to apply the data to,
        values are a tuple giving the central time of the gate, the half
        duration, and the taper duration.

    Returns
    -------
    dict
        Dictionary of time-domain strain with the gates applied.
    """
    # copy data to new dictionary
    outdict = dict(strain_dict.items())
    for ifo in gates:
        outdict[ifo] = strain.gate_data(outdict[ifo], gates[ifo])
    return outdict


def apply_gates_to_fd(stilde_dict, gates):
    """Applies the given dictionary of gates to the given dictionary of
    strain in the frequency domain.

    Gates are applied by IFFT-ing the strain data to the time domain, applying
    the gate, then FFT-ing back to the frequency domain.

    Parameters
    ----------
    stilde_dict : dict
        Dictionary of frequency-domain strain, keyed by the ifos.
    gates : dict
        Dictionary of gates. Keys should be the ifo to apply the data to,
        values are a tuple giving the central time of the gate, the half
        duration, and the taper duration.

    Returns
    -------
    dict
        Dictionary of frequency-domain strain with the gates applied.
    """
    # copy data to new dictionary
    outdict = dict(stilde_dict.items())
    # create a time-domin strain dictionary to apply the gates to
    strain_dict = dict([[ifo, outdict[ifo].to_timeseries()] for ifo in gates])
    # apply gates and fft back to the frequency domain
    for ifo,d in apply_gates_to_td(strain_dict, gates).items():
        outdict[ifo] = d.to_frequencyseries()
    return outdict


def add_gate_option_group(parser):
    """Adds the options needed to apply gates to data.

    Parameters
    ----------
    parser : object
        ArgumentParser instance.
    """
    gate_group = parser.add_argument_group("Options for gating data")

    gate_group.add_argument("--gate", nargs="+", type=str,
                            metavar="IFO:CENTRALTIME:HALFDUR:TAPERDUR",
                            help="Apply one or more gates to the data before "
                                 "filtering.")
    gate_group.add_argument("--gate-overwhitened", action="store_true",
                            help="Overwhiten data first, then apply the "
                                 "gates specified in --gate. Overwhitening "
                                 "allows for sharper tapers to be used, "
                                 "since lines are not blurred.")
    gate_group.add_argument("--psd-gate", nargs="+", type=str,
                            metavar="IFO:CENTRALTIME:HALFDUR:TAPERDUR",
                            help="Apply one or more gates to the data used "
                                 "for computing the PSD. Gates are applied "
                                 "prior to FFT-ing the data for PSD "
                                 "estimation.")
    return gate_group


def gate_and_paint(data, lindex, rindex, invpsd, copy=True):
    """Gates and in-paints data using a Toeplitz solver.

    Parameters
    ----------
    data : TimeSeries
        The data to gate.
    lindex : int
        The start index of the gate.
    rindex : int
        The end index of the gate.
    invpsd : FrequencySeries
        The inverse of the PSD.
    copy : bool, optional
        Copy the data before applying the gate. Otherwise, the gate will
        be applied in-place. Default is True.

    Returns
    -------
    TimeSeries :
        The gated and in-painted time series.
    """
    # Uses the hole-filling method of
    # https://arxiv.org/pdf/1908.05644.pdf
    # Copy the data and zero inside the hole
    if copy:
        data = data.copy()
    data[lindex:rindex] = 0
    # get the over-whitened gated data
    tdfilter = invpsd.astype('complex').to_timeseries() * invpsd.delta_t
    owhgated_data = (data.to_frequencyseries() * invpsd).to_timeseries()

    # remove the projection into the null space
    proj = linalg.solve_toeplitz(tdfilter[:(rindex - lindex)],
                                 owhgated_data[lindex:rindex])
    data[lindex:rindex] -= proj
    return data

def invert_covariance(invpsd, lindex, rindex):
    """Calculate the uninverted covariance matrix.
    Parameters
    ----------
    invpsd : FrequencySeries
        The inverse of the PSD.
    lindex : int
        The start index of the gate.
    rindex : int
        The end index of the gate.

    Returns
    -------
    array :
        The uninverted covariance matrix associated with the inverse PSD in the
        time window [lindex, rindex].
    """
    tdfilter = invpsd.astype('complex').to_timeseries() * invpsd.delta_t
    mat = linalg.toeplitz(tdfilter[:(rindex-lindex)])
    invmat = linalg.inv(mat)
    return invmat

def gate_and_paint_matmul(data, lindex, rindex, invpsd, invmat=None, copy=True):
    """Gates and in-paints data using explicit matrix multiplication.

    Parameters
    ----------
    data : TimeSeries
        The data to gate.
    lindex : int
        The start index of the gate.
    rindex : int
        The end index of the gate.
    invpsd : FrequencySeries
        The inverse of the PSD.
    invmat : array, optional
        The uninverted covariance matrix. If None, calculate on function call.
    copy : bool, optional
        Copy the data before applying the gate. Otherwise, the gate will
        be applied in-place. Default is True.
    
    Returns
    -------
    TimeSeries :
        The gated and in-painted time series.
    """
    if copy:
        data = data.copy()
    data[lindex:rindex] = 0
    # get the over-whitened gated data
    owhgated_data = (data.to_frequencyseries() * invpsd).to_timeseries()

    # invert the matrix if not provided
    if invmat is None:
        invmat = invert_covariance(invpsd, lindex, rindex)

    # remove the projection into the null space
    proj = invmat @ owhgated_data[lindex:rindex]
    data[lindex:rindex] -= proj
    return data


def batch_gate_and_paint_fd(htildes, time, window, invpsd,
                            paint_method='toeplitz', invmat=None):
    """Gates and in-paints several frequency series at once.

    This gives the same result as calling
    ``h.to_timeseries().gate(time, window=window, method='paint', ...)``
    followed by ``to_frequencyseries()`` on each of the given frequency
    series, but does all of the FFTs along a single axis and the
    in-painting as a single matrix-matrix product (or a single Toeplitz
    solve with multiple right-hand sides). This is substantially faster when
    many series need to be gated with the same gate, since the covariance
    matrix only needs to be traversed once.

    Parameters
    ----------
    htildes : list of FrequencySeries
        The frequency series to gate. All must have the same length,
        ``delta_f``, and epoch.
    time : float
        Central time of the gate in seconds.
    window : float
        Half-length in seconds of the gate.
    invpsd : FrequencySeries
        The inverse of the PSD. Must be the same length as the series.
    paint_method : {'toeplitz', 'matmul'}
        Which method to use for in-painting the gated region.
    invmat : array, optional
        The inverted covariance matrix to use if ``paint_method='matmul'``.
        If None, it will be calculated from ``invpsd``.

    Returns
    -------
    list of FrequencySeries :
        The gated and in-painted series, in the same order as ``htildes``.
    """
    h0 = htildes[0]
    nfreq = len(h0)
    delta_f = float(h0.delta_f)
    tlen = 2 * (nfreq - 1)
    delta_t = 1. / (tlen * delta_f)
    start_time = float(h0.start_time)
    # same as TimeSeries.get_gate_indices
    lindex = max(int((time - window - start_time) / delta_t), 0)
    rindex = min(int((time + window - start_time) / delta_t), tlen)
    # time shift so that the end of the gate lands on a sample, as is done
    # in TimeSeries.gate
    offset = start_time + rindex * delta_t - (time + window)
    fdata = numpy.array([h.numpy() for h in htildes])
    if offset != 0:
        shift = numpy.exp(-2j * numpy.pi * offset
                          * h0.sample_frequencies.numpy())
        fdata = fdata * shift
    # to the time domain; tlen * delta_f * delta_t = 1 converts between
    # numpy's and pycbc's FFT normalizations
    tdata = numpy.fft.irfft(fdata, n=tlen, axis=1) * (tlen * delta_f)
    tdata[:, lindex:rindex] = 0
    # the over-whitened gated data
    owhgated = numpy.fft.irfft(numpy.fft.rfft(tdata, axis=1)
                               * invpsd.numpy(), n=tlen, axis=1)
    owhgated = owhgated[:, lindex:rindex].T
    # remove the projection into the null space
    if paint_method == 'toeplitz':
        tdfilter = invpsd.astype('complex').to_timeseries() * invpsd.delta_t
        proj = linalg.solve_toeplitz(tdfilter[:(rindex - lindex)].numpy(),
                                     owhgated)
    elif paint_method == 'matmul':
        if invmat is None:
            invmat = invert_covariance(invpsd, lindex, rindex)
        proj = invmat @ owhgated
    else:
        raise ValueError(f'Unrecognized paint_method input {paint_method}')
    tdata[:, lindex:rindex] -= proj.T
    # back to the frequency domain
    fdata = numpy.fft.rfft(tdata, axis=1) * delta_t
    if offset != 0:
        fdata *= shift.conj()
        # TimeSeries.gate returns a time series, which is real; this makes
        # the DC and Nyquist bins real after undoing the time shift
        fdata[:, 0] = fdata[:, 0].real
        fdata[:, -1] = fdata[:, -1].real
    return [FrequencySeries(fd, delta_f=h.delta_f, epoch=h.epoch)
            for fd, h in zip(fdata, htildes)]
