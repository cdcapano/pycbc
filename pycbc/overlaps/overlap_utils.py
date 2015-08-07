#! /usr/bin/env python

import sqlite3
import numpy
import os, sys, shutil, socket
import copy
import tempfile
import pickle
import h5py

import lal
import lalsimulation as lalsim
from glue import segments
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from pylal import ligolw_sqlutils as sqlutils

from pycbc import types as pytypes
from pycbc import filter
from pycbc import psd as pyPSD
from pycbc.overlaps import waveform_utils

from scipy import signal


#
#   Additional filtering functions
#

def filter_by_padding(htilde, stilde, psd, fmin, target_sample_rate,
        work_v1=None, work_v2=None, work_psd=None, high_frequency_cutoff=None,
        h_norm=None, out=None, corr_out=None):
    """
    A wrapper around pycbc.filter's matched_filter_core. This function
    zero-pads htilde and stilde out to the Nyquist frequency of the target
    sample rate before filtering. The resulting overlap time series is sampled
    at the target sample rate rather than the sample rate at which h and s were
    sampled (which should be smaller). This does not recover any power from
    frequencies higher than h and s's sampling rates; it assumes that either h
    or s have zero power in the frequencies above their Nyquist frequency.  The
    goal of this is to get the overlap in points in between the sampled points,
    so that the maximum can be recovered more precisely.
    
    If need be, the psd will also be padded, though 1's will be used. The psd's
    Nyquist must be greater than or equal to the minimum of htilde and stilde's
    Nyquist frequency. The Nyquist frequency of each series is assumed to be
    len(series)*series.delta_f.

    Note that the Nyquist frequencies (and thus the equivalent sampling rates)
    of htilde, stilde, and the psd do not need to be the same. They only need
    to have the same frequency resolution (that is, the same delta_f).

    If htilde or stilde have a Nyquist frequency larger than the target sample
    rate's, a ValueError is raised. If htilde and stilde have the same Nyquist
    frequency as the target sample rate's, they are passed directly to the
    matched_filter_core.

    Parameters
    ----------
    htilde: FrequencySeries
        The template waveform.
    stilde: FrequencySeries
        The injection or data.
    psd: (real) FrequencySeries
        The psd. The Nyquist frequency of this must be >= min(htilde's Nyquist,
        stilde's Nyquist).
    fmin: float
        The lower frequency cutoff to use when performing the filtering.
    target_sample_rate: int
        The desired sample rate of the matched-filter time series.
    work_v1: {None, FrequencySeries}, optional
        Work-space frequency series to use for htilde when performing
        the filtering. Must have the same frequency resolution as htilde, and
        must have a length N/2+1 where N = target_sample_rate/htilde.delta_f.
        If None specified, one will be generated.
    work_v2: {None, FrequencySeries}, optional
        Same as work_v1, but for stilde.
    work_psd: {None, (real) FrequencySeries}, optional
        Same as work_v1, but for the psd.
    high_frequency_cutoff: {None, float}, optional
        Passed through to matched_filter_core; see that documentation for help.
    h_norm: {None, float}, optional
        Passed through to matched_filter_core; see that documentation for help.
    out: {None, Array}, optional
        Passed through to matched_filter_core; see that documentation for help.
    corr_out: {None, Array}, optional
        Passed through to matched_filter_core; see that documentation for help.
        
    Returns
    -------
    snr : TimeSeries
        A time series containing the complex snr. This will have a sample rate
        equal to the target sample rate.
    corrrelation: FrequencySeries
        A frequency series containing the correlation vector. This will be
        zero-padded out to the Nyquist of the target sample rate.
    norm : float
        The normalization of the complex snr.  
    """
    if htilde.delta_f != stilde.delta_f or htilde.delta_f != psd.delta_f:
        raise ValueError(
            "htilde, stilde, and psd frequency resolution must match")
    # check that the psd is longer than atleast htilde or stilde
    if not len(psd) >= min(len(htilde), len(stilde)):
        raise ValueError(
            "PSD must go to the same frequency as either htilde or stilde")

    # get the desired N
    N = target_sample_rate * int(1./htilde.delta_f)
    kmax = N/2 + 1

    if len(htilde) > kmax:
        raise ValueError("htilde has higher Nyquist than target_sample_rate/2")
    if len(stilde) > kmax:
        raise ValueError("stilde has higher Nyquist than target_sample_rate/2")

    # zero out or create the work space vectors
    if len(htilde) < kmax:
        if work_v1 is None:
            work_v1 = pytypes.FrequencySeries(pytypes.zeros(kmax,
                dtype=pytypes.complex_same_precision_as(htilde)),
                delta_f=htilde.delta_f)
        elif not isinstance(work_v1, pytypes.FrequencySeries):
            raise TypeError("work_v1 must be a FrequencySeries")
        elif work_v1.delta_f != htilde.delta_f:
            raise ValueError("work_v1 delta_f does not match htilde")
        elif len(work_v1) != kmax:
            raise ValueError(
                "last frequency bin of work_v1 != target_sample_rate/2")
        else:
            work_v1.clear()
        # copy htilde data into work_v1. Note that we do not copy htilde's
        # Nyquist frequency bin, so as not to include it in the filtering
        work_v1.data[:len(htilde)-1] = htilde.data[:len(htilde)-1]
    else:
        # just use htilde since it already has the desired Nyquist
        work_v1 = htilde

    if len(stilde) < kmax:
        if work_v2 is None:
            work_v2 = pytypes.FrequencySeries(pytypes.zeros(kmax,
                dtype=pytypes.complex_same_precision_as(stilde)),
                delta_f=stilde.delta_f)
        elif not isinstance(work_v2, pytypes.FrequencySeries):
            raise TypeError("work_v2 must be a FrequencySeries")
        elif work_v2.delta_f != stilde.delta_f:
            raise ValueError("work_v2 delta_f does not match stilde")
        elif len(work_v2) != kmax:
            raise ValueError(
                "last frequency bin of work_v2 != target_sample_rate/2")
        else:
            work_v2.clear()
        # copy stilde data into work_v2. Note that we do not copy stilde's
        # Nyquist frequency bin, so as not to include it in the filtering
        work_v2.data[:len(stilde)-1] = stilde.data[:len(stilde)-1]
    else:
        # just use stilde since it already has the desired Nyquist
        work_v2 = stilde
    
    if len(psd) < kmax:
        if work_psd is None:
            work_psd = pytypes.FrequencySeries(pytypes.ones(kmax,
                dtype=pytypes.real_same_precision_as(psd)),
                delta_f=psd.delta_f)
        elif not isinstance(work_psd, pytypes.FrequencySeries):
            raise TypeError("PSD must be a FrequencySeries")
        elif work_psd.delta_f != psd.delta_f:
            raise ValueError("work_psd delta_f does not match psd")
        elif len(work_psd) != kmax:
            raise ValueError(
                "last frequency bin of work_psd != target_sample_rate/2")
        else:
            work_psd.data[:] = 1
        work_psd.data[:len(psd)] = psd.data[:]
    else:
        work_psd = psd[:kmax]

    # do the filtering
    return filter.matched_filter_core(work_v1, work_v2, work_psd, fmin,
        high_frequency_cutoff, h_norm, out, corr_out)



def upsample_timeseries(timeseries, target_idx, target_sample_rate,
        max_resample_length=2, copy=True):
    """
    Upsamples a time series about the target_idx to the target_sample_rate.  A
    portion of the time series is selected with the center on the target_idx.
    This portion is tapered to 0 using a Tukey window, then upsampled to the
    desired sample rate. The Tukey window used tapers the first and last 1/4 of
    the selected time seires. If the sample rate of the timeseries is >= the
    target sample rate, a ValueError is raised.

    Parameters
    ----------
    timeseries: pycbc.types.TimeSeries
        The time series to resample. The current sample rate is taken to be
        int(1/timeseries.delta_t).
    target_idx: int
        The index of the time series about which to do the upsampling. Must
        be in the range [N/4:3N/4), where N = the length of the time series.
        If it is not, an IndexError is raised.
    target_sample_rate: int
        The sample rate to upsample to. If the sample rate of the input
        time series is >= this value, a ValueError is raised.
    max_resample_length: {2, int} optional
        The maximum duration, in seconds, of the selected portion of the time
        series to resample. Default is 2. If the duration of the time series is
        less than this, the entire segment is resampled.
    copy: {True, bool}, optional
        If set to True, the portion of the timeseries that is resampled will
        be copied. If False, the portion of the timeseries that is resampled
        will be changed due to the windowing. Default is True.
        

    Returns
    -------
    resampled_series: pycbc.types.TimeSeries
        The portion of the time series resampled to the higher sampling rate.
        The epoch of this time series is adjusted to take into account the
        offset between the start of the resampled series and the start of
        the input series.
    """
    seg_length = timeseries.duration
    sample_rate = int(1./timeseries.delta_t)
    if sample_rate >= target_sample_rate:
        raise ValueError(
            "current sample rate (%i) is >= target sample rate (%i)" %(
            sample_rate, target_sample_rate))
    scale_factor = target_sample_rate / sample_rate
    if seg_length > max_resample_length:
        resample_length = max_resample_length
    else:
        resample_length = seg_length

    selected_N = sample_rate * resample_length

    if target_idx < selected_N/2 or target_idx > len(timeseries) - selected_N/2:
        raise IndexError(
            "target_idx (%i) must be in range " %(target_idx) +\
            "[sample_rate*resample_length/2 (%i), " %(selected_N/2) +\
            "len(timeseries) - sample_rate*resample_length/2 (%i)" %(
            len(timeseries) - selected_N/2))

    # generate the window
    window_N = selected_N / 2
    window = (1. - numpy.cos(
        numpy.arange(window_N, dtype=pytypes.real_same_precision_as(timeseries))\
        /window_N * 2 * numpy.pi))/2.

    if copy:
        use_ts = numpy.zeros(selected_N, dtype=timeseries.dtype)
        use_ts[:] = timeseries.data[target_idx-selected_N/2:\
            target_idx+selected_N/2]
    else:
        use_ts = timeseries.data[target_idx-selected_N/2:\
            target_idx+selected_N/2]
    use_ts[:window_N/2] *= window[:window_N/2]
    use_ts[-window_N/2:] *= window[-window_N/2:]

    # resample
    resampled_ts = signal.resample(use_ts.astype(numpy.complex128),
        selected_N * scale_factor)

    return pytypes.TimeSeries(resampled_ts.astype(timeseries.dtype),
        delta_t=1./target_sample_rate,
        epoch=timeseries.start_time + \
            (target_idx-selected_N/2)*timeseries.delta_t)


def get_common_Nyquist(htilde, stilde):
    """
    Given two frequency series, finds the largest Nyquist frequency
    between the two.
    """
    return max((len(htilde)-1)*htilde.delta_f, (len(stilde)-1)*stilde.delta_f)


def filter_by_resampling(htilde, stilde, psd, fmin, target_sample_rate,
        max_resample_length=2, check_error=False, high_frequency_cutoff=None,
        h_norm=None, out=None, corr_out=None, zero_pad_to_common_Nyquist=False,
        work_v1=None, work_v2=None, work_psd=None):
    """
    A wrapper around pycbc.filter's matched_filter_core that uses
    upsample_timeseries to upsample the cmplx_snr to the target sample rate if
    it is larger than the sample rate of cmplx_snr.  The cmplx_snr is upsampled
    around the maximum, with just the resampled region returned. If the sample
    rate of the cmplx_snr is >= target sample rate, the cmplx_snr is returned
    as-is. The savings as compared to filtering at the target sample rate
    depends on ratio of the target sample rate to the input rate (the scale
    factor) and the size of the resampling length to the length of the
    cmplx_snr time series. For a resampling length of 2s (the default) and a
    segment length of 256s, this function was found to be ~ (5/8) * the scale
    factor faster than filter_by_padding; e.g., with a scale factor of 4, this
    is 2.5 times faster.
    
    If htilde or stilde do not have the same Nyquist, this function will
    optionally zero-pad to the larger of the two frequency series' Nyquist
    using filter_by_padding. If this is not desired (see below), an error
    will be raised if the frequency series do not have the same Nyquist.

    Parameters
    ----------
    htilde: FrequencySeries
        The template waveform.
    stilde: FrequencySeries
        The injection or data.
    psd: (real) FrequencySeries
        The PSD.
    fmin: float
        The lower frequency cutoff to use when performing the filtering.
    target_sample_rate: int
        The desired sample rate of the matched-filter time series.
    max_resample_length: int
        Number of (integer) seconds to use for resampling. Default is 2.
        See the documentation of upsample_timeseries for more details.
    check_error: bool, optional
        If set to True, the maximum of abs(cmplx_snr*norm) of the resampled
        series is compared to the result from filter_by_padding. *WARNING:*
        turning this on will negate any computational savings. This
        option is only meant as a check. Default is False.
    high_frequency_cutoff: {None, float}, optional
        Passed through to matched_filter_core; see that documentation for help.
    h_norm: {None, float}, optional
        Passed through to matched_filter_core; see that documentation for help.
    out: {None, Array}, optional
        Passed through to matched_filter_core; see that documentation for help.
    corr_out: {None, Array}, optional
        Passed through to matched_filter_core; see that documentation for help.
    zero_pad_to_common_Nyquist: bool, optional
        If set to True and stilde and htilde do not have the same Nyquist
        frequency (i.e., one frequency series is shorter than the other),
        than the shorter one will be zero-padded to the same length as the
        longer one then filtered using filter_by_padding. This assumes
        that the shorter one has no power at frequencies greater than the
        given Nyquist. The resulting cmplx_snr time series will then be
        upsampled to the target sample rate. If this is set to False,
        an error will be raised if stilde and htilde are not the same length.
    work_v1: {None, FrequencySeries}, optional
        If zero_pad_to_common_Nyquist is set to True, this is passed to
        work_v1 option in filter_by_padding to speed up filtering on repeated
        calls. The Nyquist frequency of this vector must be the same as the
        common Nyquist between htilde and stilde, which can be found using
        get_common_Nyquist. If this is not provided, an appropriately sized
        vector will be generated. See filter_by_padding for more details.
    work_v2: {None, FrequencySeries}, optional
        Same as work_v1, but for stilde. See filter_by_padding for more
        details.
    work_psd: {None, (real) FrequencySeries}, optional
        Same as work_v1, but for the psd. See filter_by_padding for more
        details.

    Returns
    -------
    snr : TimeSeries
        A time series containing the complex snr. This will have a sample rate
        >= to the target sample rate. If the sample rate is >= the target rate,
        the cmplx_snr will be the full time series. If the sample rate < than
        the target sample rate (meaning resampling was done), the cmplx_snr
        will only consist of the resampled part.
    corrrelation: FrequencySeries
        A frequency series containing the correlation vector. The length of
        this will be equal to the segment length times the *original*
        sample_rate, not the resampled rate (i.e., it is the correlation vector
        you get from matched_filter_core).
    norm : float
        The normalization of the complex snr.  
    maxidx: int
        The index of the maximum of abs(cmpx_snr)*norm.
    offset: float
        The offset, in seconds, between the original cmplx_snr and the
        resampled.
    frac_error: {float, None}
        If check_error turned on, 1 - resampled_max / check_max, where
        resampled_max is the max of resampled series and check_max is the
        maximum found by using filter_by_padding.
    """
    if zero_pad_to_common_Nyquist:
        intermediate_rate = 2*get_common_Nyquist(htilde, stilde)
        cmplx_snr, corr, norm = filter_by_padding(htilde, stilde, psd, fmin,
            intermediate_rate, work_v1, work_v2, work_psd,
            high_frequency_cutoff, h_norm, out, corr_out)
    else:
        cmplx_snr, corr, norm = filter.matched_filter_core(htilde, stilde, psd,
            fmin, high_frequency_cutoff, h_norm, out, corr_out)

    maxidx = (norm*abs(cmplx_snr)).data.argmax()

    # resample if necessary
    sample_rate = int(1./cmplx_snr.delta_t)
    if sample_rate < target_sample_rate:
        orig_start_time = cmplx_snr.start_time
        cmplx_snr = upsample_timeseries(cmplx_snr, maxidx,
            target_sample_rate, max_resample_length, copy=False)
        maxidx = (norm*abs(cmplx_snr)).data.argmax()
        offset = cmplx_snr.start_time - orig_start_time
    else:
        offset = 0.

    if check_error:
        check_snr, _, check_norm = filter_by_padding(htilde, stilde, psd, fmin,
            target_sample_rate)
        check_snr = (check_norm*abs(check_snr)).max()
        frac_error = 1. - (norm*abs(cmplx_snr[maxidx]))/check_snr
    else:
        frac_error = None

    return cmplx_snr, corr, norm, maxidx, offset, frac_error


#
#   Weight functions
#
def uniform(*args, **kwargs):
    """
    Just returns 1, regardless of what arguments are passed.
    """
    return 1.


def noAntiAligned(tmplt, *args, **kwargs):
    """
    Returns 0 if either z-component of the template's spins are < 0. Otherwise
    returns 1.
    """
    if tmplt.spin1z < 0 or tmplt.spin2z < 0:
        return 0.
    else:
        return 1.

def weight_by_volume(tmplt, tmplt_prime, use_tmplt_sigma=False,
        use_tmplt_prime_sigma=False, ifo=None, fmin=None, psd_model=None,
        asd_file=None,
        workspace=None):
    """
    Generates the weight: 

        w   = V / V'
            = (\sigma / \sigma')^3
            = (<h, h> / <h', h'>)^(3/2),

    where h is the waveform associated with the given tmplt and h' is the
    waveform associated with the given tmplt_prime.

    When calculating sigmas, the minimum possible sample rate and segment
    length for h_tmplt and h' will be used. For this reason, the PSD is
    generated on the fly so as to ensure proper frequency resolution and
    Nyquist frequencies for each.

    Parameters
    ----------
    tmplt: waveform_utils.Template
        The template used in the numerator.
    tmplt_prime: waveform_utils.Template
        The template used in the denominator.
    use_tmplt_sigma: bool
        If set to True, the sigma property of tmplt will be used for sigma.
        Otherwise, sigma is calculated. Default is False.
    use_tmplt_prime_sigma: bool
        If set to True, the sigma property of tmplt_prime will be used for
        sigma'. Otherwise, sigma' is calculated. Default is False.
    ifo: {None, str}
        The ifo to use for the sigmas. Must be set if "use_tmplt_sigma" or
        "use_tmplt_prime_sigma" is set to True.
    fmin: float
        The starting frequency to use for the overlaps. Needed if
        either use_tmplt_sigma or use_tmplt_prime_sigma is False.
    psd_model: {None, string}
        The name of the PSD model to use when calculating sigma and/or sigma'.
    asd_file: {None, string}
        The name of the asd file to use from which to get the PSD. Either this
        or psd_model must be specified if sigma or sigma' need to be
        calculated.
    workspace: {None, WorkSpace}
        Optional. Provide a workspace instance from which to retrieve the PSD.
        If the PSD with the correct sample rate and segment length exists in
        the work space, it will be retrieved instead of being generated.
        Otherwise, the psd will be generated and stored to the workspace when
        calculating sigma and/or sigma'.

    Returns
    -------
    weight: float
        The weight to use.
    """
    if not (use_tmplt_sigma and use_tmplt_prime_sigma):
        if psd_model is None and asd_file is None:
            raise ValueError("must provide either a psd_model or asd_file " +
                "to use")
        if workspace is None:
            # generate a temporary workspace
            workspace = WorkSpace()

    # get sigma
    if use_tmplt_sigma: 
        if tmplt.sigma(ifo) is None:
            raise ValueError("tmplt's sigma property must be populated when "+
                "use_tmplt_sigma is True")
        sigma = tmplt.sigma(ifo)
    else:
        htilde = tmplt.get_fd_waveform(tmplt.min_sample_rate,
            tmplt.min_seg_length, store=False) 
        if psd_model is not None:
            psd = workspace.get_psd(tmplt.f_min, tmplt.min_sample_rate,
                tmplt.min_seg_length, psd_model, dyn_range_exp=0)
        else:
            psd = workspace.get_psd_from_file(tmplt.f_min,
                tmplt.min_sample_rate, tmplt.min_seg_length, asd_file,
                dyn_range_exp=0) 
        sigma = filter.sigma(htilde, psd, fmin) 

    # get sigma'
    if use_tmplt_prime_sigma: 
        if tmplt_prime.sigma is None:
            raise ValueError("tmplt_prime's sigma property must be populated "+
                "when use_tmplt_prime_sigma is True")
        sigma_prime = tmplt_prime.sigma(ifo)
    else:
        htilde_prime = tmplt_prime.get_fd_waveform(tmplt_prime.min_sample_rate,
            tmplt_prime.min_seg_length, store=False)
        if psd_model is not None:
            psd = workspace.get_psd(tmplt_prime.f_min,
                tmplt_prime.min_sample_rate, tmplt_prime.min_seg_length,
                psd_model, dyn_range_exp=0)
        else:
            psd = workspace.get_psd_from_file(tmplt_prime.f_min,
                tmplt_prime.min_sample_rate, tmplt_prime.min_seg_length,
                asd_file, dyn_range_exp=0) 
        sigma_prime = filter.sigma(htilde_prime, psd, fmin)

    return (sigma / sigma_prime)**3.


def equalMassVol(tmplt, fmin, psd_model=None, asd_file=None, workspace=None,
        use_tmplt_sigma=False, ifo=None):
    """
    Calls weight_by_volume with tmplt set to the given tmplt and tmplt_prime
    set to a template with the same total mass as tmplt, but with the
    component masses equal.

    Parameters
    ----------
    tmplt: waveform_utils.Template
        The template for which to calculate the weight.
    fmin: float
        Required. The starting frequency to use for the overlaps.
    psd_model: {None, string}
        The name of the PSD model to use when calculating sigma.
    asd_file: {None, string}
        The name of the asd file to use from which to get the PSD. Either this
        or psd_model must be specified.
    workspace: {None, WorkSpace}
        Optional. Provide a workspace instance from which to retrieve the PSD.
        If the PSD with the correct sample rate and segment length exists in
        the work space, it will be retrieved instead of being generated.
        Otherwise, the psd will be generated and stored to the workspace.
    use_tmplt_sigma: bool
        If set to True, the sigma property of tmplt will be used for sigma.
        Otherwise, sigma is calculated. Default is False.
    ifo: {None, str}
        The ifo to use for the sigmas. Must be set if "use_tmplt_sigma" is set
        to True.

    Returns
    -------
    weight: float
        The weight to use.
    """
    # copy the template and set the masses to equal
    tmplt_prime = copy.deepcopy(tmplt)
    tmplt_prime.mass1 = tmplt_prime.mass2 = tmplt.mtotal/2.
    tmplt_prime.set_f_final()

    return weight_by_volume(tmplt, tmplt_prime,
        use_tmplt_sigma=use_tmplt_sigma, use_tmplt_prime_sigma=False,
        ifo=ifo, fmin=fmin, psd_model=psd_model, asd_file=asd_file,
        workspace=workspace)


def nonSpinVol(tmplt, fmin, psd_model=None, asd_file=None, workspace=None,
        use_tmplt_sigma=False, ifo=None):
    """
    Calls weight_by_volume with tmplt set to the given tmplt and tmplt_prime
    set to a template with the same component masses as the given
    template, but with all components of the spin set to 0.

    Parameters
    ----------
    tmplt: waveform_utils.Template
        The template for which to calculate the weight.
    fmin: float
        The starting frequency to use for the overlaps.
    psd_model: {None, string}
        The name of the PSD model to use when calculating sigma.
    asd_file: {None, string}
        The name of the asd file to use from which to get the PSD. Either this
        or psd_model must be specified.
    workspace: {None, WorkSpace}
        Optional. Provide a workspace instance from which to retrieve the PSD.
        If the PSD with the correct sample rate and segment length exists in
        the work space, it will be retrieved instead of being generated.
        Otherwise, the psd will be generated and stored to the workspace.
    use_tmplt_sigma: bool
        If set to True, the sigma property of tmplt will be used for sigma.
        Otherwise, sigma is calculated. Default is False.
    ifo: {None, str}
        The ifo to use for the sigmas. Must be set if "use_tmplt_sigma" is set
        to True.

    Returns
    -------
    weight: float
        The weight to use.
    """
    # copy the template and set the spins to 0
    tmplt_prime = copy.deepcopy(tmplt)
    tmplt_prime.spin1x = tmplt_prime.spin2x \
        = tmplt_prime.spin1y = tmplt_prime.spin2y \
        = tmplt_prime.spin1z = tmplt_prime.spin2z \
        = 0.
    tmplt_prime.set_f_final()

    return weight_by_volume(tmplt, tmplt_prime,
        use_tmplt_sigma=use_tmplt_sigma, use_tmplt_prime_sigma=False,
        ifo=ifo, fmin=fmin, psd_model=psd_model, asd_file=asd_file,
        workspace=workspace)


def equalMassNonSpinVol(tmplt, fmin, psd_model=None, asd_file=None,
        workspace=None, use_tmplt_sigma=False, ifo=None):
    """
    Calls weight_by_volume with tmplt set to the given tmplt and tmplt_prime
    set to a template with the same total mass as the given template, but
    with equal component masses and with all components of the spin set to 0.

    Parameters
    ----------
    tmplt: waveform_utils.Template
        The template for which to calculate the weight.
    fmin: float
        The starting frequency to use for the overlaps.
    psd_model: {None, string}
        The name of the PSD model to use when calculating sigma.
    asd_file: {None, string}
        The name of the asd file to use from which to get the PSD. Either this
        or psd_model must be specified.
    workspace: {None, WorkSpace}
        Optional. Provide a workspace instance from which to retrieve the PSD.
        If the PSD with the correct sample rate and segment length exists in
        the work space, it will be retrieved instead of being generated.
        Otherwise, the psd will be generated and stored to the workspace.
    use_tmplt_sigma: bool
        If set to True, the sigma property of tmplt will be used for sigma.
        Otherwise, sigma is calculated. Default is False.
    ifo: {None, str}
        The ifo to use for the sigmas. Must be set if "use_tmplt_sigma" is set
        to True.

    Returns
    -------
    weight: float
        The weight to use.
    """
    # copy the template and set the masses to equal, spins to 0.
    tmplt_prime = copy.deepcopy(tmplt)
    tmplt_prime.mass1 = tmplt_prime.mass2 = tmplt.mtotal/2.
    tmplt_prime.spin1x = tmplt_prime.spin2x \
        = tmplt_prime.spin1y = tmplt_prime.spin2y \
        = tmplt_prime.spin1z = tmplt_prime.spin2z \
        = 0.
    tmplt_prime.set_f_final()

    return weight_by_volume(tmplt, tmplt_prime,
        use_tmplt_sigma=use_tmplt_sigma, use_tmplt_prime_sigma=False,
        ifo=ifo, fmin=fmin, psd_model=psd_model, asd_file=asd_file,
        workspace=workspace)


#
#   dictionary to map lables to weight functions
#
weight_functions = {
    "uniform": uniform,
    "no_anti_aligned": noAntiAligned,
    "equal_mass_vol": equalMassVol,
    "non_spin_vol": nonSpinVol,
    "equal_mass_non_spin_vol": equalMassNonSpinVol
}



#
#   Standard file names
#
def get_outfilename(output_directory, ifo, user_tag=None, num=None):
    tag = ''
    if user_tag is not None:
        tag = '_%s' % user_tag
    if num is not None:
        tag += '-%i' % (num)
    if ifo is None:
        ifo = 'ND'
    return '%s/%s-OVERLAPS%s.sqlite' %(output_directory, ifo.upper(), tag)


def get_exval_outfilename(output_directory, ifo, user_tag = None, num = None):
    tag = ''
    if user_tag is not None:
        tag = '_%s' % user_tag
    if num is not None:
        tag += '-%i' % (num)
    if ifo is None:
        ifo = 'ND'
    elif len(ifo.split(',')) > 1:
        ifo = ''.join(map(str.strip, ifo.split(',')))
    return '%s/%s-EXPECTATION_VALUES%s.sqlite' %(output_directory, ifo.upper(),
        tag)


def copy_to_output(infile, outfile, verbose=False, warn_overwrite=True):
    if verbose:
        print >> sys.stdout, "Copying %s to %s..." %(infile, outfile)
    if os.path.exists(outfile) and warn_overwrite:
        print >> sys.stderr, "Warning: File %s being overwritten." %(outfile)
    shutil.copyfile(infile, outfile)


def get_outfile_connection(infile, outfile, replace = False, verbose = False):
    if infile == outfile:
        raise ValueError, "Input file %s is same as output" % infile
    # check if outfile exists; if so, and not replace,
    # we will copy the working_file from the outfile; otherwise, we will get
    # the working_file from the infile
    #if os.path.exists(outfile) and not replace:
    #    copy_to_output(outfile, working_outfile)
    #else:
    #    copy_to_output(infile, working_outfile)

    #connection = sqlite3.connect(working_outfile)
    if not os.path.exists(outfile) or replace:
        copy_to_output(infile, outfile)

    connection = sqlite3.connect(outfile)

    return connection #working_outfile, connection


def get_template_archive_fname(output_directory, min_sample_rate, max_sample_rate, user_tag = None):
    if user_tag is not None:
        tag = '_'+user_tag
    else:
        tag = ''
    archiveFnTemplate = '%s/HL-TEMPLATE_ARCHIVE%s-%i-%i.hdf5'
    return archiveFnTemplate %(output_directory, tag, min_sample_rate, max_sample_rate)


def get_asd_file_df(asd_file):
    f = open(asd_file, 'r')
    f1, _ = map(float, f.readline().split('  '))
    f2, _ = map(float, f.readline().split('  '))
    f.close()
    return f2 - f1

def get_psd_models():
    return pyPSD.get_lalsim_psd_list()

class WorkSpace:
    def __init__(self):
        self.psds = {}
        self.out_vecs = {}
        self.corr_vecs = {}
        self.work_FS_vecs = {}

    def get_psd(self, fmin, sample_rate, segment_length, psd_model,
            dyn_range_exp=0):
        try:
            return self.psds[psd_model, sample_rate, segment_length,
                dyn_range_exp]
        except KeyError:
            self.psds[psd_model, sample_rate, segment_length, dyn_range_exp] =\
                getattr(pyPSD, psd_model)((segment_length*sample_rate)/2 + 1,
                    1./segment_length, fmin) * 2.**(2*dyn_range_exp)
            return self.psds[psd_model, sample_rate, segment_length,
                dyn_range_exp]

    def get_psd_from_file(self, fmin, sample_rate, segment_length, filename,
            is_asd_file=True, dyn_range_exp=0):
        try:
            return self.psds[filename, sample_rate, segment_length,
                dyn_range_exp]
        except KeyError:
            try:
                self.psds[filename,sample_rate,segment_length,dyn_range_exp] =\
                    pyPSD.from_txt(filename, (sample_rate*segment_length)/2+1,
                    1./segment_length, fmin, is_asd_file) * \
                    2.**(2*dyn_range_exp)
            except ValueError:
                # if the requested max frequency is larger than the largest
                # value in the asd file, we will get an interpolation error;
                # in that case, we'll just go up to the max frequency in the
                # asd file, then just do infs after
                max_freq = numpy.loadtxt(filename)[-1,0]
                max_sample_rate = max_freq * 2
                tmp_psd = pyPSD.from_txt(filename,
                    (max_sample_rate*segment_length)/2+1,
                    1./segment_length, fmin, is_asd_file)*2.**(2*dyn_range_exp)
                psd = pytypes.FrequencySeries(numpy.zeros((
                    sample_rate*segment_length)/2+1),
                    delta_f=tmp_psd.delta_f)
                psd.data[:] = numpy.inf
                psd.data[:len(tmp_psd)] = tmp_psd.data[:]
                self.psds[filename, sample_rate, segment_length,
                    dyn_range_exp] = psd
            return self.psds[filename, sample_rate, segment_length,
                dyn_range_exp]

    def clear_psds(self):
        self.psds.clear()

    def get_out_vec(self, N, dtype):
        try:
            return self.out_vecs[N, dtype]
        except KeyError:
            self.out_vecs[N, dtype] = pytypes.zeros(N, dtype=dtype)
            return self.out_vecs[N, dtype]

    def get_corr_vec(self, N, dtype):
        try:
            return self.corr_vecs[N, dtype]
        except KeyError:
            self.corr_vecs[N, dtype] = pytypes.zeros(N, dtype=dtype)
            return self.corr_vecs[N, dtype]

    def get_work_FS_vec(self, label, kmax, seg_length, dtype):
        try:
            return self.work_FS_vecs[label, kmax, seg_length, dtype]
        except KeyError:
            self.work_FS_vecs[label, kmax, seg_length, dtype] = \
                pytypes.FrequencySeries(pytypes.zeros(kmax, dtype=dtype),
                delta_f=1./seg_length)
            return self.work_FS_vecs[label, kmax, seg_length, dtype]


def new_snr(snr, chisq, chisq_dof):
    rchisq = float(chisq) / chisq_dof
    if rchisq < 1:
        newsnr = snr
    else:
        newsnr = snr/ ((1+rchisq**3.)/2.)**(1./6)
    return newsnr

class ParamWindow(segments.segment):
    """
    A glue.segments.segment that adds additional arguments to quickly check if
    the value of a parameter lies within the desired range. The values of the
    segment give the range of parameters for which the window is valid.  The
    properties min_jitter and max_jitter give the fractional size of the
    window that is used. The function recovery_window gives the size of the
    window given some parameter window. For example, if you wanted to only
    filter templates that were within -10%, +20% of an injection's chirp mass
    for injections that had chirp masses between 8 and 12 solar masses, you
    would create the following parameter window:

    >>> pwin = overlap_utils.ParamWindow(8., 12.)
    >>> pwin
        segment(8.0, 12.0)
    >>> pwin.set_jitter(-0.1, 0.2)
   
    Now say you have an injection with chirp mass = 9.2 solar masses. The range
    of values that lie within the desired range is:

    >>> pwin.recovery_window(9.2)
        segment(8.28, 11.04)

    If you want to know whether or not a value is within the window of a given
    chirp mass:

    >>> pwin.in_recovered(9.2, 7.)
        False
    >>> pwin.in_recovered(9.2, 8.6)
        True

    If you try a value that is out of the range of validity, you get an error:

    >>> pwin.recovery_window(6.)
    ValueError: 6.000000 is not in the region of validity [8.000000, 12.000000)
    """

    min_jitter = None
    max_jitter = None

    def set_jitter(self, min_jitter, max_jitter):
        self.min_jitter = min_jitter
        self.max_jitter = max_jitter

    def recovery_window(self, value):
        if value not in self:
            raise ValueError("%f is not in the region of validity [%f, %f)" %(
                value, self[0], self[1]))
        if self.min_jitter is None or self.max_jitter is None:
            raise ValueError("min and max jitter are not set")
        return segments.segment(value*(1. + self.min_jitter),
            value*(1. + self.max_jitter))

    def in_recovered(self, reference_value, value):
        return value in self.recovery_window(reference_value)

    @property
    def min_injected(self):
        return self[0]

    @property
    def max_injected(self):
        return self[1]

    def min_recovered(self, value):
        return self.recovery_window(value)[0]

    def max_recovered(self, value):
        return self.recovery_window(value)[1]

    @property
    def mean_injected(self):
        return self.min_injected + abs(self)/2.

class ParamWindowList(segments.segmentlist):
    param = None
    inj_apprx = None
    tmplt_apprx = None

    def set_param(self, param):
        self.param = param

    def set_approximants(self, inj_apprx, tmplt_apprx):
        self.inj_apprx = inj_apprx
        self.tmplt_apprx = tmplt_apprx

    def find(self, val):
        try:
            return segments.segmentlist.find(self, val)
        except ValueError:
            if val < self[0][0]:
                return 0
            else:
                return len(self)-1

    def write_to_database(self, connection, min_s1z = -numpy.inf, min_s2z = -numpy.inf, max_s1z = numpy.inf, max_s2z = numpy.inf):
        sqlquery = """
            CREATE TABLE IF NOT EXISTS match_windows (param, inj_apprx, tmplt_apprx, min_injected, max_injected, min_injected_spin1z, max_injected_spin1z, min_injected_spin2z, max_injected_spin2z, min_jitter, max_jitter);
            CREATE INDEX IF NOT EXISTS mw_pn_injtmpltapprx_minj_manj_idx ON match_windows (param, inj_apprx, tmplt_apprx, min_injected, max_injected);
            """
        connection.cursor().executescript(sqlquery)
        sqlquery = 'INSERT INTO match_windows (param, inj_apprx, tmplt_apprx, min_injected, max_injected, min_injected_spin1z, max_injected_spin1z, min_injected_spin2z, max_injected_spin2z, min_jitter, max_jitter) VALUES (?,?,?,?,?,?,?,?,?,?,?)'
        insert_vals = [(self.param, self.inj_apprx, self.tmplt_apprx, pw.min_injected, pw.max_injected, min_s1z, max_s1z, min_s2z, max_s2z, pw.min_jitter, pw.max_jitter) for pw in self]
        connection.cursor().executemany(sqlquery, insert_vals)

    def load_from_database(self, connection, param, inj_apprx=None,
            tmplt_apprx=None, min_s1z=None, min_s2z=None, max_s1z=None,
            max_s2z=None):
        self.param = param
        self.inj_apprx = inj_apprx
        self.tmplt_apprx = tmplt_apprx
        sqlquery = """
            SELECT
                min_injected, max_injected, min_jitter, max_jitter
            FROM
                match_windows
            WHERE
                param == ?"""
        select_params = [param]
        if self.inj_apprx is not None:
            sqlquery += """ AND
                inj_apprx == ?"""
            select_params.append(self.inj_apprx)
        if self.tmplt_apprx is not None:
            sqlquery += """ AND
                tmplt_apprx == ?"""
            select_params.append(self.tmplt_apprx)
        if min_s1z is not None:
            sqlquery += ' AND min_injected_spin1z == ?'
            select_params.append(min_s1z)
        if min_s2z is not None:
            sqlquery += ' AND min_injected_spin2z == ?'
            select_params.append(min_s2z)
        if max_s1z is not None:
            sqlquery += ' AND max_injected_spin1z == ?'
            select_params.append(max_s1z)
        if max_s2z is not None:
            sqlquery += ' AND max_injected_spin2z == ?'
            select_params.append(max_s2z)
        sqlquery += ' ORDER BY min_injected ASC'
        for min_injected, max_injected, min_jitter, max_jitter in \
                connection.cursor().execute(sqlquery, tuple(select_params)):
            pw = ParamWindow(min_injected, max_injected)
            pw.set_jitter(min_jitter, max_jitter)
            self.append(pw)


class OverlapResult:
    params = ['ifo', 'effectualness', 'snr', 'chisq', 'new_snr', 'chisq_dof',
        'time_offset', 'time_offset_ns', 'snr_std', 'chisq_std', 'new_snr_std',
        'num_tries', 'num_successes', 'sample_rate', 'segment_length',
        'tmplt_approximant', 'overlap_f_min', 'waveform_f_min',
        'weight_function', 'weight']
    __slots__ = params + ['template', 'injection']
    
    def __init__(self, template, injection):
        self.template = template
        self.injection = injection
        for param in self.params:
            setattr(self, param, None)

    @property
    def gps_time(self):
        return self.injection.detector_end_time(self.ifo) + \
            lal.LIGOTimeGPS(self.time_offset, self.time_offset_ns)

    def write_to_database(self, connection, coinc_event_id,
            table_name='overlap_results'):
        write_params = [param for param in self.params if param in dir(self)]
        sqlquery = """
            INSERT INTO
                %s (coinc_event_id, %s)""" %(table_name,
                    ', '.join(write_params)) + """
            VALUES
                (?,%s)""" %(','.join(['?' for ii in range(len(write_params))]))

        connection.cursor().execute(sqlquery, tuple([coinc_event_id] + \
            [getattr(self, col) for col in write_params]))

    def write_all_results_to_database(self, connection):
        """
        Writes the parameters to the all_results table. Does not use a
        coinc_event_id, however.
        """
        write_params = [param for param in self.params if param in dir(self)]+\
            ['simulation_id', 'tmplt_id']
        write_results = tuple([getattr(self, col) for col in self.params \
            if param in dir(self)] + [self.injection.simulation_id,
            self.template.tmplt_id])
        sqlquery = 'INSERT INTO all_results (%s) VALUES (%s)' %(
            ', '.join(write_params),
            ','.join(['?' for ii in range(len(write_params))]))
        connection.cursor().execute(sqlquery, write_results)


def get_injection_template_map(connection):
    """
    Gets injections and templates that are mapped to each other.

    Parameters
    ----------
    connection: sqlite3 connection
        A connection to a SQLite database. The database must have
        a sngl_inspiral table containing templates, a sim_inspiral
        table containing injections, and a coinc_event_map table
        linking the two together.

    Returns
    -------
    template/injection map: dict
        A dictionary keyed by the coinc_event_id of the maps.
        Elements are a tuple of the injection's simulation_id and
        the template's template_id.
    """
    sqlquery = """
        SELECT
            mapA.coinc_event_id,
            sim_inspiral.simulation_id,
            sngl_inspiral.event_id
        FROM
            coinc_event_map AS mapA
        JOIN
            sngl_inspiral, sim_inspiral, coinc_event_map AS mapB
        ON
            mapA.event_id == sim_inspiral.simulation_id AND
            mapA.coinc_event_id == mapB.coinc_event_id AND
            mapB.event_id == sngl_inspiral.event_id
        """
    return dict([ [ceid, (simid, tmpltid)] for ceid, simid, tmpltid in \
        connection.cursor().execute(sqlquery)])
        

def get_best_match(connection, simulation_id, weight_func='uniform'):
    """
    Gets the best matching template to the given injection from the all_results
    table.
    
    Parameters
    ----------
    connection: sqlite3 connection
        A connection to a sqlite database.
    simulation_id: str
        The simulation_id of the injection to get the best match for.
    weight_func: {'uniform' | str}
        The weight function to use. Default is uniform.

    Returns
    -------
    best_match: OverlapResult instance or None
        If a best match is found, returns an OverlapResult instance with
        the effectualness, weight, weight_function, time_offset,
        time_offset_ns, segment_length, and sample_rate populated. The
        injection field and template field are set to the simulation_id and
        the template's id, respectively. If no best match is found (meaning
        that the injection hasn't been filtered yet), None is returned.
    """
    best_match_query = """
        SELECT
            a.simulation_id, a.tmplt_id, a.effectualness, w.weight,
            a.time_offset, a.time_offset_ns, a.segment_length, a.sample_rate
        FROM
            all_results AS a
        JOIN
            tmplt_weights AS w
        ON
            w.tmplt_id == a.tmplt_id
        WHERE
            a.simulation_id == ? AND
            w.weight_function == ?
        ORDER BY
            a.effectualness * w.weight
        DESC LIMIT 1"""
    result =  connection.cursor().execute(best_match_query,
        (simulation_id, weight_func)).fetchone()
    if result is not None:
        simid, tmpltid, eff, wt, dt, dt_ns, seg_length, sample_rate = result
        # note, we're just storing the simulation_id and the tmplt_id as
        # as the OverlapResult's injection and template parameters, not the
        # injection and the template itself
        bestMatch = OverlapResult(tmpltid, simid)
        bestMatch.effectualness = eff
        bestMatch.weight = wt
        bestMatch.weight_function = weight_func
        bestMatch.time_offset = dt
        bestMatch.time_offset_ns = dt_ns
        bestMatch.segment_length = seg_length
        bestMatch.sample_rate = sample_rate
    else:
        bestMatch = None
    return bestMatch 


def prune_all_results(connection, injections, weight_functions=['uniform'],
        vacuum=True, verbose=False):
    """
    Deletes results from the all_results table that are not the best matches
    to each injection.
    """
    if verbose:
        print >> sys.stdout, "pruning all_results table:"
    crs = connection.cursor()
    # create a temporary table to store the results to keep
    if verbose:
        print >> sys.stdout, "getting ids to keep..."
    sqlquery = """CREATE TEMP TABLE saved_ids (simulation_id, tmplt_id)"""
    crs.execute(sqlquery)
    saved_ids = []
    for inj in injections.as_list:
        for weight_func in weight_functions:
            best_match = get_best_match(connection, str(inj.simulation_id),
                weight_func)
            if best_match is not None:
                saved_ids.append((best_match.injection, best_match.template))
    # insert the best match ids into the database
    sqlquery = "INSERT INTO saved_ids (simulation_id, tmplt_id) VALUES (?,?)"
    crs.executemany(sqlquery, saved_ids)
    # add an index for faster deleting
    crs.execute("""
        CREATE INDEX tmp_sidtid_idx ON saved_ids (simulation_id, tmplt_id)
        """)
    # now delete the entries from all_results that don't match
    if verbose:
        print >> sys.stdout, "deleting entries from all_results..."
    sqlquery = """
        DELETE FROM
            all_results
        WHERE NOT EXISTS (
            SELECT
                *
            FROM
                saved_ids
            WHERE
                saved_ids.simulation_id == all_results.simulation_id AND
                saved_ids.tmplt_id == all_results.tmplt_id)
    """
    crs.execute(sqlquery)
    # drop the table
    crs.execute("DROP TABLE saved_ids")
    connection.commit()
    if vacuum:
        if verbose:
            print >> sys.stdout, "vacuuming"
        crs.execute("VACUUM")


def get_overlap_results(connection, verbose=False):
    """
    Gets the overlap_results, injections, and mapping templates.
    Returns a dectionary of overlap results keyed by the coinc_event_id.
    """
    # get the injections and templates
    # since the overlap f_min is stored in the overlap result,
    # we'll just set the injections' fmin to 0
    injections = waveform_utils.InjectionDict()
    injections.get_injections(connection, 0, calc_f_final=False,
        estimate_dur=False, verbose=verbose)
    # ditto for the templates' fmin and approximant
    templates = waveform_utils.TemplateDict()
    templates.get_templates(connection, 0, '',
            calc_f_final=False, estimate_dur=False, verbose=verbose,
            only_matching=True)
    results = {}
    sqlquery = ''.join(["""
        SELECT
            tmpltMap.event_id, injMap.event_id,
            overlap_results.coinc_event_id, """,
            ', '.join(OverlapResult.params), """
        FROM
            overlap_results
        JOIN
            coinc_event_map AS tmpltMap
        ON
            tmpltMap.coinc_event_id == overlap_results.coinc_event_id AND
            tmpltMap.table_name == "sngl_inspiral"
        JOIN
            coinc_event_map AS injMap
        ON
            injMap.coinc_event_id == overlap_results.coinc_event_id AND
            injMap.table_name == "sim_inspiral"
        """])
    for result in connection.cursor().execute(sqlquery):
        evid, simid, ceid = result[0], result[1], result[2]
        params = [result[ii] for ii in range(3, len(result))]
        thisResult = OverlapResult(templates[evid], injections[simid])
        [setattr(thisResult, param, val) for param, val in \
            zip(OverlapResult.params, params)]
        results[ceid] = thisResult

    return results


def create_all_results_table(connection):
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS all_results (""",
            ', '.join(OverlapResult.__slots__+['tmplt_id', 'simulation_id']),
            """);
        CREATE INDEX IF NOT EXISTS ar_eid_idx ON all_results (tmplt_id);
        CREATE INDEX IF NOT EXISTS ar_sid_idx ON all_results (simulation_id);
        CREATE INDEX IF NOT EXISTS ar_eff_idx ON all_results (effectualness);
        """])
    connection.cursor().executescript(sqlquery)


def create_results_table(connection, table_name='overlap_results',
        set_primary_key=True):
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS
            %s (coinc_event_id%s, """ %(table_name,
                ' PRIMARY KEY' if set_primary_key else ''),
            ', '.join(OverlapResult.params), """);
        CREATE INDEX IF NOT EXISTS
            or_ceid_idx ON %s (coinc_event_id);""" %(table_name)
        ])
    connection.cursor().executescript(sqlquery)


def create_tmplt_weights_table(connection):
    sqlquery = """
        CREATE TABLE IF NOT EXISTS
            tmplt_weights (tmplt_id, weight_function, weight);
        CREATE INDEX IF NOT EXISTS
            tw_tid_idx
        ON
            tmplt_weights (tmplt_id);
        """
    connection.cursor().executescript(sqlquery)


def add_tmplt_weight(connection, tmplt_id, weight_func, weight):
    connection.cursor().execute("""
        INSERT INTO tmplt_weights
            (tmplt_id, weight_function, weight)
        VALUES
            (?, ?, ?)""", (str(tmplt_id), weight_func, weight))


############################
#
#   FIXME: The following should be moved to ligolw_sqlutils
#
def create_cem_table(connection):
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS coinc_event_map (""",
        ', '.join(lsctables.CoincMapTable.validcolumns.keys()), """);
        CREATE INDEX IF NOT EXISTS cem_tn_ei_index ON coinc_event_map (table_name, event_id);
        CREATE INDEX IF NOT EXISTS cem_cei_index ON coinc_event_map (coinc_event_id);
        """])
    connection.cursor().executescript(sqlquery)

def add_coinc_event_map_entry(connection, coinc_event_id, event_id, table_name):
    sqlquery = 'INSERT INTO coinc_event_map (coinc_event_id, table_name, event_id) VALUES (?, ?, ?)'
    connection.cursor().execute(sqlquery, (coinc_event_id, table_name, event_id))


def create_coinc_definer_table(connection):
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS coinc_definer (""",
        ', '.join(lsctables.CoincDefTable.validcolumns.keys()),
        ', %s' %(lsctables.CoincDefTable.constraints), """);
        CREATE INDEX IF NOT EXISTS cd_des_index ON coinc_definer (description);
        """])
    connection.cursor().executescript(sqlquery)

def create_coinc_event_table(connection):
    cursor = connection.cursor()
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS coinc_event (""",
        ', '.join(lsctables.CoincTable.validcolumns.keys()),
        ', %s' %(lsctables.CoincTable.constraints), ')'])
    cursor.execute(sqlquery)
    for idx_name, idx_cols in lsctables.CoincTable.how_to_index.items():
        sqlquery = 'CREATE INDEX IF NOT EXISTS %s ON coinc_event (%s)' %(idx_name, ','.join(idx_cols))
        cursor.execute(sqlquery)
    connection.commit()

sqlutils.create_cem_table = create_cem_table
sqlutils.add_coinc_event_map_entry = add_coinc_event_map_entry
sqlutils.create_coinc_definer_table = create_coinc_definer_table
sqlutils.create_coinc_event_table = create_coinc_event_table
############################

def create_log_table(connection):
    sqlquery = """
        CREATE TABLE IF NOT EXISTS
            joblog (time, simulation_id, inj_num, tmplt_id, tmplt_num,
                finished_filtering, backup_archive, scratch_archive, host,
                username)
        """
    connection.cursor().execute(sqlquery)
    connection.commit()


def get_startup_data(connection):
    sqlquery = """
        SELECT
            simulation_id, inj_num, tmplt_id, tmplt_num, finished_filtering,
            backup_archive, scratch_archive, host, username
        FROM
            joblog
        ORDER BY
            time
        DESC LIMIT 1"""
    startup_dict = {}
    for sim_id, inj_num, tmplt_id, tmplt_num, finished_filtering, backup_archive, scratch_arxiv, host, username in connection.cursor().execute(sqlquery):
        startup_dict['simulation_id'] = sim_id
        startup_dict['inj_num'] = inj_num
        startup_dict['tmplt_id'] = tmplt_id
        startup_dict['tmplt_num'] = tmplt_num
        startup_dict['finished_filtering'] = finished_filtering
        startup_dict['backup_archive'] = backup_archive
        startup_dict['scratch_archive'] = scratch_arxiv
        startup_dict['host'] = host
        startup_dict['username'] = username

    return startup_dict


def del_old_scratch_files(local_drive_path, username, host, scratch_files = [], verbose = False):
    # construct the directory name
    scratch_dir = '%s/%s/%s' %(local_drive_path, host.split('.')[0], username)
    if verbose:
        print >> sys.stdout, "Checking for old scratch files in %s..." % scratch_dir
    for fn in scratch_files:
        if os.path.exists('%s/%s' %(scratch_dir, fn)):
            os.remove('%s/%s' %(scratch_dir, fn))
            if verbose:
                print >> sys.stdout, "Removed %s" % fn


def get_last_backups(connection):
    sqlquery = 'SELECT backup_archive FROM joblog ORDER BY time DESC LIMIT 1'
    results = connection.cursor().execute(sqlquery).fetchone()
    if results:
        last_bkup_arxv = results[0]
    else:
        last_bkup_arxv = None
    return last_bkup_arxv


def lalSeries_to_tuple(lalSeries):
    """
    @lalSeries: either a Time or Frequency Series
    """
    try:
        return ('time', lalSeries.name, lalSeries.epoch.gpsSeconds, lalSeries.epoch.gpsNanoSeconds, lalSeries.f0, lalSeries.deltaT, lalSeries.data.data)
    except AttributeError:
        return ('frequency', lalSeries.name, lalSeries.epoch.gpsSeconds, lalSeries.epoch.gpsNanoSeconds, lalSeries.f0, lalSeries.deltaF, lalSeries.data.data)


def tuple_to_lalSeries(series_tuple):
    series_type, name, sec, nanosec, f0, dx, series_data = series_tuple
    if series_type == 'time':
        series = lal.CreateREAL8TimeSeries(name, lal.LIGOTimeGPS(sec, nanosec), f0, dx, lal.lalSecondUnit, series_data.shape[0])
    elif series_type == 'frequency':
        series = lal.CreateCOMPLEX16FrequencySeries(name, lal.LIGOTimeGPS(sec, nanosec), f0, dx, lal.lalHertzUnit, series_data.shape[0])
    else:
        raise ValueError, 'unrecognized series type %s' % series_type
    series.data.data[:] = series_data

    return series

def write_backup_archive_dict(archive, fd):
    backup_archive = dict([ [key, {}] for key in archive])
    for key in archive:
        for subkey in archive[key]:
            backup_archive[key][subkey] = lalSeries_to_tuple(archive[key][subkey])
    f = os.fdopen(fd, 'w')
    pickle.dump(backup_archive, f)
    f.close()
    del backup_archive


def load_backup_archive_dict(pickle_fn):
    f = open(pickle_fn, 'r')
    backup_archive = pickle.load(f)
    f.close()
    archive = dict([ [key, {}] for key in backup_archive])
    for key in backup_archive:
        for subkey in backup_archive[key]:
            archive[key][subkey] = tuple_to_lalSeries(backup_archive[key][subkey])
    return archive
    

def write_backup_archive(backup_dir, archive):
    if isinstance(archive, h5py.highlevel.File):
        fd, backup_arxv_fn = tempfile.mkstemp(suffix='.hdf5', dir=backup_dir)
        os.close(fd)
        arxiv_fn = archive.filename
        # close the archive to ensure clean copy
        archive.close()
        shutil.copyfile(arxiv_fn, backup_arxv_fn)
    elif isinstance(archive, dict):
        arxiv_fn = None
        fd, backup_arxv_fn = tempfile.mkstemp(suffix = '.pickle',
            dir=backup_dir)
        write_backup_archive_dict(archive, fd)
    else:
        raise ValueError, "unrecognized archive format: got %s, expected dict or h5py file" % type(archive)

    return arxiv_fn, backup_arxv_fn


def checkpoint(connection, backup_dir, archive, time_now, sim_id, inj_num,
        tmplt_id, tmplt_num, finished_filtering, username,
        backup_archive=False):
    # copy the archive over
    if backup_archive:
        arxiv_fn, backup_arxv_fn = write_backup_archive(backup_dir, archive)
        if arxiv_fn is not None:
            archive = h5py.File(arxiv_fn)
    # get the current host
    host = socket.gethostname()
    # get the last backup files so we can delete them
    last_bkup_arxv = get_last_backups(connection)
    if not backup_archive:
        backup_arxv_fn = last_bkup_arxv
        if isinstance(archive, h5py.highlevel.File):
            try:
                arxiv_fn = archive.filename
            except ValueError:
                # this can happen if the archive is already closed
                # so we'll just set the filename to None
                arxiv_fn = None
        else:
            arxiv_fn = None
    if not last_bkup_arxv and not backup_archive:
        backup_arxv_fn = arxiv_fn
    # update the working file
    sqlquery = '''
        INSERT INTO
            joblog (time, simulation_id, inj_num, tmplt_id, tmplt_num,
                finished_filtering, backup_archive, scratch_archive,
                username, host)
        VALUES
            (?,?,?,?,?,?,?,?,?,?)
        '''
    connection.cursor().execute(sqlquery, (time_now, sim_id, inj_num, tmplt_id,
        tmplt_num, finished_filtering, backup_arxv_fn, arxiv_fn, username,
        host))
    connection.commit()
    # delete the last backup files
    if last_bkup_arxv is not None and backup_archive:
        os.remove(last_bkup_arxv)

    return archive

def clean_backup_files(connection):#workingfile, outfile):
    # get the last backup files
    #connection = sqlite3.connect(workingfile)
    last_bkup_arxv = get_last_backups(connection)
    #connection.close()
    # copy to the output file
    #shutil.copyfile(workingfile, outfile)
    # delete the working file
    #os.remove(workingfile)
    # delete the backups
    if last_bkup_arxv is not None and os.path.exists(last_bkup_arxv):
        os.remove(last_bkup_arxv)


def write_result_to_database(connection, result, match_tag, process_id):
    # get the id associated with this match tag
    coinc_def_id = sqlutils.write_newstyle_coinc_def_entry(connection,
        match_tag)
    # add a new entry to the coinc_event_table
    ceid = sqlutils.add_coinc_event_entries(connection, process_id,
        coinc_def_id, None, 1)[0]
    # add an entry in the coinc_event_table for the injection
    sqlutils.add_coinc_event_map_entry(connection, ceid,
        result.injection.simulation_id, 'sim_inspiral')
    # add an entry for the template
    sqlutils.add_coinc_event_map_entry(connection, ceid,
        result.template.tmplt_id, 'sngl_inspiral')
    # add an entry to the overlap table
    result.write_to_database(connection, ceid)
    #connection.commit()


def delete_overlap_entries(connection, save_description, verbose = False):
    if verbose:
        print >> sys.stdout, "Removing entries not matching %s..." % save_description
    sqlquery = "DELETE FROM coinc_definer WHERE description != ?"
    connection.cursor().execute(sqlquery, (save_description,))
    sqlquery = """
        DELETE FROM coinc_event WHERE coinc_def_id NOT IN (SELECT coinc_def_id FROM coinc_definer);
        DELETE FROM coinc_event_map WHERE coinc_event_id NOT IN (SELECT coinc_event_id FROM coinc_event);
        DELETE FROM overlap_results WHERE coinc_event_id NOT IN (SELECT coinc_event_id FROM coinc_event);
        """
    connection.cursor().executescript(sqlquery)
