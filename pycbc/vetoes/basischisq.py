#!/usr/bin/env python

# Copyright (C) 2018 Collin Capano
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

import numpy
from pycbc import types
from pycbc.types import FrequencySeries
from pycbc.waveform import apply_fseries_time_shift
from pycbc import filter
from pycbc import psd as pypsd
from pycbc.vetoes.chisq import SingleDetPowerChisq


def construct_waveform_basis(waveforms, invasd=None, low_frequency_cutoff=None,
                             normalize=True):
    """Constructs a basis out of the given list of waveforms.

    This uses a QR decomposition to construct the basis.

    Parameters
    ----------
    waveforms : list of FrequencySeries
        The waveforms to orthogonalize.
    invasd : FrequencySeries, optional
        Whiten the waveforms using the given inverse ASD before
        orthogonalizing.
    low_frequency_cutoff : float, optional
        Only orthogonalize above the given frequency. If provided, the returned
        basis will be zero at frequencies below this.
    normalize : bool, optional
        Normalize the bases before returning. Default is True.
    """
    df = waveforms[0].delta_f
    if low_frequency_cutoff is not None:
        kmin = int(low_frequency_cutoff / df)
    else:
        kmin = 0
    # exclude any waveforms that are zero in the integration region
    waveforms = [h for h in waveforms if (h.numpy()[kmin:] != 0.).any()]
    nwfs = len(waveforms)
    # get the maximum length of all of the waveforms, to ensure we get a big
    # enough array
    wf_lens = [len(h) for h in waveforms]
    # sainity checks
    if any([h.delta_f != df for h in waveforms]):
        raise ValueError("delta_f mismatch between waveforms")
    if invasd is not None:
        if invasd.delta_f != df:
            raise ValueError("delta_f mismatch with inverse ASD")
        wf_lens = [min(len(invasd), l) for l in wf_lens]
    # only create a basis if we have more than one waveform
    if nwfs > 1:
        # create the waveform array to orthogonalize
        wf_array = numpy.zeros((max(wf_lens), nwfs), dtype=waveforms[0].dtype)
        for ii,(h,kmax) in enumerate(zip(waveforms, wf_lens)):
            wf_array[:kmax,ii] = h.numpy()[:kmax]
            # whiten
            if invasd is not None:
                wf_array[:kmax,ii] *= invasd.numpy()[:kmax]
            wf_array[:kmin,ii] *= 0.
        # get bases as a list of FrequencySeries
        q, _ = numpy.linalg.qr(wf_array)
        bases = [FrequencySeries(ek, delta_f=df, epoch=h.epoch)
                 for ek,h in zip(q.T, waveforms)]
    # otherwise, just return a copy of the input
    else:
        bases = [waveforms[0].copy()]
    if normalize:
        for ek in bases:
            ek /= filter.sigma(ek, low_frequency_cutoff=low_frequency_cutoff)
    return bases


def basis_chisq(whstilde, cplx_snr, trigger_times, sigma, basis, aks,
                low_frequency_cutoff=None, high_frequency_cutoff=None,
                corrmem=None):
    """Calculates the basis chi squared for the given trigger SNRs/times.

    Parameters
    ----------
    whstilde : FrequencySeries
        The **whitened** data.
    cplx_snr : array of complex float
        A list of triggers' complex SNR of the full template with the data.
    trigger_times : array of float
        A list of the trigger times. Must be the same length as ``cplx_snr``.
    sigma : float
        The normalization used in calculation of the complex SNR; i.e., the
        overlap of the full template with itself.
    basis : list of FrequencySeries
        The list of basis waveforms to use. These are assumed to be whitened
        and orthonormal.
    aks : list of complex floats
        The template's coefficients for each basis.
    low_frequency_cutoff : float, optional
        The starting frequency to use for the overlap calculations. If None,
        will start at the beginning of the basis/data.
    high_frequency_cutoff : float, optional
        The ending frequency to use for the overlap calculations. If None,
        will go to the end of the basis.
    corremem : list of FrequencySeries, optional
        List of FrequencySeries to write the correlations between the bases and
        data to. Must be the same length as basis. If None, will create.

    Returns
    -------
    array of float
        The basis chi squared for each trigger time.
    """
    # compute the correlations between the basis and the data
    if corrmem is None:
        corrmem = [types.FrequencySeries(types.zeros(len(ek), dtype=ek.dtype),
                                         delta_f=ek.delta_f)
                   for ek in basis]
    for ek,corr in zip(basis, corrmem):
        if corr.delta_f != whstilde.delta_f:
            raise ValueError("data has a different delta_f than the basis")
        N = (min(len(whstilde), len(ek)) - 1) * 2
        kmin, kmax = filter.get_cutoff_indices(low_frequency_cutoff,
                                               high_frequency_cutoff,
                                               whstilde.delta_f, N)
        filter.correlate(ek[kmin:kmax], whstilde[kmin:kmax], corr[kmin:kmax])
        # set the epoch of the corr mem to be the same as the data, so we get
        # the right time shifts
        corr.kmin = kmin
        corr.kmax = kmax
        corr *= 4. * whstilde.delta_f
    # cycle over the triggers, shifting the correlation vectors appropriately
    chisq = numpy.zeros(len(cplx_snr),
                        dtype=types.real_same_precision_as(whstilde))
    for ii in range(len(cplx_snr)):
        rho = cplx_snr[ii]
        trigtime = trigger_times[ii]
        for corr,ak in zip(corrmem, aks):
            # rotate the correlation to the appropriate time
            # note: we have to shift backward because the template is
            # conjugated in the corr vector
            dt = -float(trigtime - whstilde.start_time)
            shifted = apply_fseries_time_shift(corr, dt, kmin=corr.kmin)
            ek_s = shifted[corr.kmin:corr.kmax].sum()
            chisq[ii] += abs(ek_s - rho*ak/sigma)**2.
    return chisq


def basis_chisq_timeseries(whstilde, cplx_snr, sigma, basis, aks,
                           low_frequency_cutoff=None,
                           high_frequency_cutoff=None):
    """Calculates a time series of basis chi squared.

    Parameters
    ----------
    whstilde : FrequencySeries
        The **whitened** data.
    cplx_snr : complex TimeSeries
        A time series of the full template's complex SNR.
    sigma : float
        The normalization used in calculation of the complex SNR; i.e., the
        overlap of the full template with itself.
    basis : list of FrequencySeries
        The list of basis waveforms to use. These are assumed to be whitened
        and orthonormal.
    aks : list of complex floats
        The template's coefficients for each basis.
    low_frequency_cutoff : float, optional
        The starting frequency to use for the overlap calculations. If None,
        will start at the beginning of the basis/data.
    high_frequency_cutoff : float, optional
        The ending frequency to use for the overlap calculations. If None,
        will go to the end of the basis.

    Returns
    -------
    TimeSeries
        A time series of the basis chi squared.
    """
    chisq = None
    for ek,ak in zip(basis, aks):
        akhat = ak / sigma
        # get the overlap of the basis with the data
        ek_s = filter.matched_filter(ek, whstilde,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff)
        diff = abs(ek_s - cplx_snr*akhat)**2.
        if chisq is None:
            chisq = diff
        else:
            chisq += diff
    return chisq


class SingleDetBasisChisq(object):
    """Class that handles precomutation and memory management for running
    a basis chisq.
    """
    _basis_cache = 'cached_basis'

    def __init__(self, snr_threshold=None):
        self.basis = None
        self.coeffs = None
        self.ndim = None
        self.dof = None
        self.snr_threshold = snr_threshold
        self.do = snr_threshold is not None
        self._invasd = {}
        self._corrmem = {}

    def get_basis(self, template, psd=None):
        """Retrieve basis and coefficients from the given template.
        """
        # we'll use the ID of the PSD to cache things
        if psd is None:
            key = None
        else:
            key = id(psd)
        return getattr(template, self._basis_cache)[key]

    def update_basis(self, template, psd=None):
        """Updates the current basis/coefficients using the given template.
        """
        basis, coeffs = self.get_basis(template, psd=psd)
        self.basis = basis
        self.coeffs = coeffs
        self.ndim = len(basis)
        self.dof = 2*self.ndim - 2
        return basis, coeffs

    def return_timeseries(self, ntrigs, dlen):
        """Determines if chi squared time series should be calculated.

        Parameters
        ----------
        ntrigs : int
            The number of triggers that an SNR is needed for.
        dlen : int
            The length of the data (in the frequency domain).

        Returns
        -------
        bool
            True if a full time series should be calculated; False if it's
            better to do a series of point estimates.
        """
        # if an FFT is done, then we need to do ~N log N operations for each
        # basis
        n_fft = dlen * numpy.log2(dlen) * self.ndim
        # if a point estimates are done then, for each trigger, we need to do
        # ~N*ndim operations to get the overlap of the basis with the data
        # + ~N operations to rotate the data in time for each trigger
        n_point = ntrigs*(dlen * self.ndim + dlen)

    def invasd(self, psd):
        """Gets the inverse ASD from the given the PSD.
        """
        if psd is None:
            return None
        key = id(psd)
        try:
            invasd = self._invasd[key]
        except KeyError:
            # means PSD has changed, clear the dict and re-calculate
            self._invasd.clear()
            invasd = pypsd.invert_psd(psd)**0.5
            self._invasd[key] = invasd
        return invasd

    def corrmem(self, stilde):
        """Creates/retrieves vectors for writing correlations.

        Vectors are cached based on the length of stilde.
        """
        N = len(stilde)
        try:
            corrmem = self._corrmem[N]
        except KeyError:
            # clear the dictionary for a new list
            self._corrmem.clear()
            corrmem = []
        # add/subtract any more needed vectors
        ncorrs = len(corrmem)
        if ncorrs < self.ndim:
            corrmem += [types.FrequencySeries(
                            types.zeros(N, dtype=stilde.dtype),
                            delta_f=stilde.delta_f)
                for _ in range(self.ndim-ncorrs)]
            self._corrmem[N] = corrmem
        elif self.ndim < ncorrs:
            corrmem = corrmem[:self.ndim]
            self._corrmem[N] = corrmem
        return corrmem

    def values(self, template, stilde, psd, trigger_snrs, trigger_idx,
               data_whitening=None):
        """Calculates the HM chisq for the given points.
        """
        if not self.do:
            return None, None
        if self.snr_threshold is not None:
            keep = abs(trigger_snrs) >= self.snr_threshold
            trigger_snrs = trigger_snrs[keep]
            trigger_idx = trigger_idx[keep]

        basis, coeffs = self.update_basis(template, psd)
        if data_whitening is None:
            # need to whiten the data
            whstilde = stilde * self.invasd(psd)
        elif data_whitening == 'whitened':
            # don't need to do anything
            whstilde = stilde
        elif data_whitening == 'overwhitened':
            whstilde = stilde * psd**0.5
        else:
            raise ValueError("unrecognized data_whitening argument {}".format(
                             data_whitening))

        trigger_times = trigger_idx * stilde.delta_t + stilde.start_time 
        sigma = template.sigmasq(psd)**0.5

        chisq = basis_chisq(whstilde, trigger_snrs, trigger_times, sigma,
                            basis, coeffs,
                            low_frequency_cutoff=template.f_lower,
                            corrmem=self.corrmem(stilde))
        # add back 1s for triggers that were below threshold
        if self.snr_threshold is not None and len(keep) != len(chisq):
            out = numpy.ones(len(keep), dtype=chisq.dtype)
            out[keep] = chisq
            chisq = out
        return chisq, numpy.repeat(self.dof, len(chisq))


class SingleDetHMChisq(SingleDetBasisChisq):
    """Class that handles precomutation and memory management for running
    the higher-mode chisq.
    """
    _basis_cache = 'cached_hmbasis'
    returns = {'hm_chisq': numpy.float32, 'hm_chisq_dof': int}

    def construct_waveform_basis(self, template, psd):
        """Constructs a basis from the template's modes."""
        invasd = self.invasd(psd)
        waveforms = template.modes.values()
        return construct_waveform_basis(waveforms, invasd, template.f_lower)

    def update_basis(self, template, psd=None):
        """Calculates/retrieves basis for the given template.
        """
        if psd is not None:
            key = id(psd)
        else:
            key = None
        try:
            # first try to return a cached basis
            return super(SingleDetHMChisq, self).update_basis(template, psd)
        except AttributeError:
            # template does not have the attribute: means that a basis hasn't
            # been calculated, so create the attribute and calculate
            setattr(template, self._basis_cache, {})
        except KeyError:
            # means the PSD has changed; delete the old basis and create a
            # new one
            getattr(template, self._basis_cache).clear()
        # Generate the mode basis
        basis = self.construct_waveform_basis(template, psd)
        # calculate the expected values
        coeffs = [numpy.complex(filter.overlap_cplx(ek, template,
                                low_frequency_cutoff=template.f_lower,
                                normalized=False))
                  for ek in basis]

        getattr(template, self._basis_cache)[key] = (basis, coeffs)
        return super(SingleDetHMChisq, self).update_basis(template, psd=psd)

    @staticmethod
    def insert_option_group(parser):
        """Adds the options needed to set up the HM Chisq."""
        group = parser.add_argument_group("HM Chisq")
        group.add_argument("--hmchisq-snr-threshold", type=float, default=None,
                            help="SNR threshold to use for applying the HM "
                                 "chisq. If not provided, the HM Chisq test "
                                 "will not be performed.")

    @classmethod
    def from_cli(cls, opts):
        """Initializes the HM Chisq using the given options."""
        return cls(snr_threshold=opts.hmchisq_snr_threshold)


class SingleDetCombinedChisq(SingleDetHMChisq):
    """Class that handles precomutation and memory management for running
    the higher-mode chisq combined with the power chisq.
    """
    _basis_cache = 'cached_cbasis'
    returns = {'combined_chisq': numpy.float32, 'combined_chisq_dof': int}

    def __init__(self, power_chisq_bins=0, snr_threshold=None):
        super(SingleDetCombinedChisq, self).__init__(
            snr_threshold=snr_threshold)
        # use an instance of SingleDetPowerChisq in order to get cache power
        # chisq bins
        self._powerchisq = SingleDetPowerChisq(power_chisq_bins,
            snr_threshold)

    def construct_waveform_basis(self, template, psd):
        """Constructs a basis from the power chisq bins and the template's
        modes.
        """
        # get the power chisq bins
        pchisq_bins = self._powerchisq.cached_chisq_bins(template, psd)
        # construct the waveforms
        waveforms = []
        for ii,kstart in enumerate(pchisq_bins[:-1]):
            kend = pchisq_bins[ii+1]
            h = template.copy()
            h[:kstart] *= 0.
            h[kend:] *= 0.
            waveforms.append(h)
        # get the modes, excluding any that are 0 in the integration region
        kmin = pchisq_bins[0]
        modes = [h for h in template.modes.values()
                 if (h.numpy()[kmin:] != 0.).any()]
        # only use modes if we have more than one
        if len(modes) > 1:
            waveforms += modes
        invasd = self.invasd(psd)
        return construct_waveform_basis(waveforms, invasd, template.f_lower)

    @staticmethod
    def insert_option_group(parser):
        """Adds the options needed to set up the Combined Chisq."""
        group = parser.add_argument_group("Combined Chisq")
        group.add_argument("--combined-chisq-snr-threshold", type=float,
                            default=None,
                            help="SNR threshold to use for applying the "
                                 "combined chisq. If not provided, the "
                                 "combined chisq test will not be performed.")

    @classmethod
    def from_cli(cls, opts):
        """Initializes the Combined Chisq using the given options."""
        return cls(power_chisq_bins=opts.chisq_bins,
            snr_threshold=opts.combined_chisq_snr_threshold)

