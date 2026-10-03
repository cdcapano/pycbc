# Copyright (C) 2020  Collin Capano and Shilpa Kastha
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

"""This module provides model classes that assume the noise is Gaussian and
introduces a gate to remove given times from the data, using the inpainting
method to fill the removed part such that it does not enter the likelihood.
"""

from abc import abstractmethod
import logging
import numpy
import scipy
import shlex
from scipy import special
import warnings

from pycbc.types import FrequencySeries
from pycbc.detector import Detector
from pycbc.pnutils import hybrid_meco_frequency
from pycbc import types
from pycbc.waveform.utils import time_from_frequencyseries
from pycbc.waveform import generator, FailedWaveformError
from pycbc.filter import highpass_fd, highpass_response
from pycbc.strain.gate import (invert_covariance, toeplitz_inverse,
                               toeplitz_inverses,
                               batch_gate_and_paint_fd_array)
from .gaussian_noise import (BaseGaussianNoise, create_waveform_generator,
                             catch_waveform_error)
from .base import apply_transforms_to_samples
from .base_data import BaseDataModel
from .data_utils import fd_data_from_strain_dict


def _toeplitz_inverses_task(args):
    """Calls toeplitz_inverses with the given (detector, invpsd, sizes)
    tuple, returning the detector and the result. Used by
    BaseGatedGaussian.fill_gate_cache with a pool."""
    det, invpsd, sizes = args
    return det, toeplitz_inverses(invpsd, sizes)


# The gate start times and widths in each detector are rounded to a multiple
# of this (2^-20 s, about a microsecond). The gating functions take the
# center and half-width of the gate, from which the start and end of the
# gate are found. For times on this grid, that is exact in double precision
# (for GPS times before 2^32 s); otherwise, the edges are only recovered to
# within the resolution of a double at GPS times (~2e-7 s), which can change
# the sample an edge lands in. That matters when a time is shared by the
# gates of different models, such as the submodels of a hierarchical model
# that split the data at a common time.
GATE_TIME_RESOLUTION = 2.**-20


def _round_gate_time(time):
    """Rounds the given time(s) to a multiple of ``GATE_TIME_RESOLUTION``.
    """
    return numpy.round(time / GATE_TIME_RESOLUTION) * GATE_TIME_RESOLUTION


class BaseGatedGaussian(BaseGaussianNoise):
    r"""Base model for gated gaussian.

    Provides additional routines for applying a time-domain gate to data.
    See :py:class:`GatedGaussianNoise` for more details.
    """
    # Whether the model accounts for the detector response changing over the
    # duration of the signal (e.g., due to the motion of the detectors). If
    # False (the default), the gate times in each detector are obtained by
    # shifting the gate by the arrival time delay at the coalescence time,
    # which is the same shift that is used to project the template into the
    # detector, so the gate width is the same in every detector. If True, the
    # start and end of the gate are each shifted by the delay at that time.
    _time_varying_response = False

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, highpass_waveforms=False, **kwargs):
        # we'll want the time-domain data, so store that
        self._td_data = {}
        # cache the overwhitened data
        self._overwhitened_data = {}
        # cache the current gated data
        self._gated_data = {}
        # cache terms related to normalization and gating
        self._invasds = {}
        self._Rss = {}
        self._lognorm = {}
        self._gatetimes = {}
        self._gate_detectors = {}
        self._det_lognls = {}
        # cache condition number calculations
        self.check_condition_number = bool(kwargs.get('check-condition-number',
                                                      False))
        self._cond = {}
        # cache inpainting options
        paint_method = kwargs.get('paint_method')
        if paint_method is None:
            paint_method = kwargs.get('paint-method', 'toeplitz')
        self.paint_method = paint_method
        self._cov_matrices = {}
        # cache samples and linear regression for determinant extrapolation
        self._cov_samples = {}
        self._cov_regressions = {}
        # highpass waveforms with the given frequency
        self.highpass_waveforms = highpass_waveforms
        if self.highpass_waveforms:
            logging.info("Will highpass waveforms at %f Hz",
                         highpass_waveforms)
        # set up the boiler-plate attributes
        super().__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        # the number of prior samples to use to fill the cache of in-painting
        # inverses in the warmup
        gate_cache_samples = kwargs.get('gate_cache_samples')
        if gate_cache_samples is None:
            gate_cache_samples = kwargs.get('gate-cache-samples', 10000)
        self.gate_cache_samples = int(gate_cache_samples)

    @classmethod
    def from_config(cls, cp, data_section='data', data=None, psds=None,
                    **kwargs):
        """Adds additional keyword arguments based on config file.

        Additional keyword arguments are:

           * ``highpass_waveforms`` : waveforms will be highpassed.

        Also forces ``invpsd-trunc-low-freq-fill-value`` to ``fmin`` if not
        specified.
        """
        if cp.has_option(data_section, 'strain-high-pass') and \
            'highpass_waveforms' not in kwargs:
            kwargs['highpass_waveforms'] = float(cp.get(data_section,
                                                        'strain-high-pass'))
        if not cp.has_option(data_section, 'invpsd-trunc-low-freq-fill-value'):
            cp.set(data_section, 'invpsd-trunc-low-freq-fill-value', 'fmin')
        return super().from_config(cp, data_section=data_section,
                                   data=data, psds=psds,
                                   **kwargs)

    @BaseDataModel.data.setter
    def data(self, data):
        """Store a copy of the FD and TD data."""
        BaseDataModel.data.fset(self, data)
        # store the td version
        self._td_data = {det: d.to_timeseries() for det, d in data.items()}

    @property
    def td_data(self):
        """The data in the time domain."""
        return self._td_data

    @BaseGaussianNoise.psds.setter
    def psds(self, psds):
        """Sets the psds, and calculates the weight and norm from them.
        The data and the low and high frequency cutoffs must be set first.
        """
        # check that the data has been set
        if self._data is None:
            raise ValueError("No data set")
        if self._f_lower is None:
            raise ValueError("low frequency cutoff not set")
        if self._f_upper is None:
            raise ValueError("high frequency cutoff not set")
        # make sure the relevant caches are cleared
        self._psds.clear()
        self._invpsds.clear()
        self._invasds.clear()
        self._gated_data.clear()
        self._cov_matrices.clear()
        self._cov_samples.clear()
        self._cov_regressions.clear()
        # store the psds
        for det, d in self._data.items():
            if psds is None:
                # No psd means assume white PSD
                p = FrequencySeries(numpy.ones(int(self._N/2+1)),
                                    delta_f=d.delta_f)
            else:
                # copy for storage
                p = psds[det].copy()
            self._psds[det] = p
            # we'll store the weight to apply to the inner product
            invp = 1./p
            self._invpsds[det] = invp
            self._invasds[det] = invp**0.5
            # store the autocorrelation function and covariance matrix for
            # each detector
            Rss = p.astype(types.complex_same_precision_as(p)).to_timeseries()
            self._Rss[det] = Rss
            # calculate and store the linear regressions to extrapolate
            # determinant values
            if self.normalize:
                self._set_covfit(det)
        self._overwhitened_data = self.whiten(self.data, 2, inplace=False)

    def _set_covfit(self, det):
        """Sets the fit function for estimating the covariance determinant.

        This must be called after the PSDs have been set, otherwise a
        ValueError will be raised.
        """
        try:
            p = self.psds[det]
        except KeyError:
            raise ValueError("No psd set for detector %s" % det)
        # the covariance matrix is the Toeplitz matrix with first column
        # Rss/2; only the (small) truncated matrices needed for the fit are
        # constructed from it
        Rss = self._Rss[det]
        samples, fit = self.logdet_fit(Rss.numpy()/2, p)
        self._cov_samples[det] = samples
        self._cov_regressions[det] = fit
        return

    def logdet_fit(self, cov_col, p):
        """Construct a linear regression from a sample of truncated covariance
        matrices.

        Parameters
        ----------
        cov_col : array
            The first column of the (Toeplitz) covariance matrix. The
            truncated matrices are constructed directly from it, so that the
            full covariance matrix, which can be very large, is never
            constructed.
        p : FrequencySeries
            The PSD.

        Returns the sample points used for linear fit generation as well as the
        linear fit parameters.
        """
        # initialize lists for matrix sizes and determinants
        sample_sizes = []
        sample_dets = []
        # set sizes of sample matrices; ensure exact calculations are only on
        # small matrices
        s = len(cov_col)
        max_size = 8192
        if s > max_size:
            sample_sizes = [s, max_size, max_size//2, max_size//4]
        else:
            sample_sizes = [s, s//2, s//4, s//8]
        for i in sample_sizes:
            # calculate logdet of the full matrix using circulant eigenvalue
            # approximation
            if i == s:
                ld = 2*numpy.log(p/(2*p.delta_t)).sum()
                sample_dets.append(ld)
            # generate three more sample matrices using exact calculations
            else:
                gate_size = s - i
                start = (s - gate_size)//2
                end = start + gate_size
                # the covariance matrix with rows and columns start:end
                # removed; element (j, k) of the full matrix is
                # cov_col[j-k] for j >= k, and the conjugate of cov_col[k-j]
                # otherwise (as for scipy.linalg.toeplitz)
                idx = numpy.concatenate([numpy.arange(start),
                                         numpy.arange(end, s)])
                lag = idx[:, None] - idx[None, :]
                tc = cov_col[abs(lag)]
                if numpy.iscomplexobj(tc):
                    tc = numpy.where(lag >= 0, tc, tc.conj())
                del lag
                ld = numpy.linalg.slogdet(tc)[1]
                sample_dets.append(ld)
        # generate a linear regression using the four points (size, logdet)
        x = numpy.vstack([sample_sizes, numpy.ones(len(sample_sizes))]).T
        m, b = numpy.linalg.lstsq(x, sample_dets, rcond=None)[0]
        return (sample_sizes, sample_dets), (m, b)

    @BaseGaussianNoise.normalize.setter
    def normalize(self, normalize):
        """Clears the current stats if the normalization state is changed.

        If normalize is set to True, the fit to the covariance determinant
        will be calculated if it hasn't yet and PSDs are set.
        """
        # call the parent setter to clear the current stats and set normalize
        BaseGaussianNoise.normalize.fset(self, normalize)
        # now set the covariance determinant fit if needed
        if normalize:
            for det in self._psds:
                if det not in self._cov_regressions:
                    # set the covariance determinant fit
                    self._set_covfit(det)

    def gate_indices(self, det):
        """Calculate the indices corresponding to start and end of gate.
        """
        # get time series start and delta_t
        ts = self.td_data[det]
        # get gate start and length from get_gate_times
        gate_start, gate_length = self.get_gate_times()[det]
        # the gate code takes central time and width
        window = gate_length / 2
        gt = gate_start + window
        lindex, rindex = ts.get_gate_indices(gt, window)
        return lindex, rindex

    def det_lognorm(self, det, start_index=None, end_index=None):
        """Calculate the normalization term from the truncated covariance
        matrix.

        Determinant is estimated using a linear fit to logdet vs truncated
        matrix size.
        """
        if not self.normalize:
            return 0
        try:
            # check if the key already exists; if so, return its value
            lognorm = self._lognorm[(det, start_index, end_index)]
        except KeyError:
            # get the size of the matrix
            n = len(self._Rss[det])
            trunc_size = n - (end_index - start_index)
            # call the linear regression
            m, b = self._cov_regressions[det]
            # extrapolate from linear fit
            ld = m*trunc_size + b
            # full normalization term:
            lognorm = -0.5*(numpy.log(2*numpy.pi)*trunc_size + ld)
            # cache the result
            self._lognorm[(det, start_index, end_index)] = lognorm
        return lognorm

    def _nowaveform_handler(self):
        """Convenience function to set logl values if no waveform generated.
        """
        return -numpy.inf

    def _loglr(self):
        r"""Computes the log likelihood ratio.
        Returns
        -------
        float
            The value of the log likelihood ratio evaluated at the given point.
        """
        return self._loglikelihood() - self._lognl()

    def whiten(self, data, whiten, inplace=False):
        """Whitens the given data.

        Parameters
        ----------
        data : dict
            Dictionary of detector names -> FrequencySeries.
        whiten : {0, 1, 2}
            Integer indicating what level of whitening to apply. Levels are:
            0: no whitening; 1: whiten; 2: overwhiten.
        inplace : bool, optional
            If True, modify the data in place. Otherwise, a copy will be
            created for whitening.


        Returns
        -------
        dict :
            Dictionary of FrequencySeries after the requested whitening has
            been applied.
        """
        if not inplace:
            data = {det: d.copy() for det, d in data.items()}
        if whiten:
            for det, dtilde in data.items():
                invpsd = self._invpsds[det]
                if whiten == 1:
                    dtilde *= invpsd**0.5
                elif whiten == 2:
                    dtilde *= invpsd
                else:
                    raise ValueError("whiten must be either 0, 1, or 2")
        return data

    def invert_covariance(self, det):
        """Get the uninverted covariance matrix for the model's inverse PSDs.
        Once the inverse matrix is calculated for a given gate time in this
        detector, store to cache; future calls of this function will pull from
        that cache instead. If the paint method is ``'gs'``, a
        :py:class:`pycbc.strain.gate.ToeplitzInverse` is returned instead of
        the explicit inverse.
        """
        # don't bother with covariance matrix if we're using toeplitz solver
        if self.paint_method == 'toeplitz':
            return None
        # check if there are cache results for this gate length
        lindex, rindex = self.gate_indices(det)
        try:
            cov_matrices = self._cov_matrices[int(rindex-lindex)]
        except KeyError:
            cov_matrices = {}
        # check if this det has a precalculated matrix for this gate length
        try:
            invmat = cov_matrices[det]
        except KeyError:
            invpsd = self._invpsds[det]
            if self.paint_method == 'gs':
                # set up the Gohberg-Semencul representation of the inverse
                invmat = toeplitz_inverse(invpsd, lindex, rindex)
            else:
                # construct and invert covariance matrix
                invmat = invert_covariance(invpsd, lindex, rindex)
            cov_matrices[det] = invmat
            # cache results
            self._cov_matrices[int(rindex-lindex)] = cov_matrices
        return invmat

    @abstractmethod
    def get_waveforms(self):
        """The waveforms generated using the current parameters.

        If the waveforms haven't been generated yet, they will be generated,
        resized to the same length as the data, and cached. If the
        ``highpass_waveforms`` attribute is set, a highpass filter will
        also be applied to the waveforms.

        Returns
        -------
        dict :
            Dictionary of detector names -> waveforms
        """
        pass

    @abstractmethod
    def get_gated_waveforms(self):
        """Generates and gates waveforms using the current parameters.

        Returns
        -------
        dict :
            Dictionary of detector names -> FrequencySeries.
        """
        pass

    def get_data(self):
        """Return a copy of the data.

        Returns
        -------
        dict :
            Dictionary of detector names -> FrequencySeries.
        """
        return {det: d.copy() for det, d in self.data.items()}

    def get_gated_data(self):
        """Return a copy of the gated data.

        The gated data will be cached for faster retrieval.

        Returns
        -------
        dict :
            Dictionary of detector names -> FrequencySeries.
        """
        gate_times = self.get_gate_times()
        out = {}
        for det, d in self.td_data.items():
            # make sure the cache at least has the detectors in it
            try:
                cache = self._gated_data[det]
            except KeyError:
                cache = self._gated_data[det] = {}
            invpsd = self._invpsds[det]
            gatestartdelay, dgatedelay = gate_times[det]
            try:
                dtilde = cache[gatestartdelay, dgatedelay]
            except KeyError:
                # doesn't exist yet, or the gate times changed
                cache.clear()
                invmat = self.invert_covariance(det)
                d = d.gate(gatestartdelay + dgatedelay/2,
                           window=dgatedelay/2, copy=True,
                           invpsd=invpsd, method='paint',
                           paint_method=self.paint_method,
                           paint_invmat=invmat)
                dtilde = d.to_frequencyseries()
                # save for next time
                cache[gatestartdelay, dgatedelay] = dtilde
            out[det] = dtilde
        return out

    def fill_gate_cache(self, samples, pool=None):
        """Adds the in-painting inverses for the gate sizes of the given
        samples to the cache.

        The gate start and end times of every sample are converted to each
        detector's frame, and the gate size (in samples) is found in the
        same way as is done when the likelihood is evaluated. The
        :py:class:`pycbc.strain.gate.ToeplitzInverse` for all of the sizes
        that are not already cached are then found with a single pass of the
        Levinson-Durbin recursion in each detector (see
        :py:func:`pycbc.strain.gate.toeplitz_inverses`). This is much faster
        than setting up each size separately as it is encountered. Any sizes
        that are missed will still be set up and cached as they are
        encountered.

        This is only supported for the ``'gs'`` paint method.

        Parameters
        ----------
        samples : dict
            Dictionary of parameter names -> arrays. Must contain the gate
            parameters (i.e., the waveform transforms must already have been
            applied): ``t_gate_start``, ``t_gate_end``, ``ra``, and ``dec``,
            and ``tc`` if ``_time_varying_response`` is False, except for the
            detectors whose gate is given by per-detector parameters (see
            :py:meth:`get_gate_times`). May contain ``tc_ref_frame``.
            Scalars are broadcast.
        pool : pool object, optional
            Pool used to set up the inverses of different detectors in
            parallel. If None, they are set up in serial.
        """
        if self.paint_method != 'gs':
            raise ValueError("filling the gate cache is only supported for "
                             "the gs paint method")
        refframe = samples.get('tc_ref_frame',
                               self.static_params.get('tc_ref_frame',
                                                      'geocentric'))
        if not isinstance(refframe, str):
            # a single reference frame is assumed
            refframe = numpy.unique(refframe)
            if len(refframe) != 1:
                raise ValueError("all samples must have the same "
                                 "tc_ref_frame")
            refframe = str(refframe[0])
        detgates = self._detector_gate_times(samples)
        params = {'t_gate_start': samples.get('t_gate_start'),
                  't_gate_end': samples.get('t_gate_end')}
        if self._needs_sky_gate(detgates):
            params['ra'] = samples['ra']
            params['dec'] = samples['dec']
            if not self._time_varying_response:
                params['tc'] = samples['tc']
        # broadcast everything to the same shape
        values = [x for x in params.values() if x is not None] + \
            [x for edges in detgates.values() for x in edges if x is not None]
        shape = numpy.broadcast_shapes(*[numpy.shape(x) for x in values])

        def bcast(x):
            if x is None:
                return None
            return numpy.broadcast_to(numpy.asarray(x, dtype=float), shape)

        detgates = {det: (bcast(start), bcast(end))
                    for det, (start, end) in detgates.items()}
        gatetimes = self._get_gate_times(
            bcast(params['t_gate_start']), bcast(params['t_gate_end']),
            bcast(params.get('ra')), bcast(params.get('dec')),
            refframe=refframe, tc=bcast(params.get('tc')),
            detgates=detgates)
        tasks = []
        for det, (start, width) in gatetimes.items():
            # same as gate_indices, i.e., TimeSeries.get_gate_indices
            ts = self.td_data[det]
            st = float(ts.start_time)
            dt = float(ts.delta_t)
            window = width / 2
            gt = start + window
            lindex = numpy.trunc((gt - window - st) / dt).astype(int)
            rindex = numpy.trunc((gt + window - st) / dt).astype(int)
            lindex = numpy.clip(lindex, 0, None)
            rindex = numpy.clip(rindex, None, len(ts))
            sizes = set(int(k) for k in numpy.unique(rindex - lindex)
                        if k > 0)
            new = [k for k in sizes if det not in self._cov_matrices.get(k, {})]
            if not new:
                continue
            logging.info("Setting up the in-painting inverses for %i gate "
                         "sizes (%i - %i samples) in %s", len(new), min(new),
                         max(new), det)
            tasks.append((det, self._invpsds[det], new))
        if pool is None:
            results = map(_toeplitz_inverses_task, tasks)
        else:
            results = pool.map(_toeplitz_inverses_task, tasks)
        for det, tinvs in results:
            for k, tinv in tinvs.items():
                self._cov_matrices.setdefault(k, {})[det] = tinv

    def _warmup(self, pool):
        """Fills the cache of in-painting inverses using samples drawn from
        the prior.

        This is only done for the ``'gs'`` paint method (the other methods
        need too much memory to cache many gate sizes), and if the
        ``gate_cache_samples`` attribute (set by the ``gate-cache-samples``
        model option) is greater than zero. The given number of samples are
        drawn with :py:meth:`prior_rvs`, the waveform transforms are applied
        to them, and the result is passed to :py:meth:`fill_gate_cache`.
        Nothing is done if the model has no prior or no variable parameters,
        or if the gate times cannot be determined from the prior (e.g., if
        ``gatefunc = hmeco``).

        Parameters
        ----------
        pool : pool object
            Pool used to parallelize the set up over detectors.
        """
        nsamples = self.gate_cache_samples
        if self.paint_method != 'gs' or nsamples <= 0:
            return
        if 'gatefunc' in self.variable_params or \
                self.static_params.get('gatefunc') is not None:
            logging.info("The gate times depend on the waveform, so not "
                         "filling the gate cache")
            return
        if not self.variable_params:
            logging.debug("No variable parameters, so not filling the gate "
                          "cache")
            return
        try:
            draws = self.prior_rvs(size=nsamples)
        except AttributeError:
            logging.debug("No prior, so not filling the gate cache")
            return
        logging.info("Filling the gate cache using %i samples from the "
                     "prior", nsamples)
        if self.sampling_transforms is not None:
            draws = self.sampling_transforms.apply(draws, inverse=True)
        samples = {p: draws[p] for p in draws.fieldnames}
        for p, val in self.static_params.items():
            samples.setdefault(p, val)
        try:
            if self.waveform_transforms is not None:
                samples = apply_transforms_to_samples(
                    samples, self.waveform_transforms)
            self.fill_gate_cache(samples, pool=pool)
        except (KeyError, ValueError, TypeError) as err:
            logging.info("The gate times could not be determined from the "
                         "prior (%s), so not filling the gate cache", err)

    def get_gate_times(self):
        """Gets the time to apply a gate based on the current sky position.

        If the parameter ``gatefunc`` is set to ``'hmeco'``, the gate times
        will be calculated based on the hybrid MECO of the given set of
        parameters; see ``get_gate_times_hmeco`` for details. Otherwise, the
        gate times will just be retrieved from the ``t_gate_start`` and
        ``t_gate_end`` parameters, which are shifted into each detector's
        frame using the sky location (see :py:meth:`_get_gate_times`).

        The start (end) of the gate in a detector may instead be given by a
        ``t_gate_start_{det}`` (``t_gate_end_{det}``) parameter, where
        ``{det}`` is the lower-case name of the detector; e.g.,
        ``t_gate_start_h1``. These are times in the detector's frame, and
        take precedence over ``t_gate_start`` (``t_gate_end``) in that
        detector. If only one edge of a detector's gate is given this way,
        the other edge is shifted into the detector's frame as usual. Two
        models that share that edge (e.g., the submodels of a hierarchical
        model that split the data at a common time, with the outer edges of
        their gates fixed in each detector) place it at exactly the same time
        in each detector, provided they are given the same shared edge,
        ``ra``, ``dec``, and ``tc``.

        If the user flagged ``check_condition_number``, also checks if
        inpainting with the calculated gate times will be numerically
        stable. See ``self.condition_number()`` for more info.

        Returns
        -------
        dict :
            Dictionary of detector names -> (gate start, gate width)
        """
        params = self.current_params
        try:
            gatefunc = self.current_params['gatefunc']
        except KeyError:
            gatefunc = None
        if gatefunc == 'hmeco':
            return self.get_gate_times_hmeco()
        detgates = self._detector_gate_times(params)
        gatestart = params.get('t_gate_start')
        gateend = params.get('t_gate_end')
        ra = dec = tc = None
        if self._needs_sky_gate(detgates):
            # we'll need the sky location for determining time shifts
            ra = params['ra']
            dec = params['dec']
            # and the coalescence time, unless the response is time varying
            if not self._time_varying_response:
                tc = params['tc']
        # try to get from cache
        key = (gatestart, gateend, ra, dec, tc,
               tuple(sorted(detgates.items())))
        try:
            gatetimes = self._gatetimes[key]
        except KeyError:
            # doesn't exist, or parameters have changed, recalculate
            self._gatetimes.clear()
            gatetimes = self._get_gate_times(gatestart, gateend, ra, dec,
                                             tc=tc, detgates=detgates)
            self._gatetimes[key] = gatetimes
        # check if the inpainting is numerically stable
        if self.check_condition_number:
            for det, d in self.td_data.items():
                lindex, rindex = d.get_gate_indices(gatetimes[det][0],
                                                    gatetimes[det][1]/2.)
                self.condition_number(det, lindex, rindex)
        return gatetimes

    def _detector_gate_times(self, params):
        """Gets the gate times that are given for each detector by the
        ``t_gate_start_{det}`` and ``t_gate_end_{det}`` parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names -> values.

        Returns
        -------
        dict :
            Dictionary of detector names -> (gate start, gate end), for the
            detectors that have at least one edge of their gate given. An
            edge that is not given is None.
        """
        detgates = {}
        for det in self._invpsds:
            start = params.get(f't_gate_start_{det.lower()}')
            end = params.get(f't_gate_end_{det.lower()}')
            if start is not None or end is not None:
                detgates[det] = (start, end)
        return detgates

    def _needs_sky_gate(self, detgates):
        """Whether any detector has an edge of its gate that is not given by
        the per-detector gate times ``detgates``, and so must be shifted into
        the detector's frame using the sky location."""
        return any(edge is None for det in self._invpsds
                   for edge in detgates.get(det, (None, None)))

    def _get_gate_times(self, gatestart, gateend, ra, dec, refframe=None,
                        tc=None, detgates=None):
        """Calculates the gate times in each detector.

        The times may also be arrays, in which case arrays are returned.

        If the ``_time_varying_response`` attribute is False (the default),
        the gate is shifted into each detector's frame by the arrival time
        delay at the coalescence time ``tc``, so that the gate width is the
        same in all detectors. Otherwise, the start and end of the gate are
        each shifted by the delay at that time.

        Edges given in ``detgates`` are used as they are, and any edge that
        is not given is shifted as described above.

        The gate start and width are rounded to a multiple of
        ``GATE_TIME_RESOLUTION``.

        Parameters
        ----------
        gatestart : float
            Start time of the gate. Only needed for detectors without a
            start time in ``detgates``.
        gateend : float
            End time of the gate. Only needed for detectors without an end
            time in ``detgates``.
        ra : float
            Right ascension of the signal. Not needed if all detectors have
            both edges in ``detgates``.
        dec : float
            Declination of the signal. Not needed if all detectors have both
            edges in ``detgates``.
        refframe : str, optional
            The frame the gate times are defined in. If None, will use the
            ``tc_ref_frame`` in the current parameters, or ``'geocentric'``
            if there is none.
        tc : float, optional
            The coalescence time (in the reference frame) at which to compute
            the arrival time delay. Only used if ``_time_varying_response``
            is False, for edges that are not in ``detgates``. If None, will
            use the ``tc`` in the current parameters.
        detgates : dict, optional
            Dictionary of detector names -> (gate start, gate end) giving
            gate times in the detectors' frames, as returned by
            :py:meth:`_detector_gate_times`.

        Returns
        -------
        dict :
            Dictionary of detector names -> (gate start, gate width)
        """
        if detgates is None:
            detgates = {}
        if refframe is None:
            refframe = self.current_params.get('tc_ref_frame', 'geocentric')
        gatetimes = {}
        for det in self._invpsds:
            start, end = detgates.get(det, (None, None))
            if start is None and gatestart is None or \
                    end is None and gateend is None:
                raise ValueError(f"no gate times given for {det}; need "
                                 f"t_gate_start (t_gate_end) or "
                                 f"t_gate_start_{det.lower()} "
                                 f"(t_gate_end_{det.lower()})")
            if start is None or end is None:
                try:
                    thisdet = self._gate_detectors[det]
                except KeyError:
                    thisdet = self._gate_detectors[det] = Detector(det)
            # account for the time delay between the waveforms of the
            # different detectors
            if self._time_varying_response:
                if start is None:
                    start = thisdet.arrival_time(gatestart, ra, dec, refframe)
                if end is None:
                    end = thisdet.arrival_time(gateend, ra, dec, refframe)
                gatestartdelay = _round_gate_time(start)
                dgatedelay = _round_gate_time(end) - gatestartdelay
            elif start is None and end is None:
                # shift the gate by the delay at tc; the width is unchanged
                if tc is None:
                    tc = self.current_params['tc']
                delay = thisdet.arrival_time_delay(tc, ra, dec, refframe)
                gatestartdelay = _round_gate_time(gatestart + delay)
                dgatedelay = _round_gate_time(gateend - gatestart)
            else:
                # shift the edge that is not given by the delay at tc
                if start is None or end is None:
                    if tc is None:
                        tc = self.current_params['tc']
                    delay = thisdet.arrival_time_delay(tc, ra, dec, refframe)
                    if start is None:
                        start = gatestart + delay
                    else:
                        end = gateend + delay
                gatestartdelay = _round_gate_time(start)
                dgatedelay = _round_gate_time(end) - gatestartdelay
            gatetimes[det] = (gatestartdelay, dgatedelay)
        return gatetimes

    def get_gate_times_hmeco(self):
        """Gets the time to apply a gate based on the current sky position.

        Returns
        -------
        dict :
            Dictionary of detector names -> (gate start, gate width)
        """
        # generate the template waveform
        wfs = self.get_waveforms()
        # get waveform parameters
        params = self.current_params
        spin1 = params['spin1z']
        spin2 = params['spin2z']
        # gate input for ringdown analysis which consideres a start time
        # and an end time
        dgate = params['gate_window']
        meco_f = hybrid_meco_frequency(params['mass1'], params['mass2'], spin1,
                                       spin2)
        # figure out the gate times
        gatetimes = {}
        for det, h in wfs.items():
            invpsd = self._invpsds[det]
            h.resize(len(invpsd))
            ht = h.to_timeseries()
            f_low = int((self._f_lower[det]+1)/h.delta_f)
            sample_freqs = h.sample_frequencies[f_low:].numpy()
            f_idx = numpy.where(sample_freqs <= meco_f)[0][-1]
            # find time corresponding to meco frequency
            t_from_freq = time_from_frequencyseries(
                h[f_low:], sample_frequencies=sample_freqs)
            if t_from_freq[f_idx] > 0:
                gatestartdelay = t_from_freq[f_idx] + float(t_from_freq.epoch)
            else:
                gatestartdelay = t_from_freq[f_idx] + ht.sample_times[-1]
            gatestartdelay = min(gatestartdelay, params['t_gate_start'])
            gatetimes[det] = (gatestartdelay, dgate)
        return gatetimes

    def condition_number(self, det, lindex, rindex):
        """Calculate the condition number associated with the inverse
        covariance matrix used to gate and inpaint. Throws a warning if the
        condition number is greater than 1e16.

        Parameters
        ----------
        det : str
            The detector described by the inverse PSD.
        lindex : int
            The start index of the gate.
        rindex : int
            The end index of the gate.

        Returns
        -------
        float :
            The condition number of the inverse covariance matrix constructed
            from the inverse PSD with the given gate length.
        """
        gate_idx_len = int(rindex - lindex)
        if gate_idx_len not in self._cond.keys():
            conds = {}
        else:
            conds = self._cond[gate_idx_len]
        if det not in conds.keys():
            # construct the matrix
            invpsd = self._invpsds[det]
            tdfilter = invpsd.astype('complex').to_timeseries() * invpsd.delta_t
            mat = scipy.linalg.toeplitz(tdfilter[:rindex-lindex])
            rcond = numpy.linalg.cond(mat)
            # cache the value
            conds[det] = rcond
            self._cond[gate_idx_len] = conds
        else:
            # pull from cache
            rcond = self._cond[gate_idx_len][det]
        if rcond >= 1e16:
            warnings.warn(f'Condition number of inverse covariance matrix is '
                      f'{rcond}; inpainting may be numerically unstable')
        return rcond

    def _lognl(self):
        """Calculates the log of the noise likelihood.
        """
        # clear variables
        lognl = 0.
        self._det_lognls.clear()
        # get the times of the gates
        gate_times = self.get_gate_times()
        for det, invpsd in self._invpsds.items():
            start_index, end_index = self.gate_indices(det)
            # linear estimation
            norm = self.det_lognorm(det, start_index, end_index)
            gatestartdelay, dgatedelay = gate_times[det]
            # we always filter the entire segment starting from kmin, since the
            # gated series may have high frequency components
            slc = slice(self._kmin[det], self._kmax[det])
            # gate the data
            data = self.td_data[det]
            invmat = self.invert_covariance(det)
            gated_dt = data.gate(gatestartdelay + dgatedelay/2,
                                 window=dgatedelay/2, copy=True,
                                 invpsd=invpsd, method='paint',
                                 paint_method=self.paint_method,
                                 paint_invmat=invmat)
            # convert to the frequency series
            gated_d = gated_dt.to_frequencyseries()
            # overwhiten
            gated_d *= invpsd
            d = self.data[det]
            # inner product
            ip = 4 * invpsd.delta_f * d[slc].inner(gated_d[slc]).real  # <d, d>
            dd = norm - 0.5*ip
            # store
            self._det_lognls[det] = dd
            lognl += dd
        return float(lognl)

    def det_lognl(self, det):
        # make sure lognl has been called
        _ = self._trytoget('lognl', self._lognl)
        # the det_lognls dict should have been updated, so can call it now
        return self._det_lognls[det]

    @staticmethod
    def _fd_data_from_strain_dict(opts, strain_dict, psd_strain_dict):
        """Wrapper around :py:func:`data_utils.fd_data_from_strain_dict`.

        Ensures that if the PSD is estimated from data, the inverse spectrum
        truncation uses a Hann window. Sets the low frequency cutoff for the
        inverse PSD to half the likelihood cutoff if not specified by the user.
        """
        if opts.psd_inverse_length and opts.invpsd_trunc_method is None:
            # make sure invpsd truncation is set to hanning
            logging.info("Using Hann window to truncate inverse PSD")
            opts.invpsd_trunc_method = 'hann'
        # set low frequency cutoff for PSDs
        if opts.psd_low_frequency_cutoff is None:
            opts.psd_low_frequency_cutoff = {}
        for d, lfs in opts.low_frequency_cutoff.items():
            if d not in opts.psd_low_frequency_cutoff:
                # set to half the model's likelihood cutoffs
                logging.info(f"Setting low frequency cutoff of {d} PSD to "
                             f"{lfs/2.}")
                opts.psd_low_frequency_cutoff[d] = lfs/2.
        out = fd_data_from_strain_dict(opts, strain_dict, psd_strain_dict)
        return out

    def write_metadata(self, fp, group=None):
        """Adds writing the psds, and analyzed detectors.

        The analyzed detectors, their analysis segments, and the segments
        used for psd estimation are written as
        ``analyzed_detectors``, ``{{detector}}_analysis_segment``, and
        ``{{detector}}_psd_segment``, respectively. These are either written
        to the specified ``group``'s attrs, or to the top level attrs if
        ``group`` is None.

        Parameters
        ----------
        fp : pycbc.inference.io.BaseInferenceFile instance
            The inference file to write to.
        group : str, optional
            If provided, the metadata will be written to the attrs specified
            by group, i.e., to ``fp[group].attrs``. Otherwise, metadata is
            written to the top-level attrs (``fp.attrs``).
        """
        BaseDataModel.write_metadata(self, fp, group=group)
        attrs = fp.getattrs(group=group)
        # write the analyzed detectors and times
        attrs['analyzed_detectors'] = self.detectors
        # store fitting values here
        for det, data in self.data.items():
            key = '{}_analysis_segment'.format(det)
            attrs[key] = [float(data.start_time), float(data.end_time)]
            # store covariance determinant extrapolation info (checkpoint)
            if self.normalize:
                attrs['{}_cov_sample'.format(det)] = self._cov_samples[det]
                attrs['{}_cov_regression'.format(det)] = \
                    self._cov_regressions[det]
        if self._psds is not None and not self.no_save_data:
            fp.write_psd(self._psds, group=group)
        # write the times used for psd estimation (if they were provided)
        for det in self.psd_segments:
            key = '{}_psd_segment'.format(det)
            attrs[key] = list(map(float, self.psd_segments[det]))
        # save the frequency cutoffs
        for det in self.detectors:
            attrs['{}_likelihood_low_freq'.format(det)] = self._f_lower[det]
            if self._f_upper[det] is not None:
                attrs['{}_likelihood_high_freq'.format(det)] = \
                    self._f_upper[det]


class GatedGaussianNoise(BaseGatedGaussian):
    r"""Model that applies a time domain gate, assuming stationary Gaussian
    noise.

    The gate start and end times are set by providing ``t_gate_start`` and
    ``t_gate_end`` parameters, respectively. This will cause the gated times
    to be excised from the analysis. The gate times in a detector may instead
    be given in that detector's frame with ``t_gate_start_{det}`` and
    ``t_gate_end_{det}`` parameters (e.g., ``t_gate_start_h1``); see
    :py:meth:`BaseGatedGaussian.get_gate_times`. For more details on the
    likelihood function and its derivation, see
    `arXiv:2105.05238 <https://arxiv.org/abs/2105.05238>`_.

    .. warning::
        The normalization of the likelihood depends on the gate times. However,
        at the moment, the normalization is not calculated, as it depends on
        the determinant of the truncated covariance matrix (see Eq. 4 of
        arXiv:2105.05238). For this reason it is recommended that you only
        use this model for fixed gate times.

    """
    name = 'gated_gaussian_noise'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, **kwargs):
        # set up the boiler-plate attributes
        super().__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        # create the waveform generator
        self.waveform_generator = create_waveform_generator(
            self.variable_params, self.data,
            waveform_transforms=self.waveform_transforms,
            recalibration=self.recalibration,
            gates=self.gates, **self.static_params)

    @property
    def _extra_stats(self):
        """No extra stats are stored."""
        return []

    @catch_waveform_error
    def _loglikelihood(self):
        r"""Computes the log likelihood after removing the power within the
        given time window,

        .. math::
            \log p(d|\Theta) = -\frac{1}{2} \sum_i
             \left< d_i - h_i(\Theta) | d_i - h_i(\Theta) \right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood.
        """
        # generate the template waveform
        wfs = self.get_waveforms()
        # get the times of the gates
        gate_times = self.get_gate_times()
        logl = 0.
        for det, h in wfs.items():
            invpsd = self._invpsds[det]
            start_index, end_index = self.gate_indices(det)
            norm = self.det_lognorm(det, start_index, end_index)
            gatestartdelay, dgatedelay = gate_times[det]
            # we always filter the entire segment starting from kmin, since the
            # gated series may have high frequency components
            slc = slice(self._kmin[det], self._kmax[det])
            # calculate the residual
            data = self.td_data[det]
            ht = h.to_timeseries()
            res = data - ht
            rtilde = res.to_frequencyseries()
            invmat = self.invert_covariance(det)
            gated_res = res.gate(gatestartdelay + dgatedelay/2,
                                 window=dgatedelay/2, copy=True,
                                 invpsd=invpsd, method='paint',
                                 paint_method=self.paint_method,
                                 paint_invmat=invmat)
            gated_rtilde = gated_res.to_frequencyseries()
            # overwhiten
            gated_rtilde *= invpsd
            rr = 4 * invpsd.delta_f * rtilde[slc].inner(gated_rtilde[slc]).real
            logl += norm - 0.5*rr
        return float(logl)

    @property
    def _extra_stats(self):
        """Adds ``loglr``, plus ``cplx_loglr`` and ``optimal_snrsq`` in each
        detector."""
        return ['loglr', 'maxl_phase'] + ['{}_optimal_snrsq'.format(det) for det in self._data]

    def _nowaveform_loglr(self):
        """Convenience function to set loglr values if no waveform generated.
        """
        setattr(self._current_stats, 'loglikelihood', -numpy.inf)
        # maxl phase doesn't exist, so set it to nan
        setattr(self._current_stats, 'maxl_phase', numpy.nan)
        for det in self._data:
            # snr can't be < 0 by definition, so return 0
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det), 0.)
        return -numpy.inf

    @property
    def multi_signal_support(self):
        """ The list of classes that this model supports in a multi-signal
        likelihood
        """
        return [type(self)]

    def multi_loglikelihood(self, models):
        """ Calculate a multi-model (signal) likelihood
        """
        # Generate the waveforms for each submodel
        wfs = []
        for m in models + [self]:
            wf = m.get_waveforms()
            wfs.append(wf)

        # combine into a single waveform
        combine = {}
        for det in self.data:
            # get max waveform length
            mlen = max([len(x[det]) for x in wfs])
            wfs_resize = [x[det].copy().resize(mlen) for x in wfs]
            combine[det] = sum([x[det] for x in wfs_resize])

        self._current_wfs = combine
        return self._loglikelihood()

    def get_waveforms(self):
        if self._current_wfs is None:
            params = self.current_params
            wfs = self.waveform_generator.generate(**params)
            for det, h in wfs.items():
                # make the same length as the data
                h.resize(len(self.data[det]))
                # apply high pass
                if self.highpass_waveforms:
                    h = highpass_fd(h, self.highpass_waveforms)
                wfs[det] = h
            self._current_wfs = wfs
        return self._current_wfs

    def get_gated_waveforms(self):
        wfs = self.get_waveforms()
        out = {}
        # apply the gate
        for det, h in wfs.items():
            ht = h.to_timeseries()
            invpsd = self._invpsds[det]
            gate_times = self.get_gate_times()
            gatestartdelay, dgatedelay = gate_times[det]
            invmat = self.invert_covariance(det)
            ht = ht.gate(gatestartdelay + dgatedelay/2,
                         window=dgatedelay/2, copy=False,
                         invpsd=invpsd, method='paint',
                         paint_method=self.paint_method,
                         paint_invmat=invmat)
            h = ht.to_frequencyseries()
            out[det] = h
        return out


class GatedGaussianMargPol(BaseGatedGaussian):
    r"""Gated gaussian model with numerical marginalization over polarization.

    This implements the GatedGaussian likelihood with an explicit numerical
    marginalization over polarization angle. This is accomplished using
    a fixed set of integration points distributed uniformly in [0, 2pi).
    By default, 1000 integration points are used.
    The 'polarization_samples' argument can be passed to set an alternate
    number of integration points.
    """
    name = 'gated_gaussian_margpol'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False, 
                 static_params=None,
                 polarization_samples=1000, **kwargs):
        # set up the boiler-plate attributes
        super().__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        # the polarization parameters
        self.polarization_samples = int(polarization_samples)
        self.pol = numpy.linspace(0, 2*numpy.pi, self.polarization_samples,
                                  endpoint=False)
        self.dets = {}
        # create the waveform generator
        self.waveform_generator = create_waveform_generator(
            self.variable_params, self.data,
            waveform_transforms=self.waveform_transforms,
            recalibration=self.recalibration,
            generator_class=generator.FDomainDetFrameTwoPolGenerator,
            **self.static_params)

    def get_waveforms(self):
        if self._current_wfs is not None:
            return self._current_wfs
        params = self.current_params
        wfs = self.waveform_generator.generate(**params)
        for det, (hp, hc) in wfs.items():
            # make the same length as the data
            hp.resize(len(self.data[det]))
            hc.resize(len(self.data[det]))
            # apply high pass
            if self.highpass_waveforms:
                hp = highpass_fd(hp, self.highpass_waveforms)
                hc = highpass_fd(hc, self.highpass_waveforms)
            wfs[det] = (hp, hc)
        self._current_wfs = wfs
        return self._current_wfs

    def get_gated_waveforms(self):
        wfs = self.get_waveforms()
        gate_times = self.get_gate_times()
        out = {}
        for det in wfs:
            invpsd = self._invpsds[det]
            gatestartdelay, dgatedelay = gate_times[det]
            # the waveforms are a dictionary of (hp, hc)
            pols = []
            for h in wfs[det]:
                ht = h.to_timeseries()
                invmat = self.invert_covariance(det)
                ht = ht.gate(gatestartdelay + dgatedelay/2,
                             window=dgatedelay/2, copy=False,
                             invpsd=invpsd, method='paint',
                             paint_method=self.paint_method,
                             paint_invmat=invmat)
                h = ht.to_frequencyseries()
                pols.append(h)
            out[det] = tuple(pols)
        return out

    def get_gate_times_hmeco(self):
        """Gets the time to apply a gate based on the current sky position.
        Returns
        -------
        dict :
            Dictionary of detector names -> (gate start, gate width)
        """
        # generate the template waveform
        wfs = self.get_waveforms()
        # get waveform parameters
        params = self.current_params
        spin1 = params['spin1z']
        spin2 = params['spin2z']
        # gate input for ringdown analysis which consideres a start time
        # and an end time
        dgate = params['gate_window']
        meco_f = hybrid_meco_frequency(params['mass1'], params['mass2'], spin1,
                                       spin2)
        # figure out the gate times
        gatetimes = {}
        # for now only calculating time from plus polarization; should be all
        # that's necessary
        for det, (hp, hc) in wfs.items():
            invpsd = self._invpsds[det]
            hp.resize(len(invpsd))
            ht = hp.to_timeseries()
            f_low = int((self._f_lower[det]+1)/hp.delta_f)
            sample_freqs = hp.sample_frequencies[f_low:].numpy()
            f_idx = numpy.where(sample_freqs <= meco_f)[0][-1]
            # find time corresponding to meco frequency
            t_from_freq = time_from_frequencyseries(
                hp[f_low:], sample_frequencies=sample_freqs)
            if t_from_freq[f_idx] > 0:
                gatestartdelay = t_from_freq[f_idx] + float(t_from_freq.epoch)
            else:
                gatestartdelay = t_from_freq[f_idx] + ht.sample_times[-1]
            gatestartdelay = min(gatestartdelay, params['t_gate_start'])
            gatetimes[det] = (gatestartdelay, dgate)
        return gatetimes

    @property
    def _extra_stats(self):
        """Adds the maxL polarization and corresponding likelihood."""
        return ['maxl_polarization', 'maxl_logl']

    @catch_waveform_error
    def _loglikelihood(self):
        r"""Computes the log likelihood after removing the power within the
        given time window,

        .. math::
            \log p(d|\Theta) = -\frac{1}{2} \sum_i
             \left< d_i - h_i(\Theta) | d_i - h_i(\Theta) \right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood.
        """
        # generate the template waveform
        wfs = self.get_waveforms()
        # get the gated waveforms and data
        gated_wfs = self.get_gated_waveforms()
        gated_data = self.get_gated_data()
        # cycle over
        loglr = 0.
        lognl = 0.
        refframe = self.current_params.get('tc_ref_frame', 'geocentric')
        ref_tc = self.current_params['tc']
        ra = self.current_params['ra']
        dec = self.current_params['dec']
        for det, (hp, hc) in wfs.items():
            # get the antenna patterns
            if det not in self.dets:
                self.dets[det] = Detector(det)
            # calculate tc in frame
            tc = self.dets[det].arrival_time(ref_tc, ra, dec, refframe)
            # evaluate antenna pattern
            fp, fc = self.dets[det].antenna_pattern(ra, dec, self.pol, tc)
            start_index, end_index = self.gate_indices(det)
            norm = self.det_lognorm(det, start_index, end_index)
            # we always filter the entire segment starting from kmin, since the
            # gated series may have high frequency components
            slc = slice(self._kmin[det], self._kmax[det])
            # get the gated values
            gated_hp, gated_hc = gated_wfs[det]
            gated_d = gated_data[det]
            # we'll overwhiten the ungated data and waveforms for computing
            # inner products
            d = self._overwhitened_data[det]
            # overwhiten the hp and hc
            invpsd = self._invpsds[det]
            hp = hp*invpsd
            hc = hc*invpsd
            # get the various gated inner products
            hpd = hp[slc].inner(gated_d[slc]).real  # <hp, d>
            hcd = hc[slc].inner(gated_d[slc]).real  # <hc, d>
            dhp = d[slc].inner(gated_hp[slc]).real  # <d, hp>
            dhc = d[slc].inner(gated_hc[slc]).real  # <d, hc>
            hphp = hp[slc].inner(gated_hp[slc]).real  # <hp, hp>
            hchc = hc[slc].inner(gated_hc[slc]).real  # <hc, hc>
            hphc = hp[slc].inner(gated_hc[slc]).real  # <hp, hc>
            hchp = hc[slc].inner(gated_hp[slc]).real  # <hc, hp>
            dd = d[slc].inner(gated_d[slc]).real  # <d, d>
            # since the antenna patterns are real,
            # <h, d>/2 + <d, h>/2 = fp*(<hp, d>/2 + <d, hp>/2)
            #                     + fc*(<hc, d>/2 + <d, hc>/2)
            hd = fp*(hpd + dhp) + fc*(hcd + dhc)
            # <h, h>/2 = <fp*hp + fc*hc, fp*hp + fc*hc>/2
            #          = fp*fp*<hp, hp>/2 + fc*fc*<hc, hc>/2
            #            + fp*fc*<hp, hc>/2 + fc*fp*<hc, hp>/2
            hh = fp*fp*hphp + fc*fc*hchc + fp*fc*(hphc + hchp)
            # sum up; note that the factor is 2df instead of 4df to account
            # for the factor of 1/2
            loglr += norm + 2*invpsd.delta_f*(hd - hh)
            lognl += -2 * invpsd.delta_f * dd
        # store the maxl polarization
        idx = loglr.argmax()
        setattr(self._current_stats, 'maxl_polarization', self.pol[idx])
        setattr(self._current_stats, 'maxl_logl', loglr[idx] + lognl)
        # compute the marginalized log likelihood
        marglogl = special.logsumexp(loglr) + lognl - numpy.log(len(self.pol))
        return float(marglogl)

    @property
    def multi_signal_support(self):
        """ The list of classes that this model supports in a multi-signal
        likelihood
        """
        return [type(self)]

    @catch_waveform_error
    def multi_loglikelihood(self, models):
        """ Calculate a multi-model (signal) likelihood
        """
        # Generate the waveforms for each submodel
        wfs = []
        for m in models + [self]:
            wf = m.get_waveforms()
            wfs.append(wf)

        # combine into a single waveform
        combine = {}
        for det in self.data:
            # get max waveform length
            mlenp = max([len(x[det][0]) for x in wfs])
            mlenc = max([len(x[det][1]) for x in wfs])
            mlen = max([mlenp, mlenc])
            # resize plus and cross
            wfs[det][0] = wfs[det][0].copy().resize(mlen)
            wfs[det][1] = wfs[det][1].copy().resize(mlen)
            # combine waveforms
            combine[det] = (sum([x[det][0] for x in wfs]), sum([x[det][1]
                                 for x in wfs]))

        self._current_wfs = combine
        return self._loglikelihood()


class GatedGaussianMargPhase(BaseGatedGaussian):
    r"""Gated Gaussian noise model that analytically marginalizes over the
    phase of a signal.

    The phase to be marginalized over is specified by the user using the 
    `ref_phase` argument. If a model consists of multiple modes each with their
    own phase, only the reference phase is marginalized over. All phases must
    be specified with the `phase_names` argument. This can be passed as a list
    or a string delimited by spaces (e.g. 'phase1 phase2 phase3').

    Marginalization is done using explicit numerical integration over 500
    thousand integration points by default. This method assumes that the
    waveform h can be written in terms of an overall phase phi as

        h = h_c * cos(phi) + h_s * sin(phi),

    where h_c and h_s are the waveform with phi set to zero and pi/2
    respectively. The number of integration points can be controlled via the
    `phase_samples` argument.
    """
    name = 'gated_gaussian_margphase'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None,
                 phase_samples=500000, phase_names=None,
                 ref_phase=None, **kwargs):
        # set up the boiler-plate attributes
        super().__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        self.det_names = list(self.data.keys())
        self.dets = {}
        # phase marginalization parameters
        self.phase_samples = int(phase_samples)
        self.phases = numpy.linspace(0, 2*numpy.pi, self.phase_samples,
                                     endpoint=False)
        if ref_phase is None:
            raise KeyError('ref_phase is set to None. Please specify the '
                           'name of the phase parameter to marginalize '
                           'over')
        self.ref_phase = ref_phase
        if phase_names is None:
            logging.warning('No phase_names provided. Assuming single mode '
                            f'specified by ref_phase {ref_phase}')
            self.phase_names = [ref_phase]
        elif type(phase_names) == list:
            self.phase_names = phase_names
        elif type(phase_names) == str:
            self.phase_names = phase_names.split(' ')
        else:
            raise TypeError('Unrecognized format for phase_names arg. Accepts '
                            'string, list, or None')
        # create the waveform generator
        self.waveform_generator = create_waveform_generator(
            self.variable_params, self.data,
            waveform_transforms=self.waveform_transforms,
            recalibration=self.recalibration,
            generator_class=generator.FDomainDetFrameTwoPhaseGenerator,
            **self.static_params)

    def get_waveforms(self):
        r"""Generate the waveforms.
        """
        if self._current_wfs is None:
            params = self.current_params
            # generate the cosine and sine terms
            wfs = self.waveform_generator.generate(phases=self.phase_names, 
                                                   ref_phase=self.ref_phase,
                                                   **params)
            for det, (hc, hs) in wfs.items():
                # make the same length as the data
                hc.resize(len(self.data[det]))
                hs.resize(len(self.data[det]))
                # apply high pass
                if self.highpass_waveforms:
                    hc = highpass_fd(hc, self.highpass_waveforms)
                    hs = highpass_fd(hs, self.highpass_waveforms)
                wfs[det] = (hc, hs)
            self._current_wfs = wfs
        return self._current_wfs

    def get_gated_waveforms(self):
        r"""Generate the gated waveforms.
        """
        wfs = self.get_waveforms()
        out = {}
        # apply the gate
        for det, (hc, hs) in wfs.items():
            hct = hc.to_timeseries()
            hst = hs.to_timeseries()
            invpsd = self._invpsds[det]
            gate_times = self.get_gate_times()
            gatestartdelay, dgatedelay = gate_times[det]
            invmat = self.invert_covariance(det)
            hct = hct.gate(gatestartdelay + dgatedelay/2,
                           window=dgatedelay/2, copy=False,
                           invpsd=invpsd, method='paint',
                           paint_method=self.paint_method,
                           paint_invmat=invmat)
            hst = hst.gate(gatestartdelay + dgatedelay/2,
                           window=dgatedelay/2, copy=False,
                           invpsd=invpsd, method='paint',
                           paint_method=self.paint_method,
                           paint_invmat=invmat)
            hc = hct.to_frequencyseries()
            hs = hst.to_frequencyseries()
            out[det] = (hc, hs)
        return out

    @property
    def _extra_stats(self):
        """Adds the maxL phase and corresponding likelihood."""
        return ['maxl_phase', 'maxl_logl']

    @catch_waveform_error
    def _loglikelihood(self):
        r"""Computes the log likelihood.
        """
        # get waveforms
        wfs = self.get_waveforms()
        gated_wfs = self.get_gated_waveforms()
        # get data
        data = self.get_data()
        gated_data = self.get_gated_data()
        # cycle over all detectors
        norm = 0.
        hchc = 0.
        hchs = 0.
        hshc = 0.
        hshs = 0.
        dhc = 0.
        dhs = 0.
        hcd = 0.
        hsd = 0.
        dd = 0.
        for det in self.det_names:
            if det not in self.dets:
                self.dets[det] = Detector(det)
            # we always filter the entire segment starting from kmin, since the
            # gated series may have high frequency components
            slc = slice(self._kmin[det], self._kmax[det])
            invpsd = self._invpsds[det]
            d = data[det].copy()
            hc, hs = wfs[det]
            gated_d = gated_data[det].copy()
            gated_hc, gated_hs = gated_wfs[det]
            # overwhiten gated waveforms and data
            gated_hc *= 2 * invpsd.delta_f * invpsd
            gated_hs *= 2 * invpsd.delta_f * invpsd
            gated_d *= 2 * invpsd.delta_f * invpsd
            # evaluate the inner products
            hchc += hc[slc].inner(gated_hc[slc]).real
            hchs += hc[slc].inner(gated_hs[slc]).real
            hshc += hs[slc].inner(gated_hc[slc]).real
            hshs += hs[slc].inner(gated_hs[slc]).real
            dhc += d[slc].inner(gated_hc[slc]).real
            dhs += d[slc].inner(gated_hs[slc]).real
            hcd += hc[slc].inner(gated_d[slc]).real
            hsd += hs[slc].inner(gated_d[slc]).real
            dd += d[slc].inner(gated_d[slc]).real
            # get the normalization in this detector
            if self.normalize:
                start_index, end_index = self.gate_indices(det)
            else:
                start_index = end_index = None
            norm += self.det_lognorm(det, start_index, end_index)
        # numerical marginalization over phases
        cphi = numpy.cos(self.phases)
        sphi = numpy.sin(self.phases)
        hh = cphi*cphi*hchc + sphi*sphi*hshs + cphi*sphi*(hchs+hshc)
        dh = cphi*dhc + sphi*dhs
        hd = cphi*hcd + sphi*hsd
        loglr = -(hh-dh-hd)
        lognl = -dd
        # get the maxL phase
        maxlidx = loglr.argmax()
        setattr(self._current_stats, 'maxl_phase', self.phases[maxlidx])
        setattr(self._current_stats, 'maxl_logl', loglr[maxlidx] + lognl + norm)
        # get the marginalized log likelihood ratio
        marglogl = special.logsumexp(loglr) + lognl + norm - numpy.log(self.phase_samples)
        return marglogl

    @property
    def multi_signal_support(self):
        """ The list of classes that this model supports in a multi-signal
        likelihood
        """
        return [type(self)]

    @catch_waveform_error
    def multi_loglikelihood(self, models):
        """ Calculate a multi-model (signal) likelihood
        """
        # Generate the waveforms for each submodel
        wfs = []
        for m in models + [self]:
            wf = m.get_waveforms()
            wfs.append(wf)
        # combine into a single waveform
        combine = {}
        for det in self.data:
            # get max waveform length
            mlen = max([len(x[det]) for x in wfs])
            [x[det].resize(mlen) for x in wfs]
            combine[det] = sum([x[det] for x in wfs])
        self._current_wfs = combine
        return self._loglikelihood()


class GatedGaussianMultimodeMargPhase(BaseGatedGaussian):
    r"""Gated Gaussian noise model that analytically marginalizes over the
    phase of a signal.

    The phase to be marginalized over is specified by the user using the
    `ref_phase` argument. If a model consists of multiple modes each with their
    own phase, only the reference phase is marginalized over. All phases must
    be specified with the `phase_names` argument. This can be passed as a list
    or a string delimited by spaces (e.g. 'phase1 phase2 phase3').

    Marginalization is done using explicit numerical integration over 500
    thousand integration points by default. This method assumes that the
    waveform h can be written in terms of an overall phase phi as

        h = h_c * cos(phi) + h_s * sin(phi),

    where h_c and h_s are the waveform with phi set to zero and pi/2
    respectively. The number of integration points can be controlled via the
    `phase_samples` argument.

    This class also allows functionality to sample over the optimal SNR of each
    mode in the signal. User must specify the names of the amplitude parameters
    for the modes the user wants to sample in SNR space. These specified
    parameters are set to a fiducial value for the purposes of waveform
    generation. A helper function then scales the amplitude of each mode to
    match the sampled SNR.

    If sampling over SNR, the user must specify the names of the modes and
    their respective amplitudes keyed by the name of the corresponding
    SNR parameter name. If, for example, one wants to sample the SNR of a mode
    `foo` with amplitude `amp_foo` and another mode `bar` with amplitude `A_bar`,
    the user must input:

        snr_mode_map={'foo': ('snr_foo', 'amp_foo')}

    The mode keys must match the corresponding output in the waveform generator.
    """
    name = 'gated_gaussian_multimargphase'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None,
                 phase_samples=500000, phase_names=None,
                 ref_phase=None, sample_snrs=False,
                 snr_mode_map=None, fiducial_amp_value=1.,
                 ref_mode=False, **kwargs):
        # caches of the current waveforms and gated waveforms, stacked into
        # 2D arrays; see _stacked
        self._current_wf_stacks = None
        self._current_gated_stacks = None
        # set up the boiler-plate attributes
        super().__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        self.det_names = list(self.data.keys())
        self.dets = {}

        # phase marginalization parameters
        self.phase_samples = int(phase_samples)
        self.phases = numpy.linspace(0, 2*numpy.pi, self.phase_samples,
                                     endpoint=False)
        # the phase dependence of the log likelihood ratio; this is a
        # 5 x phase_samples array of cos, sin, cos^2, sin^2, cos*sin
        cphi = numpy.cos(self.phases)
        sphi = numpy.sin(self.phases)
        self._phase_terms = numpy.array([cphi, sphi, cphi*cphi, sphi*sphi,
                                         cphi*sphi])
        if ref_phase is None:
            raise KeyError('ref_phase is set to None. Please specify the '
                           'name of the phase parameter to marginalize '
                           'over')
        self.ref_phase = ref_phase
        if phase_names is None:
            logging.warning('No phase_names provided. Assuming single mode '
                            f'specified by ref_phase {ref_phase}')
            self.phase_names = [ref_phase]
        elif isinstance(phase_names, list):
            self.phase_names = phase_names
        elif isinstance(phase_names, str):
            self.phase_names = phase_names.split(' ')
        else:
            raise TypeError('Unrecognized format for phase_names arg. Accepts '
                            'string, list, or None')
        self.fiducial_amp_value = float(fiducial_amp_value)

        # if sampling in snr, set names of snrs, amps, and modes
        self.sample_snrs = sample_snrs
        self.amp_names = {}
        self.snr_names = {}
        self.mode_names = []
        if self.sample_snrs:
            if snr_mode_map is None:
                raise ValueError('Must provide SNR/amp map if sampling in SNR')
            for mode, (snr, amp) in snr_mode_map.items():
                self.mode_names.append(mode)
                self.snr_names[mode] = snr
                self.amp_names[mode] = amp

        # specify whether one of the modes is a reference to all other modes;
        # it is assumed that only one mode is given to be the reference
        self.ref_mode = ref_mode
        if self.ref_mode:
            if len(self.amp_names) > 1:
                raise ValueError('More than one mode is specified for SNR '
                                 'sampling. This model only supports one mode '
                                 'sampled in SNR if ref_amp is turned on.')

        # create the waveform generator
        self.waveform_generator = create_waveform_generator(
            self.variable_params, self.data,
            waveform_transforms=self.waveform_transforms,
            recalibration=self.recalibration,
            generator_class=generator.FDomainDetFrameTwoPhaseModesGenerator,
            **self.static_params)

    @classmethod
    def from_config(cls, cp, data_section='data', data=None, psds=None,
                    **kwargs):
        """Adds additional keyword arguments based on config file.

        Additional keyword arguments are:

        * ``sample-snrs`` : Flag whether to sample in SNRs.

        * ``ref-mode`` : Flag whether the given mode to be sampled in SNR is
          the reference, i.e. other mode amplitudes are relative to the given
          mode.

        * ``snr-mode-map`` : Map of mode name output from the waveform
          generator to SNR and amplitude parameter names in that order.
          Syntax: ``MODE:SNR_NAME:AMP_NAME [MODE:SNR_NAME:AMP_NAME ...]``.
          Example: ``220:snr220:amp220 1:snr1:amp_1``
        """
        if cp.has_option('model', 'sample_snrs') or \
            cp.has_option('model', 'sample-snrs'):
            kwargs['sample_snrs'] = True
        if cp.has_option('model', 'ref_mode') or \
            cp.has_option('model', 'ref-mode'):
            kwargs['ref_mode'] = True
        if cp.has_option('model', 'snr-mode-map'):
            snr_mode_map = {}
            parsed_map = cp.get('model', 'snr-mode-map')
            for entry in shlex.split(parsed_map):
                mode, snr, amp = entry.split(':')
                snr_mode_map[mode] = (snr, amp)
            kwargs['snr_mode_map'] = snr_mode_map
        return super().from_config(cp, data_section=data_section,
                                   data=data, psds=psds,
                                   **kwargs)

    def get_waveforms(self):
        r"""Generate the waveforms.

        Returns
        -------
        dict :
            Dictionary of detector names -> modes -> (h_c, h_s), where
            ``h_c`` (``h_s``) is the cosine (sine) term of the mode in the
            detector.
        """
        if self._current_wfs is None:
            params = self.current_params.copy()
            # set specified amplitudes to fiducial value
            for amp in self.amp_names.values():
                params[amp] = self.fiducial_amp_value
            # generate the cosine and sine terms
            wfs = self.waveform_generator.generate(phases=self.phase_names,
                                                   ref_phase=self.ref_phase,
                                                   **params)
            out = {}
            stacks = {}
            for det, modes in wfs.items():
                # all of the terms are stored in a single
                # (number of terms) x (number of frequencies) array, so
                # that operations on them can be done all at once
                names = list(modes.keys())
                x0 = modes[names[0]][0]
                nfreq = len(self.data[det])
                stack = numpy.zeros((2*len(names), nfreq), dtype=x0.dtype)
                for ii, mode in enumerate(names):
                    for jj, x in enumerate(modes[mode]):
                        # this is the same as resizing to the length of the
                        # data
                        n = min(len(x), nfreq)
                        stack[2*ii+jj, :n] = x.numpy()[:n]
                if self.highpass_waveforms:
                    tlen = 2 * (nfreq - 1)
                    stack *= highpass_response(
                        tlen, 1. / (tlen * float(x0.delta_f)),
                        self.highpass_waveforms)
                    # same as highpass_fd
                    stack[:, 0] = stack[:, 0].real
                    stack[:, -1] = stack[:, -1].real
                stacks[det] = stack
                out[det] = {mode: tuple(
                    FrequencySeries(stack[2*ii+jj], delta_f=x0.delta_f,
                                    epoch=x0.epoch, copy=False)
                    for jj in range(2))
                    for ii, mode in enumerate(names)}
            self._current_wf_stacks = (out, stacks)
            self._current_wfs = out
        return self._current_wfs

    def get_gated_waveforms(self):
        r"""Generate the gated waveforms.

        Returns
        -------
        dict :
            Dictionary of detector names -> modes -> gated (h_c, h_s).
        """
        wfs = self.get_waveforms()
        stacks = self._stacked(wfs)
        gate_times = self.get_gate_times()
        out = {}
        gated_stacks = {}
        for det, modes in wfs.items():
            gate = gate_times[det]
            gatestartdelay, dgatedelay = gate
            # all of the series share the same gate, so gate them together
            names = list(modes.keys())
            fdata = stacks[det]
            # the data has the same gate, so if it hasn't been gated yet with
            # this gate, gate it along with the waveforms; the result is
            # stored to the same cache that get_gated_data uses
            cache = self._gated_data.setdefault(det, {})
            gate_data = gate not in cache
            if gate_data:
                fdata = numpy.concatenate(
                    [fdata, self.data[det].numpy()[None, :]])
            x0 = modes[names[0]][0]
            gated = batch_gate_and_paint_fd_array(
                fdata, float(x0.delta_f), float(x0.start_time),
                gatestartdelay + dgatedelay/2, dgatedelay/2,
                self._invpsds[det], paint_method=self.paint_method,
                invmat=self.invert_covariance(det))
            if gate_data:
                d = self.data[det]
                cache.clear()
                cache[gate] = FrequencySeries(gated[-1], delta_f=d.delta_f,
                                              epoch=d.epoch, copy=False)
                gated = gated[:-1]
            gated_stacks[det] = gated
            out[det] = {mode: tuple(
                FrequencySeries(gated[2*ii+jj], delta_f=x0.delta_f,
                                epoch=x0.epoch, copy=False)
                for jj in range(2))
                for ii, mode in enumerate(names)}
        self._current_gated_stacks = (out, gated_stacks)
        return out

    def _stacked(self, wfs):
        """Returns the terms of the given waveforms, stacked into a
        ``(number of terms) x (number of frequencies)`` array for each
        detector.

        If the waveforms are the ones that were created by
        :py:meth:`get_waveforms` or :py:meth:`get_gated_waveforms`, the
        arrays that were stored when they were created are returned.
        Otherwise, the arrays are created.
        """
        for cached in (self._current_wf_stacks, self._current_gated_stacks):
            if cached is not None and cached[0] is wfs:
                return cached[1]
        return {det: numpy.array([x.numpy() for terms in modes.values()
                                  for x in terms])
                for det, modes in wfs.items()}

    @property
    def _extra_stats(self):
        """Adds the maxL phase and corresponding likelihood."""
        return ['maxl_phase', 'maxl_logl'] + \
            [f'scale_factor_{mode}' for mode in self.mode_names]

    def _det_inner_products(self, det, wfs, gated_wfs, gated_data):
        r"""Computes the inner products in the given detector.

        The inner products are computed between all of the terms
        :math:`u_k`, where :math:`k` runs over the modes and, for each mode,
        :math:`(h_c, h_s)`. The second argument of each inner product is
        gated.

        Returns
        -------
        uv : array
            The ``2M x 2M`` array of :math:`\left<u_k, u_l\right>`.
        ud : array
            The length ``2M`` array of
            :math:`\left<u_k, d\right> + \left<d, u_k\right>`.
        dd : float
            :math:`\left<d, d\right>`.
        """
        # we always filter the entire segment starting from kmin, since the
        # gated series may have high frequency components
        slc = slice(self._kmin[det], self._kmax[det])
        invpsd = self._invpsds[det]
        fac = 4 * invpsd.delta_f
        # overwhiten the ungated data and waveforms; the terms are stacked
        # into (number of terms) x (number of frequencies) arrays, so that
        # all of the inner products can be done with matrix products
        d = self._overwhitened_data[det].numpy()[slc]
        gated_d = gated_data[det].numpy()[slc]
        # the terms are stacked in the same order as the modes, since the
        # modes are the keys of the waveform dictionaries
        hs = self._stacked(wfs)[det][:, slc] * invpsd.numpy()[slc]
        hs = hs.conj()
        gated_hs = self._stacked(gated_wfs)[det][:, slc]
        # <u, v> for all u, v; note that this is not symmetric
        uv = fac * (hs @ gated_hs.T).real
        # <u, d> + <d, u>
        ud = fac * ((hs @ gated_d).real + (gated_hs @ d.conj()).real)
        # <d, d>
        dd = fac * numpy.vdot(d, gated_d).real
        return uv, ud, dd

    def _scale_factors(self, modes, uv):
        """Computes the scale factor of every mode.

        Parameters
        ----------
        modes : list
            The names of the modes, in the order of the terms in ``uv``.
        uv : array
            The ``2M x 2M`` array of inner products between the terms,
            summed over detectors.

        Returns
        -------
        dict :
            Dictionary of mode -> scale factor.
        """
        scales = {}
        for ii, mode in enumerate(modes):
            snr = None
            if mode in self.snr_names:
                snr = self.current_params.get(self.snr_names[mode])
            if snr is None:
                scales[mode] = 1.
                continue
            # the fiducial network SNR^2 of the mode is <h_c, h_c>
            scales[mode] = snr / uv[2*ii, 2*ii]**0.5
        # scale all other modes by the reference mode's scale factor if spec'd
        if self.ref_mode:
            rf = self.mode_names[0]
            for mode in modes:
                if mode not in self.mode_names:
                    scales[mode] = scales[mode] * scales[rf]
        return scales

    @catch_waveform_error
    def _loglikelihood(self):
        r"""Computes the log likelihood marginalized over the reference
        phase.

        Returns
        -------
        float
            The value of the marginalized log likelihood.
        """
        # get waveforms
        wfs = self.get_waveforms()
        gated_wfs = self.get_gated_waveforms()
        # get data
        gated_data = self.get_gated_data()
        modes = list(wfs[self.det_names[0]].keys())
        # the inner products between all of the terms, summed over
        # detectors
        uv = 0.
        ud = 0.
        lognl = 0.
        for det in self.det_names:
            duv, dud, dd = self._det_inner_products(det, wfs, gated_wfs,
                                                    gated_data)
            uv = uv + duv
            ud = ud + dud
            # get the normalization in this detector
            if self.normalize:
                start_index, end_index = self.gate_indices(det)
            else:
                start_index = end_index = None
            lognl += self.det_lognorm(det, start_index, end_index) - 0.5*dd
        # get the scale factor of each mode
        scales = self._scale_factors(modes, uv)
        if numpy.isnan(list(scales.values())).any():
            # a negative showed up somewhere in the snr calcs;
            # reject this waveform
            raise FailedWaveformError
        for mode in modes:
            setattr(self._current_stats, f'scale_factor_{mode}',
                    scales[mode])
        # scale the terms; the cosine (sine) terms are the even (odd) ones
        weights = numpy.repeat([scales[mode] for mode in modes], 2)
        uv = uv * numpy.outer(weights, weights)
        ud = ud * weights
        cidx = slice(0, None, 2)
        sidx = slice(1, None, 2)
        # the coefficients of (cos, sin, cos^2, sin^2, cos*sin), from
        # <h, d>/2 + <d, h>/2 - <h, h>/2
        coeffs = numpy.array([
            0.5 * ud[cidx].sum(),
            0.5 * ud[sidx].sum(),
            -0.5 * uv[cidx, cidx].sum(),
            -0.5 * uv[sidx, sidx].sum(),
            -0.5 * (uv[cidx, sidx].sum() + uv[sidx, cidx].sum())])
        # numerical marginalization over phases
        loglr = coeffs @ self._phase_terms
        # get the maxL phase
        maxlidx = loglr.argmax()
        self._current_stats.maxl_phase = self.phases[maxlidx]
        self._current_stats.maxl_logl = loglr[maxlidx] + lognl
        # get the marginalized log likelihood
        marglogl = special.logsumexp(loglr) + lognl \
            - numpy.log(self.phase_samples)
        return float(marglogl)

    def _nowaveform_handler(self):
        """Sets the extra stats to nan if no waveform was generated."""
        for stat in ['maxl_phase', 'maxl_polarization']:
            setattr(self._current_stats, stat, numpy.nan)
        for mode in self.mode_names:
            setattr(self._current_stats, f'scale_factor_{mode}', numpy.nan)
        self._current_stats.maxl_logl = -numpy.inf
        return -numpy.inf

    @property
    def multi_signal_support(self):
        """ The list of classes that this model supports in a multi-signal
        likelihood
        """
        return [type(self)]

    @catch_waveform_error
    def multi_loglikelihood(self, models):
        """ Calculate a multi-model (signal) likelihood
        """
        # Generate the waveforms for each submodel
        if any(m.sample_snrs for m in models + [self]):
            raise NotImplementedError("multi-signal likelihoods are not "
                                      "supported when sampling SNRs")
        wfs = []
        for m in models + [self]:
            wf = m.get_waveforms()
            wfs.append(wf)
        # combine into a single waveform
        combine = {}
        for det in self.data:
            # get max waveform length
            mlen = max([len(x[det]) for x in wfs])
            [x[det].resize(mlen) for x in wfs]
            combine[det] = sum([x[det] for x in wfs])
        self._current_wfs = combine
        return self._loglikelihood()


class GatedGaussianMultimodeMargPhasePol(BaseGatedGaussian):
    r"""Gated Gaussian noise model that numerically marginalizes over both
    polarization and the phase of a (multimode) signal.

    This combines :py:class:`GatedGaussianMargPol` and
    :py:class:`GatedGaussianMultimodeMargPhase`. The waveform in detector
    :math:`i` is written as

    .. math::

        h_i(\phi, \psi) = \sum_m s_m(\psi) \left\{
            F^i_{+}(\psi)\left[P^m_c\cos\phi + P^m_s\sin\phi\right]
            + F^i_{\times}(\psi)\left[X^m_c\cos\phi + X^m_s\sin\phi\right]
            \right\},

    where :math:`P^m_{c,s}` (:math:`X^m_{c,s}`) are the plus (cross)
    polarizations of mode :math:`m`, with the reference phase set to 0
    (:math:`c`) and :math:`\pi/2` (:math:`s`), and :math:`s_m` is an optional
    amplitude scale factor (see below). All other mode phases are shifted by
    the same amount as the reference phase, so that the non-reference phases
    act as phases relative to the reference mode. The phase :math:`\phi` and
    the polarization :math:`\psi` are then marginalized over numerically
    using a fixed grid of points uniformly distributed in :math:`[0, 2\pi)`
    in each (:math:`2\pi` is excluded, since it is the same angle as 0).

    Expanding the gated inner products gives a log likelihood ratio that is
    linear in :math:`(\cos\phi, \sin\phi, \cos^2\phi, \sin^2\phi,
    \cos\phi\sin\phi)`, with coefficients that are (quadratic) functions of
    the polarization. The likelihood over the entire grid can therefore be
    evaluated with a single matrix product.

    The phase to marginalize over is set with the ``ref_phase`` argument. The
    reference phase is always set to zero for waveform generation, so it
    should not be a variable parameter. All of the phase parameters of the
    waveform must be listed with the ``phase_names`` argument. This can be
    passed as a list or a string delimited by spaces (e.g.
    'phase1 phase2 phase3').

    The number of integration points in phase and polarization can be set
    with the ``phase_samples`` and ``polarization_samples`` arguments,
    respectively. By default, 512 points are used in each. This is sufficient
    for ringdown SNRs of ~30, for which the marginalized log likelihood is
    converged to numerical precision. The number of points needed grows
    linearly with the SNR, since the width of the likelihood peak in phase
    and polarization shrinks as ~1/SNR. Using roughly 5 x SNR points in each
    (e.g., 512 for an SNR of ~100, or 4096 for an SNR of ~1000) keeps the
    error in the marginalized log likelihood below ~1e-4. Polarization is the
    more demanding of the two, since the antenna patterns vary as twice the
    polarization angle.

    As with :py:class:`GatedGaussianMultimodeMargPhase`, the optimal SNR of
    modes may be sampled instead of their amplitude by setting
    ``sample_snrs`` and providing a ``snr_mode_map``. This maps the name of
    each mode (as returned by the waveform generator) to the names of its SNR
    and amplitude parameters; e.g., to sample the SNR ``snr220`` of a mode
    ``220`` with amplitude ``amp220``, use
    ``snr_mode_map={'220': ('snr220', 'amp220')}``. The waveform is generated
    with the amplitudes of those modes set to ``fiducial_amp_value``, and each
    mode is then rescaled by :math:`s_m = \rho_m / \rho^{\rm fid}_m` so that
    it has the network SNR given by its SNR parameter. The fiducial SNR
    :math:`\rho^{\rm fid}_m` is computed from the cosine (reference phase = 0)
    term. Since it depends on the polarization, the scale factor is computed
    for every polarization in the marginalization grid. If ``ref_mode`` is
    set, only one mode may be given in the ``snr_mode_map``; modes whose SNR
    is not sampled are then scaled by that mode's scale factor, so that their
    amplitude remains relative to it.

    If SNRs are not sampled, the modes are summed before gating, since they
    all share the same scale.

    This model requires a waveform approximant that returns the individual
    modes (e.g., ``TdModesfromFinalMassSpin``).
    """
    name = 'gated_gaussian_multimargphasepol'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None,
                 phase_samples=512, polarization_samples=512,
                 phase_names=None, ref_phase=None, sample_snrs=False,
                 snr_mode_map=None, fiducial_amp_value=1., ref_mode=False,
                 **kwargs):
        # caches of the current waveforms and gated waveforms, stacked into
        # 2D arrays; see _stacked
        self._current_wf_stacks = None
        self._current_gated_stacks = None
        # set up the boiler-plate attributes
        super().__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        self.det_names = list(self.data.keys())
        self.dets = {}
        # phase marginalization parameters
        if ref_phase is None:
            raise KeyError('ref_phase is set to None. Please specify the '
                           'name of the phase parameter to marginalize '
                           'over')
        if ref_phase in self.variable_params:
            raise ValueError(f'ref_phase {ref_phase} is marginalized over; '
                             'it should not be a variable parameter')
        self.ref_phase = ref_phase
        if phase_names is None:
            logging.warning('No phase_names provided. Assuming single mode '
                            f'specified by ref_phase {ref_phase}')
            self.phase_names = [ref_phase]
        elif isinstance(phase_names, list):
            self.phase_names = phase_names
        elif isinstance(phase_names, str):
            self.phase_names = phase_names.split(' ')
        else:
            raise TypeError('Unrecognized format for phase_names arg. Accepts '
                            'string, list, or None')
        if self.ref_phase not in self.phase_names:
            self.phase_names.append(self.ref_phase)
        self.phase_samples = int(phase_samples)
        self.phases = numpy.linspace(0, 2*numpy.pi, self.phase_samples,
                                     endpoint=False)
        # polarization marginalization parameters
        self.polarization_samples = int(polarization_samples)
        self.pol = numpy.linspace(0, 2*numpy.pi, self.polarization_samples,
                                  endpoint=False)
        # the phase dependence of the log likelihood ratio; this is a
        # 5 x phase_samples array of cos, sin, cos^2, sin^2, cos*sin
        cphi = numpy.cos(self.phases)
        sphi = numpy.sin(self.phases)
        self._phase_terms = numpy.array([cphi, sphi, cphi*cphi, sphi*sphi,
                                         cphi*sphi])
        # if sampling in snr, set names of snrs, amps, and modes
        self.sample_snrs = sample_snrs
        self.fiducial_amp_value = float(fiducial_amp_value)
        self.amp_names = {}
        self.snr_names = {}
        self.mode_names = []
        if self.sample_snrs:
            if snr_mode_map is None:
                raise ValueError('Must provide SNR/amp map if sampling in SNR')
            for mode, (snr, amp) in snr_mode_map.items():
                self.mode_names.append(mode)
                self.snr_names[mode] = snr
                self.amp_names[mode] = amp
        # specify whether one of the modes is a reference to all other modes;
        # it is assumed that only one mode is given to be the reference
        self.ref_mode = ref_mode
        if self.ref_mode and len(self.amp_names) > 1:
            raise ValueError('More than one mode is specified for SNR '
                             'sampling. This model only supports one mode '
                             'sampled in SNR if ref_mode is turned on.')
        # create the waveform generator
        gen_class = generator.FDomainDetFrameTwoPolTwoPhaseModesGenerator
        self.waveform_generator = create_waveform_generator(
            self.variable_params, self.data,
            waveform_transforms=self.waveform_transforms,
            recalibration=self.recalibration,
            generator_class=gen_class,
            **self.static_params)

    @classmethod
    def from_config(cls, cp, data_section='data', data=None, psds=None,
                    **kwargs):
        """Adds additional keyword arguments based on config file.

        Additional keyword arguments are:

        * ``sample-snrs`` : Flag whether to sample in SNRs.

        * ``ref-mode`` : Flag whether the given mode to be sampled in SNR is
          the reference, i.e. other mode amplitudes are relative to the given
          mode.

        * ``snr-mode-map`` : Map of mode name output from the waveform
          generator to SNR and amplitude parameter names in that order.
          Syntax: ``MODE:SNR_NAME:AMP_NAME [MODE:SNR_NAME:AMP_NAME ...]``.
          Example: ``220:snr220:amp220 1:snr1:amp_1``
        """
        if cp.has_option('model', 'sample_snrs') or \
                cp.has_option('model', 'sample-snrs'):
            kwargs['sample_snrs'] = True
        if cp.has_option('model', 'ref_mode') or \
                cp.has_option('model', 'ref-mode'):
            kwargs['ref_mode'] = True
        if cp.has_option('model', 'snr-mode-map'):
            snr_mode_map = {}
            for entry in shlex.split(cp.get('model', 'snr-mode-map')):
                mode, snr, amp = entry.split(':')
                snr_mode_map[mode] = (snr, amp)
            kwargs['snr_mode_map'] = snr_mode_map
        return super().from_config(cp, data_section=data_section,
                                   data=data, psds=psds, **kwargs)

    def get_waveforms(self):
        r"""Generate the waveforms.

        If SNRs are sampled, the amplitudes of the modes in the
        ``snr_mode_map`` are set to the fiducial value.

        Returns
        -------
        dict :
            Dictionary of detector names -> modes -> (hp_c, hp_s, hc_c, hc_s),
            where ``hp_c, hp_s`` (``hc_c, hc_s``) are the cosine and sine
            terms of the plus (cross) polarization. The time shift to each
            detector has been applied, but not the antenna patterns. If SNRs
            are not being sampled, the modes are summed together and stored
            under a single ``'summed'`` key.
        """
        if self._current_wfs is None:
            params = self.current_params.copy()
            # the reference phase is marginalized over
            params[self.ref_phase] = 0.
            # set specified amplitudes to the fiducial value
            for amp in self.amp_names.values():
                params[amp] = self.fiducial_amp_value
            wfs = self.waveform_generator.generate(phases=self.phase_names,
                                                   ref_phase=self.ref_phase,
                                                   **params)
            out = {}
            stacks = {}
            for det, modes in wfs.items():
                # all of the terms are stored in a single
                # (number of terms) x (number of frequencies) array, so
                # that operations on them can be done all at once
                names = list(modes.keys())
                x0 = modes[names[0]][0]
                nfreq = len(self.data[det])
                stack = numpy.zeros((4*len(names), nfreq), dtype=x0.dtype)
                for ii, mode in enumerate(names):
                    for jj, x in enumerate(modes[mode]):
                        # this is the same as resizing to the length of the
                        # data
                        n = min(len(x), nfreq)
                        stack[4*ii+jj, :n] = x.numpy()[:n]
                if not self.sample_snrs:
                    # all modes have the same scale, so sum them now to
                    # reduce the number of series that need to be highpassed
                    # and gated
                    stack = stack.reshape(len(names), 4, nfreq).sum(axis=0)
                    names = ['summed']
                if self.highpass_waveforms:
                    tlen = 2 * (nfreq - 1)
                    stack *= highpass_response(
                        tlen, 1. / (tlen * float(x0.delta_f)),
                        self.highpass_waveforms)
                    # same as highpass_fd
                    stack[:, 0] = stack[:, 0].real
                    stack[:, -1] = stack[:, -1].real
                stacks[det] = stack
                out[det] = {mode: tuple(
                    FrequencySeries(stack[4*ii+jj], delta_f=x0.delta_f,
                                    epoch=x0.epoch, copy=False)
                    for jj in range(4))
                    for ii, mode in enumerate(names)}
            self._current_wf_stacks = (out, stacks)
            self._current_wfs = out
        return self._current_wfs

    def get_gated_waveforms(self):
        r"""Generate the gated waveforms.

        Returns
        -------
        dict :
            Dictionary of detector names -> modes -> gated
            (hp_c, hp_s, hc_c, hc_s).
        """
        wfs = self.get_waveforms()
        stacks = self._stacked(wfs)
        gate_times = self.get_gate_times()
        out = {}
        gated_stacks = {}
        for det, modes in wfs.items():
            gate = gate_times[det]
            gatestartdelay, dgatedelay = gate
            # all of the series share the same gate, so gate them together
            names = list(modes.keys())
            fdata = stacks[det]
            # the data has the same gate, so if it hasn't been gated yet with
            # this gate, gate it along with the waveforms; the result is
            # stored to the same cache that get_gated_data uses
            cache = self._gated_data.setdefault(det, {})
            gate_data = gate not in cache
            if gate_data:
                fdata = numpy.concatenate(
                    [fdata, self.data[det].numpy()[None, :]])
            x0 = modes[names[0]][0]
            gated = batch_gate_and_paint_fd_array(
                fdata, float(x0.delta_f), float(x0.start_time),
                gatestartdelay + dgatedelay/2, dgatedelay/2,
                self._invpsds[det], paint_method=self.paint_method,
                invmat=self.invert_covariance(det))
            if gate_data:
                d = self.data[det]
                cache.clear()
                cache[gate] = FrequencySeries(gated[-1], delta_f=d.delta_f,
                                              epoch=d.epoch, copy=False)
                gated = gated[:-1]
            gated_stacks[det] = gated
            out[det] = {mode: tuple(
                FrequencySeries(gated[4*ii+jj], delta_f=x0.delta_f,
                                epoch=x0.epoch, copy=False)
                for jj in range(4))
                for ii, mode in enumerate(names)}
        self._current_gated_stacks = (out, gated_stacks)
        return out

    def _stacked(self, wfs):
        """Returns the terms of the given waveforms, stacked into a
        ``(number of terms) x (number of frequencies)`` array for each
        detector.

        If the waveforms are the ones that were created by
        :py:meth:`get_waveforms` or :py:meth:`get_gated_waveforms`, the
        arrays that were stored when they were created are returned.
        Otherwise, the arrays are created.
        """
        for cached in (self._current_wf_stacks, self._current_gated_stacks):
            if cached is not None and cached[0] is wfs:
                return cached[1]
        return {det: numpy.array([x.numpy() for terms in modes.values()
                                  for x in terms])
                for det, modes in wfs.items()}

    def get_gate_times_hmeco(self):
        """Gets the time to apply a gate based on the current sky position.

        The time is calculated from the cosine term of the plus polarization,
        summed over modes.

        Returns
        -------
        dict :
            Dictionary of detector names -> (gate start, gate width)
        """
        # generate the template waveform
        wfs = self.get_waveforms()
        # get waveform parameters
        params = self.current_params
        spin1 = params['spin1z']
        spin2 = params['spin2z']
        dgate = params['gate_window']
        meco_f = hybrid_meco_frequency(params['mass1'], params['mass2'], spin1,
                                       spin2)
        # figure out the gate times
        gatetimes = {}
        for det, modes in wfs.items():
            hp = sum(terms[0] for terms in modes.values())
            ht = hp.to_timeseries()
            f_low = int((self._f_lower[det]+1)/hp.delta_f)
            sample_freqs = hp.sample_frequencies[f_low:].numpy()
            f_idx = numpy.where(sample_freqs <= meco_f)[0][-1]
            # find time corresponding to meco frequency
            t_from_freq = time_from_frequencyseries(
                hp[f_low:], sample_frequencies=sample_freqs)
            if t_from_freq[f_idx] > 0:
                gatestartdelay = t_from_freq[f_idx] + float(t_from_freq.epoch)
            else:
                gatestartdelay = t_from_freq[f_idx] + ht.sample_times[-1]
            gatestartdelay = min(gatestartdelay, params['t_gate_start'])
            gatetimes[det] = (gatestartdelay, dgate)
        return gatetimes

    @property
    def _extra_stats(self):
        """Adds the maxL phase, polarization, and corresponding likelihood,
        and the scale factor of each mode whose SNR is sampled (at the maxL
        polarization)."""
        return ['maxl_phase', 'maxl_polarization', 'maxl_logl'] + \
            [f'scale_factor_{mode}' for mode in self.mode_names]

    def _nowaveform_handler(self):
        """Sets the extra stats to nan if no waveform was generated."""
        for stat in ['maxl_phase', 'maxl_polarization']:
            setattr(self._current_stats, stat, numpy.nan)
        for mode in self.mode_names:
            setattr(self._current_stats, f'scale_factor_{mode}', numpy.nan)
        setattr(self._current_stats, 'maxl_logl', -numpy.inf)
        return -numpy.inf

    def _det_inner_products(self, det, modes, wfs, gated_wfs, gated_data):
        r"""Computes the inner products in the given detector.

        The inner products are computed between all of the terms
        :math:`u_k`, where :math:`k` runs over the modes and, for each mode,
        :math:`(P_c, P_s, X_c, X_s)`. The second argument of each inner
        product is gated.

        Returns
        -------
        uv : array
            The ``4M x 4M`` array of :math:`\left<u_k, u_l\right>`.
        ud : array
            The length ``4M`` array of
            :math:`\left<u_k, d\right> + \left<d, u_k\right>`.
        dd : float
            :math:`\left<d, d\right>`.
        """
        # we always filter the entire segment starting from kmin, since the
        # gated series may have high frequency components
        slc = slice(self._kmin[det], self._kmax[det])
        invpsd = self._invpsds[det]
        fac = 4 * invpsd.delta_f
        # overwhiten the ungated data and waveforms; the terms are stacked
        # into (number of terms) x (number of frequencies) arrays, so that
        # all of the inner products can be done with matrix products
        d = self._overwhitened_data[det].numpy()[slc]
        gated_d = gated_data[det].numpy()[slc]
        # the terms are stacked in the same order as the modes, since the
        # modes are the keys of the waveform dictionaries
        hs = self._stacked(wfs)[det][:, slc] * invpsd.numpy()[slc]
        hs = hs.conj()
        gated_hs = self._stacked(gated_wfs)[det][:, slc]
        # <u, v> for all u, v; note that this is not symmetric
        uv = fac * (hs @ gated_hs.T).real
        # <u, d> + <d, u>
        ud = fac * ((hs @ gated_d).real + (gated_hs @ d.conj()).real)
        # <d, d>
        dd = fac * numpy.vdot(d, gated_d).real
        return uv, ud, dd

    def _scale_factors(self, modes, fpfc, uvs):
        """Computes the scale factor of every mode at every polarization.

        Returns
        -------
        dict :
            Dictionary of mode -> array of scale factors, one for each
            polarization.
        """
        scales = {}
        if not self.sample_snrs:
            return {mode: numpy.ones(self.polarization_samples)
                    for mode in modes}
        # the fiducial SNR^2 of each mode from the cosine term; this is
        # <h_c, h_c> with h_c = fp*P_c + fc*X_c, summed over detectors
        for ii, mode in enumerate(modes):
            snr = None
            if mode in self.snr_names:
                snr = self.current_params.get(self.snr_names[mode])
            if snr is None:
                scales[mode] = numpy.ones(self.polarization_samples)
                continue
            pc, xc = 4*ii, 4*ii + 2
            fid_snrsq = 0.
            for det in self.det_names:
                fp, fc = fpfc[det]
                uv = uvs[det]
                fid_snrsq += fp*fp*uv[pc, pc] + fc*fc*uv[xc, xc] \
                    + fp*fc*(uv[pc, xc] + uv[xc, pc])
            scales[mode] = snr / fid_snrsq**0.5
        # scale all other modes by the reference mode's scale factor if spec'd
        if self.ref_mode:
            rf = self.mode_names[0]
            for mode in modes:
                if mode not in self.mode_names:
                    scales[mode] = scales[mode] * scales[rf]
        return scales

    @catch_waveform_error
    def _loglikelihood(self):
        r"""Computes the log likelihood marginalized over phase and
        polarization.

        Returns
        -------
        float
            The value of the marginalized log likelihood.
        """
        # get waveforms
        wfs = self.get_waveforms()
        gated_wfs = self.get_gated_waveforms()
        # get data
        gated_data = self.get_gated_data()
        refframe = self.current_params.get('tc_ref_frame', 'geocentric')
        ref_tc = self.current_params['tc']
        ra = self.current_params['ra']
        dec = self.current_params['dec']
        modes = list(wfs[self.det_names[0]].keys())
        # compute the antenna patterns and inner products in each detector
        fpfc = {}
        uvs = {}
        uds = {}
        lognl = 0.
        for det in self.det_names:
            if det not in self.dets:
                self.dets[det] = Detector(det)
            # calculate tc in frame
            tc = self.dets[det].arrival_time(ref_tc, ra, dec, refframe)
            # evaluate antenna pattern
            fpfc[det] = self.dets[det].antenna_pattern(ra, dec, self.pol, tc)
            uvs[det], uds[det], dd = self._det_inner_products(
                det, modes, wfs, gated_wfs, gated_data)
            # get the normalization in this detector
            start_index, end_index = self.gate_indices(det)
            norm = self.det_lognorm(det, start_index, end_index)
            lognl += norm - 0.5*dd
        # get the scale factor of each mode at each polarization
        scales = self._scale_factors(modes, fpfc, uvs)
        if any(numpy.isnan(s).any() for s in scales.values()):
            # a negative showed up somewhere in the snr calcs;
            # reject this waveform
            raise FailedWaveformError
        # indices of the cosine (P_c, X_c) and sine (P_s, X_s) terms
        cidx = numpy.array([4*ii + jj for ii in range(len(modes))
                            for jj in (0, 2)])
        sidx = cidx + 1
        # the coefficients of (cos, sin, cos^2, sin^2, cos*sin) at each
        # polarization, summed over detectors
        coeffs = numpy.zeros((self.polarization_samples, 5))
        for det in self.det_names:
            fp, fc = fpfc[det]
            uv = uvs[det]
            ud = uds[det]
            # the weight of each term u_k at each polarization is the mode
            # scale factor times fp (for plus terms) or fc (for cross terms);
            # this is a polarization_samples x 4M array
            weights = numpy.stack(
                [scales[mode] * f for mode in modes
                 for f in (fp, fp, fc, fc)], axis=1)
            wc = weights[:, cidx]
            ws = weights[:, sidx]
            # <h, d>/2 + <d, h>/2
            coeffs[:, 0] += 0.5 * wc @ ud[cidx]
            coeffs[:, 1] += 0.5 * ws @ ud[sidx]
            # -<h, h>/2
            coeffs[:, 2] -= 0.5 * numpy.einsum(
                'pk,kl,pl->p', wc, uv[numpy.ix_(cidx, cidx)], wc)
            coeffs[:, 3] -= 0.5 * numpy.einsum(
                'pk,kl,pl->p', ws, uv[numpy.ix_(sidx, sidx)], ws)
            coeffs[:, 4] -= 0.5 * numpy.einsum(
                'pk,kl,pl->p', wc,
                uv[numpy.ix_(cidx, sidx)] + uv[numpy.ix_(sidx, cidx)].T, ws)
        # the log likelihood ratio over the polarization x phase grid
        loglr = coeffs @ self._phase_terms
        # store the maxl phase and polarization
        maxidx = loglr.argmax()
        maxloglr = loglr.flat[maxidx]
        polidx, phaseidx = numpy.unravel_index(maxidx, loglr.shape)
        setattr(self._current_stats, 'maxl_phase', self.phases[phaseidx])
        setattr(self._current_stats, 'maxl_polarization', self.pol[polidx])
        setattr(self._current_stats, 'maxl_logl', maxloglr + lognl)
        for mode in self.mode_names:
            setattr(self._current_stats, f'scale_factor_{mode}',
                    scales[mode][polidx])
        # compute the marginalized log likelihood; this is the same as
        # special.logsumexp(loglr), but is faster since we already have the
        # max
        loglr -= maxloglr
        numpy.exp(loglr, out=loglr)
        marglogl = maxloglr + numpy.log(loglr.sum()) + lognl \
            - numpy.log(loglr.size)
        return float(marglogl)

    @property
    def multi_signal_support(self):
        """ The list of classes that this model supports in a multi-signal
        likelihood
        """
        return [type(self)]

    @catch_waveform_error
    def multi_loglikelihood(self, models):
        """ Calculate a multi-model (signal) likelihood
        """
        if any(m.sample_snrs for m in models + [self]):
            raise NotImplementedError("multi-signal likelihoods are not "
                                      "supported when sampling SNRs")
        # Generate the waveforms for each submodel; since SNRs are not
        # sampled, each only has a single (summed) mode
        wfs = [m.get_waveforms() for m in models + [self]]
        # combine into a single waveform
        combine = {}
        for det in self.data:
            mlen = max(len(x) for wf in wfs for x in wf[det]['summed'])
            summed = []
            for idx in range(4):
                terms = [wf[det]['summed'][idx].copy() for wf in wfs]
                for x in terms:
                    x.resize(mlen)
                summed.append(sum(terms))
            combine[det] = {'summed': tuple(summed)}
        self._current_wfs = combine
        return self._loglikelihood()
