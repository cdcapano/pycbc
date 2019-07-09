# Copyright (C) 2019  Collin Capano
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

"""This module provides model classes for a gapped Gaussian noise.
"""

import numpy
import logging

from pycbc.types import (TimeSeries, FrequencySeries)
from pycbc.waveform import generator
from pycbc.opt import LimitedSizeDict
from pycbc.waveform import NoWaveformError

from .base_data import BaseDataModel


LOG2PI = numpy.log(2*numpy.pi)


class GatedGaussian(BaseDataModel):
    name = 'gated_gaussian'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 static_params=None,
                 analysis_start_time=None, analysis_end_time=None,
                 **kwargs):
        # set up the boiler-plate attributes
        super(GatedGaussian, self).__init__(
            variable_params, data, static_params=static_params, **kwargs)
        # create the waveform generator
        self._waveform_generator = create_waveform_generator(
            self.variable_params, self.data, recalibration=self.recalibration,
            gates=self.gates, **self.static_params)
        # check that the data sets all have the same lengths
        dlens = numpy.array([len(d) for d in self.data.values()])
        if not all(dlens == dlens[0]):
            raise ValueError("all data must be of the same length")
        self.start_time = list(self.data.values())[0].start_time
        self.end_time = list(self.data.values())[0].end_time
        if analysis_start_time is None:
            analysis_start_time = self.start_time
        if analysis_end_time is None:
            analysis_end_time = self.end_time
        self.analysis_start_time = analysis_start_time
        self.analysis_end_time = analysis_end_time
        self.data = {det: d.time_slice(analysis_start_time, analysis_end_time)
                     for det, d in self.data.items()}
        # get the autocorrelation function from the psd
        self.cachesize = 10
        self.autocorrs = {}
        self._fullcov = {}
        self._fullinvcov = {}
        self.current_waveforms = {}
        self.current_data = {}
        if psds is None:
            psds = {det: None for det in self.data}
        self._psds = {}
        for det, psd in psds.items():
            delta_t = data[det].delta_t
            if psd is None:
                # assume white: in this case, the response function is just
                # a delta function
                rss = TimeSeries(numpy.zeros(dlens[0]), delta_t=delta_t,
                                 epoch=self.start_time)
                rss[0] = 2*delta_t
            else:
                # stick psd outside of low frequency range
                kmin = int(low_frequency_cutoff/psd.delta_f)
                maxpsd = psd[kmin:].max()
                psd[:kmin] = maxpsd #2*delta_t
                # set nyquist to the same
                psd[len(psd)-1] = maxpsd
                self._psds[det] = psd
                rss = psd.astype(numpy.complex).to_timeseries()
                rss /= 2.
            Nrss = len(rss)
            N = len(self.data[det])
            if Nrss < N:
                raise ValueError("1/psd.delta_f < data length")
            self.autocorrs[det] = rss
            # create the covariance matrix
            rss = rss.numpy()
            initshift = int((analysis_start_time-self.start_time)
                            /data[det].delta_t)
            self._fullcov[det] = self._create_matrix(rss, N, initshift)
            # create the inverse covariance matrix
            invpsd = numpy.zeros(len(psd), dtype=float)
            #nppsd = psd.numpy()
            #idx = nppsd != 0
            #invpsd[idx] = 1./(nppsd[idx])
            #invpsd = FrequencySeries(invpsd, delta_f=psd.delta_f,
            #                         epoch=data[det].start_time)
            #invpsd = FrequencySeries(numpy.zeros(N), delta_f=psd.delta_f,
            #                         epoch=data[det].start_time)
            #invpsd = 1./psd
            #indx = numpy.zeros((N, N))
            #colidx, rowidx = numpy.meshgrid(numpy.arange(N), numpy.arange(N))
            #cosarg = 2*numpy.pi*(rowidx-colidx)/float(N)
            #invrss = 2 * delta_t**2 * \
            #    invpsd.astype(numpy.complex).to_timeseries()
            #self._fullinvcov[det] = self._create_matrix(invrss, N, initshift)
        self._cov = {det: LimitedSizeDict(size_limit=self.cachesize)
                     for det in data}
        self._invcov = {det: LimitedSizeDict(size_limit=self.cachesize)
                        for det in data}
        self._logdet = {det: LimitedSizeDict(size_limit=self.cachesize)
                             for det in data}
        self._dd = {det: LimitedSizeDict(size_limit=self.cachesize)
                    for det in data}

    @property
    def _extra_stats(self):
        """Adds ``loglr``, plus ``cplx_loglr`` and ``optimal_snrsq`` in each
        detector."""
        return ['lognl'] + \
               ['{}_lognl'.format(det) for det in self._data] + \
               ['{}_optimal_snrsq'.format(det) for det in self._data] + \
               ['{}_logdetcov'.format(det) for det in self._data] + \
               ['{}_datalen'.format(det) for det in self._data] 

    @staticmethod
    def _create_matrix(timeseries, N, initshift):
        """Creates a square matrix from the given timeseries."""
        matrix = numpy.zeros((N, N), dtype=float)
        timeseries = numpy.roll(timeseries, initshift)
        for ii in range(N):
            matrix[ii, :] = timeseries[initshift:N+initshift]
            timeseries = numpy.roll(timeseries, 1)
        return matrix

    @staticmethod
    def _slice_matrix(matrix, gstart=None, gstop=None):
        """Applies a gap to the given square matrix."""
        if gstart is not None or gstop is not None:
            if gstart is not None and gstop is None:
                # easy
                matrix = matrix[:gstart, :gstart]
            elif gstart is None and gstop is not None:
                # also easy
                matrix = matrix[gstop:, gstop:]
            else:
                # have to rebuild
                q1 = matrix[:gstart, :gstart]
                q2 = matrix[:gstart, gstop:]
                q3 = matrix[gstop:, :gstart]
                q4 = matrix[gstop:, gstop:]
                # stack
                matrix = numpy.block([[q1, q2], [q3, q4]])
        return matrix

    def cov(self, detector, gstart=None, gstop=None):
        """Returns the covariance matrix.

        Parameters
        ----------
        detector : str
            The name of the detector to retrieve.
        gstart : int, optional
            Gate the covariance matrix starting from the given index.
        gstop : int, optional
            Gate the covariance matrix up to the given index.
        """
        if gstart is None and gstop is None:
            # just return the full covariance matrix
            return self._fullcov[detector]
        try:
            return self._cov[detector][gstart, gstop]
        except KeyError:
            pass
        cov = self._slice_matrix(self._fullcov[detector], gstart, gstop)
        # cache for next time
        self._cov[detector][gstart, gstop] = cov
        return cov

    def logdet(self, detector, gstart=None, gstop=None):
        """The log of the determinant of the covariance matrix."""
        try:
            return self._logdet[detector][gstart, gstop]
        except KeyError:
            pass
        cov = self.cov(detector, gstart=gstart, gstop=gstop)
        sign, det = numpy.linalg.slogdet(cov)
        if sign == 0:
            raise ValueError("singular covariance matrix")
        elif sign == -1:
            logging.warn("determinant of covariance matrix may be negative")
            #raise ValueError("determinant of covariance matrix is negative")
        self._logdet[detector][gstart, gstop] = det
        return det

    def invcov(self, detector, gstart=None, gstop=None):
        """Inverts the covariance matrix."""
        #if gstart is None and gstop is None:
        #    # just return the full covariance matrix
        #    return self._fullinvcov[detector]
        try:
            return self._invcov[detector][gstart, gstop]
        except KeyError:
            pass
        # sliced covariance matrix
        cov = self.cov(detector, gstart, gstop)
        invcov = numpy.linalg.inv(cov)
        self._invcov[detector][gstart, gstop] = invcov
        return invcov
        #invcov = self._slice_matrix(self._fullinvcov[detector], gstart, gstop)
        # cache for next time
        #self._invcov[detector][gstart, gstop] = invcov
        #return invcov

    def _lognl(self, data, detector, invcov, gstart=None, gstop=None):
        try:
            return self._dd[detector][gstart, gstop]
        except KeyError:
            dd = numpy.matmul(data, numpy.matmul(invcov, data))
            self._dd[detector, gstart, gstop] = dd
            return dd

    @staticmethod
    def _shift_and_slice(data, t_gate_start=None, t_gate_end=None,
                         keepidx=None):
        """Shifts the data by sub sample, and excises the exclusion region."""
        # convert times to indices
        if t_gate_start is not None:
            gstart = float(t_gate_start-data.start_time)/data.delta_t
            gstart = min(len(data), max(0, gstart))
        else:
            gstart = None
        if t_gate_end is not None:
            gstop = float(t_gate_end-data.start_time)/data.delta_t
            gstop = min(len(data), max(0, gstop))
        else:
            gstop = None
        # figure out if the gate is on an integer time or not; if not, shift
        # the data so that it is
        if gstart is not None:
            remainder = 0 # gstart % 1 # FIXME: not doing shifts for now
            gstart = int(gstart)
            if remainder:
                # shift the data backward
                data = data.cyclic_time_shift(-remainder*data.delta_t)
        if gstop is not None:
            remainder = 0 # gstop % 1  FIXME: not doing shifts for now
            gstop = int(numpy.ceil(gstop))
            if remainder:
                # shift the data after the gap forward
                shiftd = data.cyclic_time_shift(remainder*data.delta_t)
                data[gstop:] = shiftd[gstop:]
        # slice the data
        if gstart is not None or gstop is not None:
            sample_times = data.sample_times  # XXX: delete me
            if keepidx is None:
                keepidx = numpy.ones(len(data), dtype=bool)
                keepidx[slice(gstart, gstop)] = False
            data = data[keepidx]
            data.sample_times = sample_times[keepidx]  # XXX: delete me
        return data, gstart, gstop, keepidx

    def _loglikelihood(self):
        params = self.current_params.copy()
        t_gate_start = params.pop('t_gate_start', None)
        t_gate_end = params.pop('t_gate_end', None)
        try:
            wfs = self._waveform_generator.generate(**params)
        except NoWaveformError:
            self.current_waveforms.clear()
            self.current_data.clear()
            return 0.
        logl = 0.
        lognl = 0.
        for det, h in wfs.items():
            d = self.data[det]
            det_tc = h.detector_tc
            tflight = det_tc - params['tc']
            h = h.time_slice(h.start_time, h.end_time)
            # figure out the time to start/stop the gate in the detector frame
            det_gate_start = t_gate_start
            if det_gate_start is not None:
                det_gate_start += tflight
            det_gate_end = t_gate_end
            if det_gate_end is not None:
                det_gate_end += tflight
            d, gstart, gstop, keep = self._shift_and_slice(d, det_gate_start,
                                                           det_gate_end)
            h, _, _, _ = self._shift_and_slice(h, det_gate_start, det_gate_end,
                                               keepidx=keep)
            h.detector_tc = det_tc
            self.current_waveforms[det] = h
            self.current_data[det] = d
            invcov = self.invcov(det, gstart, gstop)
            ovwh = numpy.matmul(invcov, h)
            hd = numpy.matmul(d, ovwh)
            hh = numpy.matmul(h, ovwh)
            dd = self._lognl(d, det, invcov, gstart, gstop)
            logdet = self.logdet(det, gstart, gstop)
            m = len(d)
            denom = 0.5*(logdet + m*LOG2PI)
            thislognl = -0.5*dd - denom
            logl += hd - 0.5*hh + thislognl
            lognl += thislognl
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det),
                    hh)
            setattr(self._current_stats, '{}_lognl'.format(det), thislognl)
            setattr(self._current_stats, '{}_logdetcov'.format(det), logdet)
            setattr(self._current_stats, '{}_datalen'.format(det), m)
        self._current_stats.lognl = lognl
        return logl

    def _loglr(self):
        return self.loglikelihood - self._current_stats.lognl


#
# =============================================================================
#
#                               Support functions
#
# =============================================================================
#


def create_waveform_generator(variable_params, data,
                              recalibration=None, gates=None,
                              **static_params):
    """Creates a time domain waveform generator for use with a model.

    Parameters
    ----------
    variable_params : list of str
        The names of the parameters varied.
    data : dict
        Dictionary mapping detector names to either a
        :py:class:`<pycbc.types.TimeSeries TimeSeries>` or
        :py:class:`<pycbc.types.FrequencySeries FrequencySeries>`.
    recalibration : dict, optional
        Dictionary mapping detector names to
        :py:class:`<pycbc.calibration.Recalibrate>` instances for
        recalibrating data.
    gates : dict of tuples, optional
        Dictionary of detectors -> tuples of specifying gate times. The
        sort of thing returned by :py:func:`pycbc.gate.gates_from_cli`.

    Returns
    -------
    pycbc.waveform.FDomainDetFrameGenerator
        A waveform generator for frequency domain generation.
    """
    # figure out what generator to use based on the approximant
    try:
        approximant = static_params['approximant']
    except KeyError:
        raise ValueError("no approximant provided in the static args")
    generator_function = generator.select_waveform_generator(approximant,
                                                             prefer='td')
    # get data parameters; we'll just use one of the data to get the
    # values, then check that all the others are the same
    delta_f = None
    for d in data.values():
        if delta_f is None:
            delta_f = d.delta_f
            delta_t = d.delta_t
            start_time = d.start_time
        else:
            if not all([d.delta_f == delta_f, d.delta_t == delta_t,
                        d.start_time == start_time]):
                raise ValueError("data must all have the same delta_t, "
                                 "delta_f, and start_time")
    waveform_generator = generator.TDomainDetFrameGenerator(
        generator_function, epoch=start_time,
        variable_args=variable_params, detectors=list(data.keys()),
        delta_f=delta_f, delta_t=delta_t,
        recalib=recalibration, gates=gates,
        **static_params)
    return waveform_generator
