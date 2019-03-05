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

from pycbc.types import TimeSeries
from pycbc.waveform import generator
from pycbc.opt import LimitedSizeDict
from pycbc.waveform import NoWaveformError

from .base_data import BaseDataModel


LOG2PI = numpy.log(2*numpy.pi)


class TimeDomainGaussian(BaseDataModel):
    name = 'tdgaussian'

    def __init__(self, variable_params, data, psds=None, static_params=None,
                 analysis_start_time=None, analysis_end_time=None,
                 **kwargs):
        # set up the boiler-plate attributes
        super(TimeDomainGaussian, self).__init__(
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
        self.end_time = list(self.data.values())[0].start_time
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
        self.current_waveforms = {}
        self.current_data = {}
        if psds is None:
            psds = {det: None for det in self.data}
        for det, psd in psds.items():
            if psd is None:
                # assume white: in this case, the response function is just
                # a delta function
                rss = TimeSeries(numpy.zeros(dlens), delta_t=data[det].delta_t,
                                 epoch=self.start_time)
                rss[0] = 1.
            else:
                rss = psd.astype(numpy.complex).to_timeseries()
                rss[1:] /= 2.
            Nrss = len(rss)
            N = len(self.data[det])
            if Nrss < N:
                raise ValueError("1/psd.delta_f < data length")
            self.autocorrs[det] = rss
            # create the covariance matrix
            cov = numpy.zeros((N, N), dtype=float)
            rss = rss.numpy()
            initshift = int((analysis_start_time-self.start_time)
                            /data[det].delta_t)
            rss = numpy.roll(rss, initshift)
            for ii in range(N):
                cov[ii, :] = rss[initshift:N+initshift]
                rss = numpy.roll(rss, 1)
            self._fullcov[det] = cov
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
        cov = self._fullcov[detector]
        if gstart is not None or gstop is not None:
            if gstart is not None and gstop is None:
                # easy
                cov = cov[:gstart, :gstart]
            elif gstart is None and gstop is not None:
                # also easy
                cov = cov[gstop:, gstop:]
            else:
                # have to rebuild
                q1 = cov[:gstart, :gstart]
                q2 = cov[:gstart, gstop:]
                q3 = cov[gstop:, :gstart]
                q4 = cov[gstop:, gstop:]
                # stack
                cov = numpy.block([[q1, q2], [q3, q4]])
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
            raise ValueError("determinant of covariance matrix is negative")
        self._logdet[detector][gstart, gstop] = det
        return det

    def invcov(self, detector, gstart=None, gstop=None):
        """Inverts the covariance matrix."""
        try:
            # try to return from cache
            return self._invcov[detector][gstart, gstop]
        except KeyError:
            pass
        cov = self.cov(detector, gstart, gstop)
        invcov = numpy.linalg.inv(cov)
        # cache for next time
        self._invcov[detector][gstart, gstop] = invcov
        return invcov

    def _lognl(self, data, detector, invcov, gstart=None, gstop=None):
        try:
            return self._dd[detector][gstart, gstop]
        except KeyError:
            dd = numpy.matmul(data, numpy.matmul(invcov, data))
            self._dd[detector, gstart, gstop] = dd
            return dd

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
            h = h.time_slice(d.start_time, d.end_time)
            if t_gate_start is not None:
                # figure out the time to start the gate in the detector frame
                det_gate_start = t_gate_start + tflight
                gstart = int(float(det_gate_start-h.start_time)/h.delta_t)
                gstart = min(len(d), max(0, gstart))
            else:
                gstart = None
            if t_gate_end is not None:
                # figure out the time to stop the gate in the detector frame
                det_gate_end = t_gate_end + tflight
                gstop = int(numpy.ceil(float(det_gate_end-h.start_time)
                            /h.delta_t))
                gstop = min(len(d), max(0, gstop))
            else:
                gstop = None
            if gstart is not None or gstop is not None:
                sample_times = h.sample_times
                keep = numpy.ones(len(h), dtype=bool)
                keep[slice(gstart, gstop)] = False
                h = h[keep]
                d = d[keep]
                h.detector_tc = det_tc
                # DELETE ME
                h.sample_times = sample_times[keep]
                d.sample_times = sample_times[keep]
            self.current_waveforms[det] = h
            self.current_data[det] = d
            invcov = self.invcov(det, gstart, gstop)
            ovwh = numpy.matmul(invcov, h)
            hd = 2*numpy.matmul(d, ovwh)
            hh = 2*numpy.matmul(h, ovwh)
            dd = 2*self._lognl(d, det, invcov, gstart, gstop)
            logdet = self.logdet(det, gstart, gstop)
            m = len(d)
            denom = 0.5*(logdet + m*LOG2PI)
            thislognl = -0.5*dd - denom
            loglr = hd - 0.5*hh
            logl += loglr + thislognl
            lognl += thislognl
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det),
                    2*loglr)
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
