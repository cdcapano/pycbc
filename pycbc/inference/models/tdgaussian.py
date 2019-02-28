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

from pycbc.opt import LimitedSizeDict

from .base_data import BaseDataModel


class TimeDomainGaussian(BaseDataModel):
    name = 'tdgaussian'

    def __init__(self, variable_params, data, psds, static_params=None,
                 **kwargs):
        # set up the boiler-plate attributes
        super(GaussianNoise, self).__init__(variable_params, data,
                                            static_params=static_params,
                                            **kwargs)
        # create the waveform generator
        self._waveform_generator = create_waveform_generator(
            self.variable_params, self.data, recalibration=self.recalibration,
            gates=self.gates, **self.static_params)
        # check that the data sets all have the same lengths
        dlens = numpy.array([len(d) for d in self.data.values()])
        if not all(dlens == dlens[0]):
            raise ValueError("all data must be of the same length")
        # get the autocorrelation function from the psd
        self.cachesize = 10
        self.autocorrs = {}
        self._fullcov = {}
        for det, psd in psds.items():
            rss = psd.timeseries()/2.
            N = len(rss)
            if N != len(self.data[det]):
                raise ValueError("psd/data length mismatch")
            self.autocorrs[det] = rss
            # create the covariance matrix
            cov = numpy.zeros((N, N)), dtype=float)
            for ii in range(N):
                for jj in range(N):
                    cov[ii, jj] = rss[ii-jj]
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
        if gstart is not None or gstop is not None:
            if gstart and not gstop:
                # easy
                cov = cov[:gstart, :gstart]
            elif gstop and not gstart:
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
        cov = self.cov(detector, gstart=gstart, gstop=gstop
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

    def _lognl(self, detector, data, invcov, gstart=None, gstop=None):
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
        # put gates in terms of time since start
        if t_gate_start is not None:
            t_gate_start -= self.epoch
        if t_gate_end is not None:
            t_gate_end -= self.epoch
        # generate the waveform
        try:
            wfs = self._waveform_generator.generate(**params)
        except NoWaveformError:
            return 0.
        logl = 0.
        lognl = 0.
        for det, h in wfs.items():
            if t_gate_start is not None:
                # figure out the time to start the gate in the detector frame
                det_gate_start = t_gate_start + (h.det_tc - h.start_time )
                gstart = int(det_gate_start/h.delta_t)
            else:
                gstart = None
            if t_gate_end is not None:
                # figure out the time to stop the gate in the detector frame
                det_gate_end = t_gate_end + (h.det_tc - h.start_time )
                gstop = int(numpy.ceil(det_gate_end/h.delta_t))
            else:
                gstop = None
            keep = slice(gstart, gstop)
            h = h[keep]
            d = self.data[det][keep]
            invcov = self.invcov(det, gstart, gstop)
            ovwh = numpy.matmul(invcov, h)
            hd = numpy.matmul(d, ovwh)
            hh = numpy.matmul(h, ovwh)
            dd = self._lognl(detector, data, invcov, gstart, gstop)
            logdet = self.logdet(detector, gstart, gstop)
            m = len(d)
            denom = 0.5*(logdet + m*numpy.log(2*numpy.pi))
            logl += hd - 0.5*hh - 0.5*dd - denom
            lognl += dd
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det),
                    0.5*hh)
            setattr(self._current_stats, '{}_lognl'.format(det), 0.5*dd)
            setattr(self._current_stats, '{}_logdetcov'.format(det), logdet)
            setattr(self._current_stats, '{}_datalen'.format(det), m)
        self._current_stats.lognl = 0.5*lognl
        return logl


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
    generator_function = generator.select_waveform_generator(approximant)
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
