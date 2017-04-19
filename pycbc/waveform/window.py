# Copyright (C) 2017  Collin Capano
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This modules provides classes and functions for windowing waveforms.
"""

import numpy
from pycbc.waveform.waveform import NoWaveformError
from pycbc.waveform.utils import time_from_frequencyseries
from pycbc.window import TimeDomainWindow, WindowBoundsError
from pycbc import pnutils
from pycbc.types import TimeSeries, FrequencySeries

class WaveformTDWindow(TimeDomainWindow):
    """Windows waveforms at specified times and/or frequencies.

    When initialized, a taper time, frequency, or frequency function may
    be specified for each side of the window. These are used to determine
    where to apply the left and right tapers when `apply_window` is called.
    If a frequency is provided for a taper, the time to apply the taper
    is estimated using the stationary phase approximation. If a frequency
    function is provided, a frequency to apply the taper for the parameters
    of a waveform is calculated, then from that a time estimated.

    Unlike `TimeDomainWindow`, times for applying the left and right taper are
    measured from the coalescence time of the waveform, which is assumed to be
    at the end of the data segment. Times before the coalescence time are
    negative; times after are positive.

    Instances of this class may be called like a function, in which case
    `apply_window` is called. See that function for more details.

    Parameters
    ----------
    left_taper_time : {None, float, "start"}
        If a float, the time in seconds relative to the coalescence time at
        which to start the left taper. In this case, the same time will be used
        for all waveforms. Alternatively, the string "start" may be passed.
        In that case, the start of the waveform (estimated by looking for the
        first non-zero value) will be used for the start of the left taper.
    right_taper_time : float, optional
        The time in seconds relative to the coalescence time at which to
        end the right taper. If provided, the same time will be used for
        all waveforms. Note that unlike `left_taper_time`, "start" is not an
        option (ending the right taper at the start of the waveform would just
        result in a NoWaveformError).
    left_taper_frequency : {None, float, str}
        If a float, the frequency, in Hz, of the waveform at which time to
        start the left taper. In this case, the same frequency will be used
        for all waveforms. Alternatively, a string may be provided that gives
        the name of a function to use for computing a frequency. This can
        be any string recognized by `pnutils.named_frequency_cutoffs`. If
        set, the waveform's parameters need to be provided when calling
        `apply_window`.  To convert from frequency to time, the stationary
        phase approximation is used. 
    right_taper_frequency : {None, float, str}
        Same as `left_taper_frequency`, but for the right side.
    \**kwargs :
        All other keyword arguments are passed to `TimeDomainWindow`. See
        that class for details.

    Raises
    ------
    NoWaveformError
        Raised when `apply_window` is called on the waveform and the resulting
        left (right) time of the window  occurs after (before) the end (start)
        of the waveform, such that the entire waveform would be zeroed.
    """
    def __init__(self, left_taper_time=None, right_taper_time=None,
                 left_taper_frequency=None, right_taper_frequency=None,
                 **kwargs):
        # initialize the window
        super(WaveformTDWindow, self).__init__(**kwargs)
        # add the location settings
        left_loc_provided = (left_taper_time is not None or
                             left_taper_frequency is not None)
        if self.left_taper == 'lal' and left_loc_provided:
            raise ValueError("The lal taper function does not take a "
                             "start time or frequency")
        elif self.left_taper is not None and not left_loc_provided:
            raise ValueError("Non-lal taper functions require either a taper "
                             "time, taper frequency, or frequency function")
        self.left_taper_time = left_taper_time
        self.left_taper_frequency = left_taper_frequency
        # if left_taper_frequency is in pnutils.named_frequency_cutoffs, assume
        # it is a function
        try:
            self.left_taper_freqfunc = pnutils.named_frequency_cutoffs[
                left_taper_frequency]
        except KeyError:
            self.left_taper_freqfunc = None
        # right
        right_loc_provided = (right_taper_time is not None or
                              right_taper_frequency is not None)
        if self.right_taper == 'lal' and right_loc_provided:
                raise ValueError("The lal taper function does not take a "
                                 "end time or frequency")
        elif self.right_taper is not None and not right_loc_provided:
            raise ValueError("Non-lal taper functions require either a taper "
                             "time, taper frequency, or frequency function")
        self.right_taper_time = right_taper_time
        self.right_taper_frequency = right_taper_frequency
        # if right_taper_frequency is in pnutils.named_frequency_cutoffs,
        # assume it is a function
        try:
            self.right_taper_freqfunc = pnutils.named_frequency_cutoffs[
                right_taper_frequency]
        except KeyError:
            self.right_taper_freqfunc = None

    def _apply_to_frequencyseries(self, h, break_time=0., params=None,
                                  ifo=None, copy=True):
        """Applies window assuming h is a FrequencySeries.
        """
        # confine break time to within the data; this allows the break time
        # to be specified as either the time before the coalescence or the time
        # after the coalescence
        break_time = break_time % (1./h.delta_f)
        # figure out where the waveform has support
        nzidx = numpy.nonzero(abs(h))[0]
        if len(nzidx) == 0:
            raise NoWaveformError("waveform has no non-zero values")
        kmin, kmax = nzidx[0], nzidx[-1]
        # figure out times
        left_time = self.left_taper_time
        if left_time == 'start':
            # we'll estimate the time of the first non-zero frequency
            if kmin+2 > kmax:
                # means there are only two frequencies with non-zero support,
                # just assume the waveform spans the entire segment
                left_time = -len(h)/h.delta_f
            else:
                left_time = time_from_frequencyseries(h[kmin:kmin+2])[0]
        left_freq = self.left_taper_frequency
        if self.left_taper_freqfunc is not None:
            if params is None:
                raise ValueError("must provide waveform parameters for the "
                                 "frequency function to use for the left")
            left_freq = self.left_taper_freqfunc(params)
        #
        #   left taper
        #
        if left_freq is not None:
            # we only need to analyze h right around the desired frequency
            k = int(left_freq / h.delta_f)
            if k >= kmax:
                # left taper starts after the waveform ends
                raise NoWaveformError("left taper starts after the end of the "
                                      "waveform")
            elif k < kmin:
                # left taper starts before the waveform begins; so nothing to
                # apply
                t = None
            else:
                t = time_from_frequencyseries(h[k:k+2])[0]
            if left_time is None:
                left_time = t
            elif t is not None:
                left_time = max(t, left_time)
        if left_time is not None:
            # TimeDomainWindow measures time from the start of the segment,
            # so adjust
            dur = 1./h.delta_f
            left_time = dur + left_time - break_time
        #
        #   right taper
        #
        right_time = self.right_taper_time
        right_freq = self.right_taper_frequency
        if self.right_taper_freqfunc is not None:
            if params is None:
                raise ValueError("must provide waveform parameters for the "
                                 "frequency function to use for the right")
            right_freq = self.right_taper_freqfunc(params)
        if right_freq is not None:
            # we only need to analyze h right around the desired frequency
            k = int(right_freq / h.delta_f)
            if k < kmin:
                # right taper starts before the waveform starts
                raise NoWaveformError("right taper ends before the start of "
                                      "the waveform")
            elif k >= kmax:
                # right taper ends after the waveform ends, so nothing to apply
                t = None
            else:
                t = time_from_frequencyseries(h[k:k+2])[0]
            if right_time is None:
                right_time = t
            elif t is not None:
                right_time = min(t, right_time)
        if right_time is not None:
            # TimeDomainWindow measures time from the start of the segment,
            # so adjust
            dur = 1./h.delta_f
            right_time = dur + right_time - break_time
        try:
            h = super(WaveformTDWindow, self).apply_window(h,
                      left_time=left_time, right_time=right_time,
                      break_time=break_time, ifo=ifo, copy=copy)
        except WindowBoundsError as e:
            raise NoWaveformError(e)
        return h

    def _apply_to_timeseries(self, h, break_time=0., params=None,
                             ifo=None, copy=True):
        """Applies window assuming h is a TimeSeries.
        """
        if self.left_taper_frequency is not None or \
                self.right_taper_frequency is not None:
            # we'll need to compute t(f), so convert to frequency domain and
            # do everything there
            h = h.to_frequencyseries()
            h = self._apply_to_frequencyseries(h, break_time=break_time,
                                               params=params, ifo=ifo,
                                               copy=False).to_timeseries()
        else:
            # confine break time to within the data; this allows the break
            # time to be specified as either the time before the coalescence
            # or the time after the coalescence
            break_time = break_time % (len(h)*h.delta_t)
            # left
            left_time = self.left_taper_time
            if left_time == 'start':
                # estimate the start of the waveform
                breakidx = int(break_time / h.delta_t)
                nzidx = numpy.nonzero(h[breakidx:])[0]
                if len(nzidx) == 0:
                    # nothing from the break idx to the end, try the rest of
                    # the waveform
                    nzidx = numpy.nonzero(h[:breakidx])[0]
                    if len(nzidx) == 0:
                        # still nothing, means the waveform is empty
                        raise NoWaveformError("waveform has no non-zero "
                                              "values")
                    left_time = (len(h) - breakidx + nzidx[0])*h.delta_t
                else:
                    left_time = nzidx[0]*h.delta_t
                # Note: left time is now measured from the start of the segment
                # after the segment is rolled such that the break time starts
                # at the beginning, as is needed for TimeDomainWindow
            else:
                # TimeDomainWindow measures time from the start of the segment,
                # so adjust
                left_time = len(h)*h.delta_t + left_time - break_time
            # right
            right_time = self.right_taper_time
            if right_time is not None:
                # TimeDomainWindow measures time from the start of the segment,
                # so adjust
                right_time = len(h)*h.delta_t + right_time - break_time
            try:
                h = super(WaveformTDWindow, self).apply_window(h,
                          left_time=left_time, right_time=right_time,
                          break_time=break_time, ifo=ifo, copy=copy)
            except WindowBoundsError as e:
                raise NoWaveformError(e)
        return h

    def apply_window(self, h, break_time=0., params=None, ifo=None, copy=True):
        """Applies the window to the given waveform.

        This function differs from `TimeDomainWindow.apply_window` in that it
        does not take a left or right taper time. Instead, the taper times are
        computed from the `(left|right)`
        `(taper_time|taper_frequency|taper_freqfunc)` attributes. Also, the
        `break_time` (the time at which to define the start/end of the time
        series) should be specified relative to the coalescence time, with
        negative indicating before the coalesence and positive after.

        Parameters
        ----------
        h : TimeSeries or FrequencySeries
            The waveform to apply the window to.
        break_time : float, optional
            The number of seconds relative to the coalescence time at which to
            break the time series. Default is 0.
        params : dict, optional
            Specify the parameters of the waveform. Needed if either left or
            right `taper_freqfunc` is not None.
        ifo : str, optional
            If the waveform will be (over-)whitened before tapering, and the
            psds attribute is a dictionary, the ifo of the detector to use.
            See `TimeDomainWindow.apply_window` for details.
        copy : bool, optional
            Whether to copy the data before applying the window/whitening. If
            False, the taper will be applied in place. Default is True.
        """
        if isinstance(h, TimeSeries):
            return self._apply_to_timeseries(h, break_time=break_time,
                                             params=params, ifo=ifo, copy=copy)
        elif isinstance(h, FrequencySeries):
            return self._apply_to_frequencyseries(h, break_time=break_time,
                                                  params=params, ifo=ifo,
                                                  copy=copy)
        else:
            raise TypeError("h must be either TimeSeries or FrequencySeries")
    
    __call__ = apply_window


__all__ = ['WaveformTDWindow']
