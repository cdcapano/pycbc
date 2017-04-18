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

from pycbc.waveform import NoWaveformError
from pycbc.window import TimeDomainWindow, WindowBoundsError
from pycbc import pnutils
from pycbc.types import TimeSeries

class WaveformTDWindow(TimeDomainWindow):
    """Windows waveforms at specified times and/or frequencies.

    When initialized, a taper time, frequency, or frequency function may
    be specified for each side of the window. These are used to determine
    where to apply the left and right tapers when `apply_window` is called.
    If a frequency is provided for a taper, the time to apply the taper
    is estimated using the stationary phase approximation. If a frequency
    function is provided, a frequency to apply the taper for the parameters
    of a waveform is calculated, then from that a time estimated.

    Unlike `TimeDomainWindow`, times for applying the left and right taper
    are measured from the coalescence time of the waveform. Times before
    the coalescence time are negative; times after are positive.

    Instances of this class may be called like a function, in which case
    `apply_window` is called. See that function for more details.

    Parameters
    ----------
    left_taper_time : float, optional
        The time in seconds relative to the coalescence time at which to
        start the left taper. If provided, the same time will be used for
        all waveforms.
    right_taper_time : float, optional
        The time in seconds relative to the coalescence time at which to
        end the right taper. If provided, the same time will be used for
        all waveforms.
    left_taper_frequency : float, optional
        The frequency, in Hz, of the waveform at which time to start the
        left taper. To convert from frequency to time, the stationary phase
        approximation is used. If provided, the same frequency will be
        used for all waveforms.
    right_taper_frequency : float, optional
        Same as `left_taper_frequency`, but for the right side.
    left_taper_freqfunc : str, optional
        Specify a function name to use to compute a frequency at which to
        apply the left taper. Can be any string recognized by
        `pnutils.named_frequency_cutoffs`. If set, the waveform's
        parameters need to be provided when calling `apply_window`.
    right_taper_freqfunc : str, optional
        Same as `left_taper_freqfunc`, but for the right side.
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
                 left_taper_freqfunc=None, right_taper_freqfunc=None,
                 **kwargs):
        # initialize the window
        super(WaveformTDWindow, self).__init__(**kwargs)
        # add the location settings
        if left_taper == 'lal' and (left_taper_time is not None or
                                    left_taper_frequency is not None or
                                    left_taper_freqfunc is not None):
            raise ValueError("The lal taper function does not take a "
                             "start time or frequency")
        elif left_taper is not None and (left_taper_time is None and
                                         left_taper_frequency is None and
                                         left_taper_freqfunc is None):
            raise ValueError("Non-lal taper functions require either a taper "
                             "time, taper frequency, or frequency function")
        self.left_taper_time = left_taper_time
        self.left_taper_frequency = left_taper_frequency
        self.left_taper_freqfunc = left_taper_freqfunc
        if right_taper == 'lal':
            if right_taper_duration is not None:
                raise ValueError("The lal taper function does not take a "
                                 "duration")
            if left_taper_time is not None or \
                    left_taper_frequency is not None or \
                    left_taper_freqfunc is not None:
                raise ValueError("The lal taper function does not take a "
                                 "end time or frequency")
        elif right_taper is not None and right_taper_duration is None:
            raise ValueError("Non-lal taper functions require a duration")
        elif right_taper is not None and (right_taper_time is None and
                                          right_taper_frequency is None and
                                          right_taper_freqfunc is None):
            raise ValueError("Non-lal taper functions require either a taper "
                             "time, taper frequency, or frequency function")
        self.right_taper = right_taper
        self.right_taper_duration = right_taper_duration
        self.right_taper_time = right_taper_time
        self.right_taper_frequency = right_taper_frequency
        self.right_taper_freqfunc = right_taper_freqfunc

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
        # figure out times
        left_time = self.left_taper_time
        left_freq = self.left_taper_frequency
        if self.left_taper_freqfunc is not None:
            if params is None:
                raise ValueError("must provide waveform parameters for the "
                                 "frequency function to use for the left")
            left_freq = pnutils.named_frequency_cutoffs[
                self.left_taper_freqfunc](params)
        convert_to_ts = False
        #
        #   left taper
        #
        if left_freq is not None:
            # need frequencyseries version to get f(t)
            if isinstance(h, TimeSeries):
                h = h.to_frequencyseries()
                convert_to_ts = True
            # we only need to analyze h right around the desired frequency
            indx = int(left_freq / h.delta_f)
            t = time_from_frequencyseries(h[indx:indx+2])[0]
            if left_time is None:
                left_time = t
            else:
                left_time = max(t, left_time)
        if left_time is not None:
            # TimeDomainWindow measures time from the start of the segment,
            # so adjust
            if isinstance(h, TimeSeries):
                dur = len(h)*h.delta_t
            else:
                dur = 1./h.delta_f
            left_time = dur + left_time + break_time
        #
        #   right taper
        #
        right_time = self.right_taper_time
        right_freq = self.right_taper_frequency
        if self.right_taper_freqfunc is not None:
            if params is None:
                raise ValueError("must provide waveform parameters for the "
                                 "frequency function to use for the right")
            right_freq = pnutils.named_frequency_cutoffs[
                self.right_taper_freqfunc](params)
        if right_freq is not None:
            # need frequencyseries version to get f(t)
            if isinstance(h, TimeSeries):
                h = h.to_frequencyseries()
                convert_to_ts = True
            # we only need to analyze h right around the desired frequency
            indx = int(right_freq / h.delta_f)
            t = time_from_frequencyseries(h[indx:indx+2])[0]
            if right_time is None:
                right_time = t
            else:
                right_time = min(t, right_time)
        if right_time is not None:
            # TimeDomainWindow measures time from the start of the segment,
            # so adjust
            if isinstance(h, TimeSeries):
                dur = len(h)*h.delta_t
            else:
                dur = 1./h.delta_f
            right_time = dur + left_time + break_time
        try:
            h = super(WaveformTDWindow, self).apply_window(h,
                      left_time=left_time, right_time=right_time,
                      break_time=break_time, ifo=ifo, copy=copy)
        except WindowBoundsError as e:
            raise NoWaveformError(e)
        if convert_to_ts:
            h = h.to_timeseries()
        return h


__all__ = ['WaveformTDWindow']
