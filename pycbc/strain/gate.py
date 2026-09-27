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
import scipy.fft
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

class ToeplitzInverse(object):
    """Applies the inverse of a symmetric, positive-definite Toeplitz matrix.

    This uses the Gohberg-Semencul formula, which writes the inverse of an
    ``n x n`` symmetric Toeplitz matrix :math:`T` in terms of the first column
    :math:`x = T^{-1} e_0` of the inverse:

    .. math::

        T^{-1} = \\frac{1}{x_0}\\left[L(x)L(x)^T - L(ZJx)L(ZJx)^T\\right],

    where :math:`L(v)` is the lower-triangular Toeplitz matrix with first
    column :math:`v`, :math:`J` reverses the order of a vector, and :math:`Z`
    shifts it down by one element. Since products with triangular Toeplitz
    matrices are convolutions, :math:`T^{-1}` can be applied with FFTs in
    :math:`O(n \\log n)` operations, rather than the :math:`O(n^2)` needed
    for a product with the explicit inverse. Only the first column needs to
    be stored. It is obtained with the Levinson recursion (which takes
    :math:`O(n^2)` operations and :math:`O(n)` memory), followed by a step
    of iterative refinement, which brings the residual down to round-off.

    Parameters
    ----------
    col : array
        The first column of the Toeplitz matrix.
    refinement_steps : int, optional
        The number of steps of iterative refinement of the first column of
        the inverse. Default is 1, which is enough to bring the residual down
        to round-off.
    first_column : array, optional
        An estimate of the first column of the inverse, e.g., from
        :py:func:`levinson_first_columns`. If provided, it is used as the
        starting point for the iterative refinement rather than solving for
        it with the Levinson recursion.
    """
    def __init__(self, col, refinement_steps=1, first_column=None):
        col = numpy.asarray(col).real
        n = len(col)
        self.n = n
        # zero padding to at least 2n - 1 makes the circular convolutions
        # linear
        self.nfft = scipy.fft.next_fast_len(2*n - 1, real=True)
        e0 = numpy.zeros(n)
        e0[0] = 1.
        if first_column is None:
            x = linalg.solve_toeplitz(col, e0)
        else:
            x = numpy.asarray(first_column, dtype=float)
            if len(x) != n:
                raise ValueError('first_column must have the same length as '
                                 'col')
        self._set_column(x)
        # iterative refinement; T is applied as a circular convolution with
        # the symmetric extension of its first column
        fcol = scipy.fft.rfft(numpy.concatenate(
            [col, numpy.zeros(self.nfft - 2*n + 1), col[:0:-1]]))
        for _ in range(refinement_steps):
            resid = e0 - scipy.fft.irfft(fcol * scipy.fft.rfft(x, self.nfft),
                                         self.nfft)[:n]
            x = x + self.apply(resid)
            self._set_column(x)

    def _set_column(self, x):
        """Sets the first column of the inverse."""
        self.x0 = x[0]
        zjx = numpy.zeros(self.n)
        zjx[1:] = x[:0:-1]
        self._fx = scipy.fft.rfft(x, self.nfft)
        self._fzjx = scipy.fft.rfft(zjx, self.nfft)

    def apply(self, b):
        """Returns :math:`T^{-1} b` for every row :math:`b` of ``b``.

        Parameters
        ----------
        b : array
            A ``(number of vectors) x n`` array, or a single vector of length
            ``n``.

        Returns
        -------
        numpy.ndarray :
            Array with the same shape as ``b``.
        """
        n = self.n
        nfft = self.nfft
        # L(v)^T y is the reverse of L(v) applied to the reversed y
        fb = scipy.fft.rfft(b[..., ::-1], nfft, axis=-1)
        r1 = scipy.fft.irfft(fb * self._fx, nfft, axis=-1)[..., n-1::-1]
        r2 = scipy.fft.irfft(fb * self._fzjx, nfft, axis=-1)[..., n-1::-1]
        out = scipy.fft.irfft(scipy.fft.rfft(r1, nfft, axis=-1) * self._fx
                              - scipy.fft.rfft(r2, nfft, axis=-1) * self._fzjx,
                              nfft, axis=-1)[..., :n]
        out /= self.x0
        return out


def levinson_first_columns(col, sizes):
    """Returns the first column of the inverse of several leading blocks of a
    symmetric, positive-definite Toeplitz matrix.

    The ``k x k`` leading block of a Toeplitz matrix is the Toeplitz matrix
    with first column ``col[:k]``. The Levinson-Durbin recursion finds the
    first column :math:`f_k` of the inverse of each block from that of the
    previous one:

    .. math::

        f_{k+1} = \\frac{1}{1 - \\epsilon_k^2}\\left(
            \\begin{bmatrix} f_k \\\\ 0 \\end{bmatrix} - \\epsilon_k
            \\begin{bmatrix} 0 \\\\ J f_k \\end{bmatrix}\\right),
        \\quad \\epsilon_k = \\sum_{i=0}^{k-1} c_{k-i} f_k[i],

    where :math:`J` reverses the order of a vector. The columns for all of
    the requested sizes are therefore obtained in a single pass, which takes
    :math:`O(n^2)` operations for the largest size :math:`n`; this is the
    same cost as solving for the largest size alone.

    Parameters
    ----------
    col : array
        The first column of the Toeplitz matrix. Must be at least as long as
        the largest size.
    sizes : iterable of int
        The sizes of the leading blocks to get the inverses of.

    Returns
    -------
    dict :
        Dictionary of size -> first column of the inverse of the leading
        block of that size.
    """
    col = numpy.asarray(col).real
    sizes = set(int(k) for k in sizes)
    nmax = max(sizes)
    if nmax > len(col):
        raise ValueError('col is shorter than the largest size')
    out = {}
    # the recursion alternates between two buffers to avoid allocating new
    # arrays at every step
    fbuf = numpy.zeros(nmax)
    gbuf = numpy.zeros(nmax)
    fbuf[0] = 1. / col[0]
    if 1 in sizes:
        out[1] = fbuf[:1].copy()
    for k in range(1, nmax):
        f = fbuf[:k]
        eps = numpy.dot(col[k:0:-1], f)
        g = gbuf[:k+1]
        g[:k] = f
        g[k] = 0.
        g[1:] -= eps * f[::-1]
        g /= 1. - eps*eps
        fbuf, gbuf = gbuf, fbuf
        if k+1 in sizes:
            out[k+1] = fbuf[:k+1].copy()
    return out


def toeplitz_inverses(invpsd, sizes, refinement_steps=1):
    """Returns :py:class:`ToeplitzInverse` of the covariance matrix for
    several gate sizes.

    This gives the same result as calling :py:func:`toeplitz_inverse` for
    each size, but finds all of the inverses with a single pass of the
    Levinson-Durbin recursion (see :py:func:`levinson_first_columns`), which
    is much faster when there are many sizes.

    Parameters
    ----------
    invpsd : FrequencySeries
        The inverse of the PSD.
    sizes : iterable of int
        The gate sizes (``rindex - lindex``) to get the inverses for.
    refinement_steps : int, optional
        The number of steps of iterative refinement to apply to each
        inverse. Default is 1.

    Returns
    -------
    dict :
        Dictionary of size -> :py:class:`ToeplitzInverse`.
    """
    tdfilter = invpsd.astype('complex').to_timeseries() * invpsd.delta_t
    col = tdfilter.numpy().real
    columns = levinson_first_columns(col, sizes)
    return {k: ToeplitzInverse(col[:k], refinement_steps=refinement_steps,
                               first_column=x)
            for k, x in columns.items()}


def toeplitz_inverse(invpsd, lindex, rindex):
    """Returns a :py:class:`ToeplitzInverse` of the covariance matrix.

    This is the same (uninverted) covariance matrix that is explicitly
    inverted by :py:func:`invert_covariance`.

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
    ToeplitzInverse :
        Object that applies the inverse of the matrix.
    """
    tdfilter = invpsd.astype('complex').to_timeseries() * invpsd.delta_t
    return ToeplitzInverse(tdfilter[:(rindex-lindex)].numpy())


def gate_and_paint_gs(data, lindex, rindex, invpsd, invmat=None, copy=True):
    """Gates and in-paints data using the Gohberg-Semencul formula.

    This gives the same result as :py:func:`gate_and_paint_matmul`, but
    uses a :py:class:`ToeplitzInverse` rather than the explicit inverse.

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
    invmat : ToeplitzInverse, optional
        The inverse to use. If None, calculate on function call.
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
    if invmat is None:
        invmat = toeplitz_inverse(invpsd, lindex, rindex)
    # remove the projection into the null space
    proj = invmat.apply(owhgated_data.numpy()[lindex:rindex])
    data[lindex:rindex] -= proj
    return data


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
    paint_method : {'toeplitz', 'matmul', 'gs'}
        Which method to use for in-painting the gated region. The ``'gs'``
        method applies the inverse of the covariance matrix with the
        Gohberg-Semencul formula; see :py:class:`ToeplitzInverse`.
    invmat : array or ToeplitzInverse, optional
        The inverted covariance matrix to use if ``paint_method='matmul'``,
        or the :py:class:`ToeplitzInverse` to use if ``paint_method='gs'``.
        If None, it will be calculated from ``invpsd``.

    Returns
    -------
    list of FrequencySeries :
        The gated and in-painted series, in the same order as ``htildes``.
    """
    h0 = htildes[0]
    fdata = batch_gate_and_paint_fd_array(
        numpy.array([h.numpy() for h in htildes]), float(h0.delta_f),
        float(h0.start_time), time, window, invpsd,
        paint_method=paint_method, invmat=invmat)
    return [FrequencySeries(fd, delta_f=h.delta_f, epoch=h.epoch, copy=False)
            for fd, h in zip(fdata, htildes)]


def batch_gate_and_paint_fd_array(fdata, delta_f, start_time, time, window,
                                  invpsd, paint_method='toeplitz',
                                  invmat=None):
    """Gates and in-paints several frequency series stored in a 2D array.

    This is the same as :py:func:`batch_gate_and_paint_fd`, but takes and
    returns the frequency series as a ``(number of series) x (number of
    frequencies)`` array.

    Parameters
    ----------
    fdata : numpy.ndarray
        The frequency series to gate, one per row. This is not modified.
    delta_f : float
        The frequency spacing of the series.
    start_time : float
        The start time (epoch) of the series.
    time : float
        Central time of the gate in seconds.
    window : float
        Half-length in seconds of the gate.
    invpsd : FrequencySeries
        The inverse of the PSD. Must be the same length as the series.
    paint_method : {'toeplitz', 'matmul', 'gs'}
        Which method to use for in-painting the gated region. The ``'gs'``
        method applies the inverse of the covariance matrix with the
        Gohberg-Semencul formula; see :py:class:`ToeplitzInverse`.
    invmat : array or ToeplitzInverse, optional
        The inverted covariance matrix to use if ``paint_method='matmul'``,
        or the :py:class:`ToeplitzInverse` to use if ``paint_method='gs'``.
        If None, it will be calculated from ``invpsd``.

    Returns
    -------
    numpy.ndarray :
        The gated and in-painted series, in the same order as ``fdata``.
    """
    nfreq = fdata.shape[1]
    tlen = 2 * (nfreq - 1)
    delta_t = 1. / (tlen * delta_f)
    # same as TimeSeries.get_gate_indices
    lindex = max(int((time - window - start_time) / delta_t), 0)
    rindex = min(int((time + window - start_time) / delta_t), tlen)
    # time shift so that the end of the gate lands on a sample, as is done
    # in TimeSeries.gate
    offset = start_time + rindex * delta_t - (time + window)
    # the time shift is applied to the over-whitening filter and to the
    # correction, rather than to the series, since the output is the
    # series minus the (unshifted) correction
    if offset != 0:
        shift = numpy.exp(-2j * numpy.pi * offset
                          * (numpy.arange(nfreq) * delta_f))
        owhfilter = shift * invpsd.numpy()
    else:
        shift = None
        owhfilter = invpsd.numpy()
    # Gating and in-painting a series v gives v' = z - E T^{-1} (K z)_g,
    # where z is v with the gated samples zeroed, K is the over-whitening
    # operator (a circulant matrix), (.)_g takes the samples in the gate,
    # E puts a vector of gate samples back into a full-length series, and
    # T = K_gg is the (Toeplitz) matrix that is inverted by the in-painting
    # methods. Since (K z)_g = (K v)_g - T v_g, this simplifies to
    # v' = v - E T^{-1} (K v)_g. This only needs the over-whitened
    # (ungated) series in the gate, and an FFT of the correction, rather
    # than the FFTs to the time domain and back of the zeroed series.
    # Note that tlen * delta_f * delta_t = 1 converts between numpy's and
    # pycbc's FFT normalizations.
    owh = numpy.fft.irfft(fdata * owhfilter, n=tlen, axis=1)
    owh = owh[:, lindex:rindex] * (tlen * delta_f)
    # apply T^{-1} to get the correction in the gate as a
    # (number of series) x (gate length) array
    if paint_method == 'toeplitz':
        tdfilter = invpsd.astype('complex').to_timeseries() * invpsd.delta_t
        corr = linalg.solve_toeplitz(tdfilter[:(rindex - lindex)].numpy(),
                                     owh.T).T
    elif paint_method == 'matmul':
        if invmat is None:
            invmat = invert_covariance(invpsd, lindex, rindex)
        # BLAS is several times faster multiplying the (thin) gated series by
        # a C-ordered matrix than by a Fortran-ordered one; the matrix
        # returned by invert_covariance is Fortran ordered, so its transpose
        # is C ordered
        if invmat.flags.c_contiguous:
            corr = (invmat @ owh.T).T
        else:
            corr = owh @ invmat.T
    elif paint_method == 'gs':
        if invmat is None:
            invmat = toeplitz_inverse(invpsd, lindex, rindex)
        corr = invmat.apply(owh)
    else:
        raise ValueError(f'Unrecognized paint_method input {paint_method}')
    # subtract the correction in the frequency domain
    tcorr = numpy.zeros((fdata.shape[0], tlen))
    tcorr[:, lindex:rindex] = corr
    fcorr = numpy.fft.rfft(tcorr, axis=1)
    fcorr *= delta_t
    if shift is not None:
        fcorr *= shift.conj()
        # the Nyquist term of the shifted series is made real before the
        # correction is subtracted (since the series is real in the time
        # domain); this is the same as what is done by TimeSeries.gate
        nyq = fdata[:, -1] * shift[-1]
        nyq = (nyq.real - fcorr[:, -1] * shift[-1]) * shift[-1].conj()
    out = fdata - fcorr
    # TimeSeries.gate returns a time series, which is real; this makes the
    # DC and Nyquist bins real
    out[:, 0] = out[:, 0].real
    if shift is not None:
        out[:, -1] = nyq.real
    else:
        out[:, -1] = out[:, -1].real
    return out
