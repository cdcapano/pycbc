# Copyright (C) 2018 Miriam Cabero, Collin Capano
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

import logging
import re
import numpy

import pycbc
from pycbc import conversions, boundaries

from . import uniform, bounded

logger = logging.getLogger('pycbc.distributions.qnm')


class UniformF0Tau(uniform.Uniform):
    """A distribution uniform in QNM frequency and damping time.

    Constraints may be placed to exclude frequencies and damping times
    corresponding to specific masses and spins.

    To ensure a properly normalized pdf that accounts for the constraints
    on final mass and spin, a renormalization factor is calculated upon
    initialization. This is calculated numerically: f0 and tau are drawn
    randomly, then the norm is scaled by the fraction of points that yield
    final masses and spins within the constraints. The `norm_tolerance` keyword
    arguments sets the error on the estimate of the norm from this numerical
    method. If this value is too large, such that no points are found in
    the allowed region, a ValueError is raised.

    Parameters
    ----------
    f0 : tuple or boundaries.Bounds
        The range of QNM frequencies (in Hz).
    tau : tuple or boundaries.Bounds
        The range of QNM damping times (in s).
    final_mass : tuple or boundaries.Bounds, optional
        The range of final masses to allow. Default is [0,inf).
    final_spin : tuple or boundaries.Bounds, optional
        The range final spins to allow. Must be in [-0.996, 0.996], which is
        the default.
    rdfreq : str, optional
        Use the given string as the name for the f0 parameter. Default is 'f0'.
    damping_time : str, optional
        Use the given string as the name for the tau parameter. Default is
        'tau'.
    norm_tolerance : float, optional
        The tolerance on the estimate of the normalization. Default is 1e-3.
    norm_seed : int, optional
        Seed to use for the random number generator when estimating the norm.
        Default is 0. After the norm is estimated, the random number generator
        is set back to the state it was in upon initialization.

    Examples
    --------

    Create a distribution:

    >>> dist = UniformF0Tau(f0=(10., 2048.), tau=(1e-4,1e-2))

    Check that all random samples drawn from the distribution yield final
    masses > 1:

    >>> from pycbc import conversions
    >>> samples = dist.rvs(size=1000)
    >>> (conversions.final_mass_from_f0_tau(samples['f0'],
            samples['tau']) > 1.).all()
    True

    Create a distribution with tighter bounds on final mass and spin:

    >>> dist = UniformF0Tau(f0=(10., 2048.), tau=(1e-4,1e-2),
            final_mass=(20., 200.), final_spin=(0,0.996))

    Check that all random samples drawn from the distribution are in the
    final mass and spin constraints:

    >>> samples = dist.rvs(size=1000)
    >>> (conversions.final_mass_from_f0_tau(samples['f0'],
            samples['tau']) >= 20.).all()
    True
    >>> (conversions.final_mass_from_f0_tau(samples['f0'],
            samples['tau']) < 200.).all()
    True
    >>> (conversions.final_spin_from_f0_tau(samples['f0'],
            samples['tau']) >= 0.).all()
    True
    >>> (conversions.final_spin_from_f0_tau(samples['f0'],
            samples['tau']) < 0.996).all()
    True

    """

    name = 'uniform_f0_tau'

    def __init__(self, f0=None, tau=None, final_mass=None, final_spin=None,
                 rdfreq='f0', damping_time='tau', norm_tolerance=1e-3,
                 norm_seed=0):
        if f0 is None:
            raise ValueError("must provide a range for f0")
        if tau is None:
            raise ValueError("must provide a range for tau")
        self.rdfreq = rdfreq
        self.damping_time = damping_time
        parent_args = {rdfreq: f0, damping_time: tau}
        super(UniformF0Tau, self).__init__(**parent_args)
        if final_mass is None:
            final_mass = (0., numpy.inf)
        if final_spin is None:
            final_spin = (-0.996, 0.996)
        self.final_mass_bounds = boundaries.Bounds(
            min_bound=final_mass[0], max_bound=final_mass[1])
        self.final_spin_bounds = boundaries.Bounds(
            min_bound=final_spin[0], max_bound=final_spin[1])
        # Re-normalize to account for cuts: we'll do this by just sampling
        # a large number of spaces f0 taus, and seeing how many are in the
        # desired range.
        # perseve the current random state
        s = numpy.random.get_state()
        numpy.random.seed(norm_seed)
        nsamples = int(1./norm_tolerance**2)
        draws = super(UniformF0Tau, self).rvs(size=nsamples)
        # reset the random state
        numpy.random.set_state(s)
        num_in = self._constraints(draws).sum()
        # if num_in is 0, than the requested tolerance is too large
        if num_in == 0:
            raise ValueError("the normalization is < then the norm_tolerance; "
                             "try again with a smaller nrom_tolerance")
        self._lognorm += numpy.log(num_in) - numpy.log(nsamples)
        self._norm = numpy.exp(self._lognorm)

    def contains(self, params):
        isin = super(UniformF0Tau, self).contains(params)
        if not getattr(isin, 'ndim', 0):
            # the mass and spin conversion costs 6.5 us; skip it for a single
            # point the bounds already reject
            return isin and self._constraints(params)
        return isin & self._constraints(params)

    def _constraints(self, params):
        f0 = params[self.rdfreq]
        tau = params[self.damping_time]
        # check if we need to specify a particular mode (l,m) != (2,2)
        if re.match(r'f_\d{3}', self.rdfreq):
            mode = self.rdfreq.strip('f_')
            l, m = int(mode[0]), int(mode[1])
        else:
            l, m = 2, 2
        # temporarily silence invalid warnings... these will just be ruled out
        # automatically
        with numpy.errstate(invalid="ignore"):
            mf = conversions.final_mass_from_f0_tau(f0, tau, l=l, m=m)
            sf = conversions.final_spin_from_f0_tau(f0, tau, l=l, m=m)
            isin = (self.final_mass_bounds.contains(mf)
                    & self.final_spin_bounds.contains(sf))
        return isin

    def rvs(self, size=1):
        """Draw random samples from this distribution.

        Parameters
        ----------
        size : int, optional
            The number of draws to do. Default is 1.

        Returns
        -------
        array
            A structured array of the random draws.
        """
        size = int(size)
        dtype = [(p, float) for p in self.params]
        arr = numpy.zeros(size, dtype=dtype)
        remaining = size
        keepidx = 0
        while remaining:
            draws = super(UniformF0Tau, self).rvs(size=remaining)
            mask = self._constraints(draws)
            addpts = mask.sum()
            arr[keepidx:keepidx+addpts] = draws[mask]
            keepidx += addpts
            remaining = size - keepidx
        return arr

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Initialize this class from a config file.

        Bounds on ``f0``, ``tau``, ``final_mass`` and ``final_spin`` should
        be specified by providing ``min-{param}`` and ``max-{param}``. If
        the ``f0`` or ``tau`` param should be renamed, ``rdfreq`` and
        ``damping_time`` should be provided; these must match
        ``variable_args``. If ``rdfreq`` and ``damping_time`` are not
        provided, ``variable_args`` are expected to be ``f0`` and ``tau``.

        Only ``min/max-f0`` and ``min/max-tau`` need to be provided.

        Example:

        .. code-block:: ini

            [{section}-f0+tau]
            name = uniform_f0_tau
            min-f0 = 10
            max-f0 = 2048
            min-tau = 0.0001
            max-tau = 0.010
            min-final_mass = 10

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            WorkflowConfigParser instance to read.
        section : str
            The name of the section to read.
        variable_args : str
            The name of the variable args. These should be separated by
            ``pycbc.VARARGS_DELIM``.

        Returns
        -------
        UniformF0Tau :
            This class initialized with the parameters provided in the config
            file.
        """
        tag = variable_args
        variable_args = set(variable_args.split(pycbc.VARARGS_DELIM))
        # get f0 and tau
        f0 = bounded.get_param_bounds_from_config(cp, section, tag, 'f0')
        tau = bounded.get_param_bounds_from_config(cp, section, tag, 'tau')
        # see if f0 and tau should be renamed
        if cp.has_option_tag(section, 'rdfreq', tag):
            rdfreq = cp.get_opt_tag(section, 'rdfreq', tag)
        else:
            rdfreq = 'f0'
        if cp.has_option_tag(section, 'damping_time', tag):
            damping_time = cp.get_opt_tag(section, 'damping_time', tag)
        else:
            damping_time = 'tau'
        # check that they match whats in the variable args
        if not variable_args == set([rdfreq, damping_time]):
            raise ValueError("variable args do not match rdfreq and "
                             "damping_time names")
        # get the final mass and spin values, if provided
        final_mass = bounded.get_param_bounds_from_config(
            cp, section, tag, 'final_mass')
        final_spin = bounded.get_param_bounds_from_config(
            cp, section, tag, 'final_spin')
        extra_opts = {}
        if cp.has_option_tag(section, 'norm_tolerance', tag):
            extra_opts['norm_tolerance'] = float(
                cp.get_opt_tag(section, 'norm_tolerance', tag))
        if cp.has_option_tag(section, 'norm_seed', tag):
            extra_opts['norm_seed'] = int(
                cp.get_opt_tag(section, 'norm_seed', tag))
        return cls(f0=f0, tau=tau,
                   final_mass=final_mass, final_spin=final_spin,
                   rdfreq=rdfreq, damping_time=damping_time,
                   **extra_opts)


class BayesWaveSignalSNR(bounded.BoundedDist):
    r"""The prior on the SNR of a signal wavelet used by BayesWave.

    This is the prior that BayesWave puts on the signal-to-noise ratio
    :math:`\rho` of each wavelet in its signal model (Eq. 14 of Cornish et
    al. 2021, `arXiv:2011.09494 <https://arxiv.org/abs/2011.09494>`_):

    .. math::

        p(\rho) = \frac{3 \rho}{4 \rho_*^2 (1 + \rho / (4 \rho_*))^5}.

    It peaks at :math:`\rho = \rho_*` and falls off as :math:`\rho^{-4}` at
    large :math:`\rho`, which disfavors both very weak and very loud
    components. With :math:`u = \rho / (4 \rho_*)` the pdf is
    :math:`12 u (1 + u)^{-5} du`, so it is normalized on
    :math:`[0, \infty)`, and its cumulative distribution is

    .. math::

        F(\rho) = 1 - \frac{1 + 4u}{(1 + u)^4}.

    If the bounds of a parameter are narrower than :math:`[0, \infty)`, the
    distribution is truncated to them and renormalized. The inverse of the
    cumulative distribution, which is used to draw samples and by samplers
    that sample the unit cube, is found by bisection.

    Parameters
    ----------
    rho_star : float, optional
        The SNR at which the pdf peaks, :math:`\rho_*`. Default is 5, the
        default in BayesWave.
    \**params :
        The keyword arguments should provide the names of the parameters and
        their bounds, as either tuples or ``boundaries.Bounds`` instances.
        The bounds must be within :math:`[0, \infty)`. Parameters with bounds
        of None are bounded by :math:`[0, \infty)`.

    Examples
    --------
    Create the prior for an SNR between 1 and 50, and evaluate its pdf:

    >>> from pycbc import distributions
    >>> dist = distributions.BayesWaveSignalSNR(rho_star=5., snr=(1., 50.))
    >>> dist.pdf(snr=5.)
    0.05378...
    """
    name = 'bayeswave_signal_snr'

    def __init__(self, rho_star=5., **params):
        for param, bnds in params.items():
            if bnds is None:
                params[param] = (0., numpy.inf)
        super().__init__(**params)
        self.rho_star = float(rho_star)
        if not self.rho_star > 0:
            raise ValueError("rho_star must be positive")
        self._sfbounds = {}
        self._cdfbounds = {}
        self._lognorm = 0.
        for param in self._params:
            lower, upper = self._bounds[param]
            if lower < 0:
                raise ValueError("the bounds of {} must be >= 0, got [{}, {}]"
                                 .format(param, lower, upper))
            sfbounds = (self._sf(lower), self._sf(upper))
            self._sfbounds[param] = sfbounds
            self._cdfbounds[param] = (self._cdf(lower), self._cdf(upper))
            # the probability within the bounds
            self._lognorm -= numpy.log(sfbounds[0] - sfbounds[1])
        self._norm = numpy.exp(self._lognorm)

    @property
    def norm(self):
        """float: The normalization of the multi-dimensional pdf."""
        return self._norm

    @property
    def lognorm(self):
        """float: The log of the normalization."""
        return self._lognorm

    def _sf(self, rho):
        """The survival function, 1 - F(rho), of the untruncated
        distribution."""
        u = numpy.asarray(rho, dtype=float) / (4. * self.rho_star)
        with numpy.errstate(invalid='ignore', over='ignore'):
            sf = (1. + 4. * u) / (1. + u)**4
        return numpy.where(numpy.isinf(u), 0., sf)

    def _cdf(self, rho):
        """The cdf, F(rho), of the untruncated distribution. This is written
        as u^2 (6 + 4u + u^2) / (1 + u)^4, which equals 1 - sf without the
        loss of precision at small rho."""
        u = numpy.asarray(rho, dtype=float) / (4. * self.rho_star)
        with numpy.errstate(invalid='ignore', over='ignore'):
            cdf = u * u * (6. + u * (4. + u)) / (1. + u)**4
        return numpy.where(numpy.isinf(u), 1., cdf)

    def _logpdf_untruncated(self, rho):
        """The log of the pdf of the untruncated distribution."""
        with numpy.errstate(divide='ignore'):
            return (numpy.log(3. * rho / (4. * self.rho_star**2))
                    - 5. * numpy.log1p(rho / (4. * self.rho_star)))

    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        return numpy.exp(self._logpdf(**kwargs))

    def _logpdf(self, **kwargs):
        """Returns the log of the pdf at the given values. The keyword
        arguments must contain all of parameters in self's params.
        Unrecognized arguments are ignored. Only values within the bounds are
        passed in (see :py:meth:`BoundedDist.logpdf`), and they may be arrays.
        """
        for p in self._params:
            if p not in kwargs:
                raise ValueError(
                    'Missing parameter {} to construct pdf.'.format(p))
        return self._lognorm + sum(
            self._logpdf_untruncated(kwargs[p]) for p in self._params)

    def _cdfinv_param(self, param, value):
        """Return the inverse cdf, mapping the unit interval to the parameter
        bounds.
        """
        value = numpy.asarray(value, dtype=float)
        sflower, sfupper = self._sfbounds[param]
        cdflower, cdfupper = self._cdfbounds[param]
        # the value of the (untruncated) survival function and cdf to find;
        # the cdf is used where it is less than 1/2, and the survival
        # function elsewhere, so that the solution keeps its relative
        # precision at both ends
        target = sflower - value * (sflower - sfupper)
        target_cdf = cdflower + value * (cdfupper - cdflower)
        use_cdf = target_cdf < 0.5
        # bracket the solution in u = rho / (4 rho_*); the survival function
        # decreases monotonically, and is < 5 / u^3 for u >= 1
        lower, upper = self._bounds[param]
        ulo = numpy.full(target.shape, lower / (4. * self.rho_star))
        if numpy.isfinite(upper):
            uhi = numpy.full(target.shape, upper / (4. * self.rho_star))
        else:
            # a target of 0 (value = 1) is the end point, set below
            with numpy.errstate(divide='ignore'):
                uhi = numpy.where(target > 0,
                                  numpy.maximum(1., (5. / target)**(1. / 3.)),
                                  1.)
        # bisect
        for _ in range(100):
            umid = 0.5 * (ulo + uhi)
            below = numpy.where(
                use_cdf,
                umid * umid * (6. + umid * (4. + umid)) / (1. + umid)**4
                < target_cdf,
                (1. + 4. * umid) / (1. + umid)**4 > target)
            ulo = numpy.where(below, umid, ulo)
            uhi = numpy.where(below, uhi, umid)
        rho = 2. * self.rho_star * (ulo + uhi)
        # the end points
        rho = numpy.where(value <= 0., lower, rho)
        rho = numpy.where(value >= 1., upper, rho)
        if rho.ndim == 0:
            rho = float(rho)
        return rho

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file. The
        parameters for the distribution are retrieved from the section titled
        "[`section`-`variable_args`]" in the config file. The bounds are given
        with ``min-{param}`` and ``max-{param}`` (by default, the SNRs are
        bounded by [0, inf)), and ``rho_star`` sets :math:`\\rho_*` (default
        5).

        Example:

        .. code-block:: ini

            [{section}-amp220_snr]
            name = bayeswave_signal_snr
            rho_star = 5
            min-amp220_snr = 1
            max-amp220_snr = 50

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            ``pycbc.VARARGS_DELIM``. These must appear in the "tag" part of
            the section header.

        Returns
        -------
        BayesWaveSignalSNR
            A distribution instance.
        """
        return bounded.bounded_from_config(cls, cp, section, variable_args,
                                           bounds_required=False)
