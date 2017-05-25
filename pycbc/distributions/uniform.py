# Copyright (C) 2016  Collin Capano
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
"""
This modules provides classes for evaluating uniform distributions.
"""

import numpy
from pycbc.distributions import bounded
from pycbc.distributions.boundaries import Bounds
from pycbc import conversions

class Uniform(bounded.BoundedDist):
    """
    A uniform distribution on the given parameters. The parameters are
    independent of each other. Instances of this class can be called like
    a function. By default, logpdf will be called, but this can be changed
    by setting the class's __call__ method to its pdf method.

    Parameters
    ----------
    \**params :
        The keyword arguments should provide the names of parameters and their
        corresponding bounds, as either tuples or a `boundaries.Bounds`
        instance.

    Attributes
    ----------
    name : 'uniform'
        The name of this distribution.

    Attributes
    ----------
    params : list of strings
        The list of parameter names.
    bounds : dict
        A dictionary of the parameter names and their bounds.
    norm : float
        The normalization of the multi-dimensional pdf.
    lognorm : float
        The log of the normalization.

    Examples
    --------
    Create a 2 dimensional uniform distribution:

    >>> dist = prior.Uniform(mass1=(10.,50.), mass2=(10.,50.))

    Get the log of the pdf at a particular value:

    >>> dist.logpdf(mass1=25., mass2=10.)
        -7.3777589082278725

    Do the same by calling the distribution:

    >>> dist(mass1=25., mass2=10.)
        -7.3777589082278725

    Generate some random values:

    >>> dist.rvs(size=3)
        array([(36.90885758394699, 51.294212757995254),
               (39.109058546060346, 13.36220145743631),
               (34.49594465315212, 47.531953033719454)], 
              dtype=[('mass1', '<f8'), ('mass2', '<f8')])
    
    Initialize a uniform distribution using a boundaries.Bounds instance,
    with cyclic bounds:

    >>> dist = distributions.Uniform(phi=Bounds(10, 50, cyclic=True))
    
    Apply boundary conditions to a value:

    >>> dist.apply_boundary_conditions(phi=60.)
        {'mass1': array(20.0)}
    
    The boundary conditions are applied to the value before evaluating the pdf;
    note that the following returns a non-zero pdf. If the bounds were not
    cyclic, the following would return 0:

    >>> dist.pdf(phi=60.)
        0.025
    """
    name = 'uniform'
    def __init__(self, **params):
        super(Uniform, self).__init__(**params)
        # compute the norm and save
        # temporarily suppress numpy divide by 0 warning
        numpy.seterr(divide='ignore')
        self._lognorm = -sum([numpy.log(abs(bnd[1]-bnd[0]))
                                    for bnd in self._bounds.values()])
        self._norm = numpy.exp(self._lognorm)
        numpy.seterr(divide='warn')

    @property
    def norm(self):
        return self._norm

    @property
    def lognorm(self):
        return self._lognorm

    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        if kwargs in self:
            return self._norm
        else:
            return 0.

    def _logpdf(self, **kwargs):
        """Returns the log of the pdf at the given values. The keyword
        arguments must contain all of parameters in self's params. Unrecognized
        arguments are ignored.
        """
        if kwargs in self:
            return self._lognorm
        else:
            return -numpy.inf


    def rvs(self, size=1, param=None):
        """Gives a set of random values drawn from this distribution.

        Parameters
        ----------
        size : {1, int}
            The number of values to generate; default is 1.
        param : {None, string}
            If provided, will just return values for the given parameter.
            Otherwise, returns random values for each parameter.

        Returns
        -------
        structured array
            The random values in a numpy structured array. If a param was
            specified, the array will only have an element corresponding to the
            given parameter. Otherwise, the array will have an element for each
            parameter in self's params.
        """
        if param is not None:
            dtype = [(param, float)]
        else:
            dtype = [(p, float) for p in self.params]
        arr = numpy.zeros(size, dtype=dtype)
        for (p,_) in dtype:
            arr[p] = numpy.random.uniform(self._bounds[p][0],
                                        self._bounds[p][1],
                                        size=size)
        return arr

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file. The parameters
        for the distribution are retrieved from the section titled
        "[`section`-`variable_args`]" in the config file.

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            `prior.VARARGS_DELIM`. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        Uniform
            A distribution instance from the pycbc.inference.prior module.
        """
        return super(Uniform, cls).from_config(cp, section, variable_args,
                     bounds_required=True)


class UniformMasses(Uniform):
    name = 'uniform_masses'
    def __init__(self, mass1=None, mass2=None, mchirp_bounds=None,
                 q_bounds=None, seed=None):
        super(UniformMasses, self).__init__(mass1=mass1, mass2=mass2)
        if mchirp_bounds is not None:
            self.mchirp_bounds = Bounds(mchirp_bounds[0], mchirp_bounds[1])
        else:
            self.mchirp_bounds = None
        if q_bounds is not None:
            self.q_bounds = Bounds(q_bounds[0], q_bounds[1])
        else:
            self.q_bounds = None
        rstate = numpy.random.get_state()
        # set the seed
        if seed is None:
            seed = 18293
        numpy.random.seed(seed)
        nsamples = int(1e6)
        rvals = super(UniformMasses, self).rvs(size=nsamples)
        # reset the random state back to what it was
        numpy.random.set_state(rstate)
        keep = self._apply_constraints(rvals)
        adjust = float(keep.sum())/nsamples
        self._norm /= adjust
        self._lognorm -= numpy.log(adjust)

    def __contains__(self, params):
        out = self._apply_constraints(params) & \
            super(UniformMasses, self).__contains__(params)
        if out.size == 1:
            out = out[0]
        return out

    def _apply_constraints(self, values):
        """Applies physical constraints to the given parameter values.

        Parameters
        ----------
        values : {arr or dict}
            A dictionary or structured array giving the values.

        Returns
        -------
        bool
            Whether or not the values satisfy physical
        """
        mass1 = conversions._ensurearray(values['mass1'])
        mass2 = conversions._ensurearray(values['mass2'])
        test = numpy.ones(mass1.size, dtype=bool)
        if self.mchirp_bounds is not None:
            try:
                mchirp = conversions._ensurearray(values['mchirp'])
            except:
                mchirp = conversions.mchirp_from_mass1_mass2(mass1, mass2)
            test &= self.mchirp_bounds.__contains__(mchirp)
        if self.q_bounds is not None:
            try:
                q = conversions._ensurearray(values['q'])
            except:
                q = conversions.q_from_mass1_mass2(mass1, mass2)
            test &= self.q_bounds.__contains__(q)
        return test

    def rvs(self, size=1, param=None):
        """Gives a set of random values drawn from this distribution.

        Parameters
        ----------
        size : {1, int}
            The number of values to generate; default is 1.
        param : {None, string}
            If provided, will just return values for the given parameter.
            Otherwise, returns random values for each parameter.

        Returns
        -------
        structured array
            The random values in a numpy structured array. If a param was
            specified, the array will only have an element corresponding to the
            given parameter. Otherwise, the array will have an element for each
            parameter in self's params.
        """
        size = int(size)
        dtype = [(p, float) for p in self.params]
        arr = numpy.zeros(size, dtype=dtype)
        remaining = size
        keepidx = 0
        while remaining:
            draws = super(UniformMasses, self).rvs(size=remaining)
            mask = self._apply_constraints(draws)
            addpts = mask.sum()
            arr[keepidx:keepidx+addpts] = draws[mask]
            keepidx += addpts
            remaining = size - keepidx
        return arr

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file. The parameters
        for the distribution are retrieved from the section titled
        "[`section`-`variable_args`]" in the config file.

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            `prior.VARARGS_DELIM`. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        Uniform
            A distribution instance from the pycbc.inference.prior module.
        """
        mchirp_bounds = bounded.get_param_bounds_from_config(cp, section,
            variable_args, 'mchirp')
        q_bounds = bounded.get_param_bounds_from_config(cp, section,
            variable_args, 'q')
        additional_opts = {'mchirp_bounds': mchirp_bounds, 'q_bounds': q_bounds}
        skip_opts = ['min-mchirp', 'max-mchirp', 'min-q', 'max-q']
        return bounded.bounded_from_config(cls, cp, section, variable_args,
            bounds_required=True, additional_opts=additional_opts,
            skip_opts=skip_opts)


__all__ = ['Uniform', 'UniformMasses']
