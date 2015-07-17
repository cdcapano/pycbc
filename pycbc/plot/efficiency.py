# Copyright (C) 2015  Collin Capano
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This module provides classes to organize injections in a hierarchy based on
their parameters for purposes of calculating sensitive volumes.
"""

import os
import numpy
import operator
import hashlib
import copy
import ConfigParser
from glue import segments

from pycbc import distributions
from pycbc.plot import plot_utils
from pycbc.plot import data_utils

from scipy import special
def beta_weight(result, alpha, beta):
    rmax = numpy.sqrt((numpy.array(result.inj_sigma.values())**2.).sum())/4.
    r = result.distance
    x = r/rmax
    return x**(1.-alpha) * (1.-x)**(1.-beta) * rmax * r**2.* special.beta(
        alpha, beta)

#
#
#   HTML utilities
#
#
def mathjax_html_header():
    """
    Standard header to use for html pages to display latex math.

    Returns
    -------
    header: str
        The necessary html head needed to use latex on an html page.
    """
    return """
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$']]}});
</script>
<script type="text/javascript"
    src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
"""

def mapper_header(mapper_jsfile):
    """
    Standard header for mapper js file.
    """
    return """
<script type="text/javascript" src="%s"></script>\n""" %(mapper_jsfile)

#
#
#   Data organization classes
#
#
class PHyperCube:

    use_distance_bins = False

    def __init__(self, bounds=None):
        self._bounds = {}
        if bounds is not None:
            self.set_bounds(**bounds)
        self.data = None
        self._ranking_stat = None
        self._rank_by = None
        self._stat_label = None
        self._nsamples = None
        self.global_norm = None
        self.total_nsamples = None
        self._cached_volumes = {}
        self._astro_prior = None
        self._cached_astro_weights = {}
        # for grouping together phyper cubes
        self.parent = None
        self.children = []
        # for storing plots and pages
        self.volumes_vs_stat_plot = None
        self.tiles_plot = None
        self.subtiles_plot = None
        self.additional_plots = []
        self.html_page = None
        # for easy identification of self within a collection of cubes in a
        # layer; this is useful for quickly picking out what additional plots
        # are associated with this cube in a plot_cache file; see Layer for
        # more details
        self.id_in_layer = None

    def set_bound(self, param, lower, upper):
        self._bounds[param] = segments.segment(lower, upper)

    def get_bound(self, param):
        try:
            return self._bounds[param]
        except KeyError:
            raise ValueError("parameter %s not in bounds; " %(param) + 
                "bound parameters are %s" %(', '.join([self._bounds.keys()])))

    def set_bounds(self, **kwargs):
        for param,(lower, upper) in kwargs.items():
            self.set_bound(param, lower, upper)

    @property
    def bounds(self):
        return self._bounds

    @property
    def params(self):
        return self._bounds.keys()

    def __contains__(self, other):
        """
        Determines whether or not other's bounds are in self's bounds. Is the
        same as "other in self". Since both self and other may be
        multi-dimensional, 3 possible values may be returned. They are:
        
          * 1: Means that both self and other's bounds have the same
            parameters. The segment bound of every parameter in other is in
            the segment bound of the same parameter in self, as determined
            by glue.segments.segment.__contains__.
          * -1: Means that self and other share some of the same parameters.
            The segment bound of every common parameter in other is in the
            segment bound of the same parameter in self.
          * 0: Means that some or all of the parameters of self and other are
            the same, but that one or more segment bounds in other are not in
            self.

        If none of the parameters in other's bounds match the parameters in
        self, a KeyError is raised.

        Parameters
        ----------
        other: PHyperCube
            The PHyperCube who's bounds are checked to see if they are in self.

        Returns
        -------
        int {-1, 0, 1}
            The result of the test.
        """
        common_params = set(other.params).intersection(set(self.params))
        if common_params == set():
            raise KeyError("other and self have no common parameters")
        result = int(all(other.get_bound(param) in self.get_bound(param) \
            for param in common_params))
        if len(self.params) == len(other.params) == len(common_params):
            result *= -1
        return result

    def central_value(self, param):
        param_range = self.get_bound(param)
        return param_range[0] + abs(param_range)/2.

    def set_astro_prior(self, prior):
        """
        Sets an astrophysical prior to use when computing sensitive volume.
        """
        self._astro_prior = prior

    @property
    def astro_prior(self):
        return self._astro_prior

    def set_data(self, data):
        """
        Given some data, extracts the points that are relevant to this cube.
        Also clears cached data (volumes, weights).
        """
        self.data = data_utils.slice_results(data, self.bounds)
        self._nsamples = len(self.data)
        self._cached_volumes.clear()
        self._cached_astro_weights.clear()
    
    @property
    def nsamples(self):
        """
        Gives the number of samples in self's data.
        """
        return self._nsamples

    def set_ranking_params(self, ranking_stat, rank_by, stat_label=None):
        """
        Set self's ranking_stat and rank_by. These are needed to compute
        sensitive volumes.

        Parameters
        ----------
        ranking_stat: str
            The stat to use when determining whether or injections are found
            or missed. Must be an attribute, or a math operation on a
            combination of attributes, of the elements in self.data
        rank_by: str {'max'|'min'}
            How the ranking stat scales with signficance. If 'max' ('min'),
            larger (smaller) values of the ranking stat will be considered
            more significant.
        """
        self._ranking_stat = ranking_stat
        if not (rank_by == 'max' or rank_by == 'min'):
            raise ValueError("unrecognized rank_by argument %s; " %(rank_by) +\
                "options are 'max' or 'min'")
        self._rank_by = rank_by
        if stat_label is not None:
            self._stat_label = stat_label
        else:
            self._stat_label = self._ranking_stat

    @property
    def ranking_stat(self):
        return self._ranking_stat

    @property
    def rank_by(self):
        return self._rank_by
        
    @property
    def stat_label(self):
        return self._stat_label

    def get_volume(self, threshold):
        """
        Retrieves the sensitive volume using the given threshold of self's
        ranking_stat. This is done by integrating the efficiency in self
        using self._integrate_efficiency. Volumes are cached to
        self._cached_volumes for faster retrieval later on. Self's
        _ranking_stat and _rank_by parameters must be set (see
        set_ranking_params).

        Parameters
        ----------
        threshold: float
            Value to use for considering what is a found injection and what is
            missed.

        Returns
        -------
        volume: float
            The sensitive volume.
        err: float
            The error on the sensitive volume measurement, found from the
            standard deviation of Monte Carlo integral.
        """
        try:
            return self._cached_volumes[threshold, self._ranking_stat, 
                self._rank_by]
        except KeyError:
            if self._ranking_stat is None:
                raise ValueError("must set ranking_stat and rank_by first; " +
                    "see set_ranking_params()")
            if self.use_distance_bins:
                volume, err_high, err_low = self._old_integrate_efficiencies(
                    self._ranking_stat, self._rank_by, threshold,
                    astro_prior=self.astro_prior)
            else:
                volume, err_high = self._integrate_efficiency(threshold,
                    self._ranking_stat, self._rank_by, self.astro_prior)
                err_low = err_high
            # FIXME
            if volume < 1e-7:
                volume = err_low = err_high = 0.
            self._cached_volumes[threshold,self._ranking_stat,self._rank_by] =\
                volume, err_high, err_low
            return volume, err_high, err_low

    def _integrate_efficiency(self, threshold, ranking_stat, rank_by,
            astro_prior=None):
        """
        Integrates the efficiency in self's bounds to get a measure of the
        sensitive volumes.
        """
        if rank_by == 'max':
            compare_operator = operator.ge
        elif rank_by == 'min':
            compare_operator = operator.lt
        else:
            raise ValueError("unrecognized rank_by argument %s; " %(rank_by) +\
                "options are 'max' or 'min'")
        if self.nsamples <= 1:
            raise ValueError("there must be more than one sample to " +
                "compute efficiency")
        integrand = numpy.array([x.injection.min_vol + x.injection.vol_weight*\
            float(_isfound(x, ranking_stat, compare_operator, threshold)) \
            for x in self.data if x.injection.vol_weight is not None])
        # apply the appropriate astrophysical prior
        # FIXME: this currently only works on masses
        try:
            weights = self._cached_astro_weights[astro_prior, self.global_norm]
        except KeyError:
            if astro_prior is not None:
                weights = numpy.array([distributions.convert_distribution(
                    x, x.injection.mass_distr, astro_prior) for x in self.data
                    if x.injection.vol_weight is not None])
            else:
                weights = numpy.ones(self.nsamples, dtype=float)
            # apply the appropriate norm to the weights
            if self.global_norm is not None:
                weights /= self.global_norm
            else:
                # otherwise, we're assuming a universe in which signals only exist
                # in this cube
                weights /= weights.sum()
            # save to cache
            self._cached_astro_weights[astro_prior, self.global_norm] = weights
        # if using a global norm, we'll set the number of samples to be the
        # total number of injections across the entire space, so that
        # volumes add to the total volume
        if self.global_norm is not None:
            Nsamples = self.total_nsamples
        else:
            Nsamples = self.nsamples
        mean = (weights * integrand).sum()
        # calucate a weighted variance
        # Note: there are multiple definitions in the literature as to what
        # to use for the standard error of a weighted mean. See D. Gatz and
        # L. Smith "The Standard Error of a Weighted Mean Concentration" (1995)
        # for a discussion. Here, I'll use:
        # var(x_w) = sum_i w_i (x_i - <x_w>)**2
        # where <x_w> is the weighted mean
        # [wikipedia.org/Variance#Discrete_random_variable]. Note that the
        # weights are  normalized, so I don't have to divide by anything).
        # The standard error of the mean is then:
        # d<x_w> = \sqrt(var(x_w)/Nsamples)
        # It might be good to check this eventually using a boot strap method.
        var = (weights * (integrand - mean)**2.).sum()
        return mean, numpy.sqrt(var/Nsamples)


    def clear_cache(self):
        """
        Clears self's _cached_volumes.
        """
        self._cached_volumes.clear()

    def create_html_page(self, out_dir, html_name, threshold,
            livetime=None, mainplots_widths=1000, print_relative_error=False,
            mapper=None, comments=None):
        """
        Create's self html page. See _create_html_page for details.
        """
        _create_html_page(self, out_dir, html_name, threshold=threshold,
            livetime=livetime, mainplots_widths=mainplots_widths,
            print_relative_error=print_relative_error, mapper=mapper,
            comments=comments)
        self.html_page = '%s/%s' %(out_dir, html_name)


    def _old_calculate_efficiencies(self, ranking_stat, rank_by, threshold,
            nbins=20, d_distr='linear', astro_prior=None):
        if rank_by == 'max':
            compare_operator = operator.ge
        elif rank_by == 'min':
            compare_operator = operator.lt
        else:
            raise ValueError("unrecognized rank_by argument %s; " %(rank_by) +\
                "options are 'max' or 'min'")
        distances = numpy.array([x.distance for x in self.data])

        weights = numpy.array([x.injection.vol_weight for x in self.data])
        integrand = numpy.array([\
            float(_isfound(x, ranking_stat, compare_operator, threshold)) \
            for x in self.data])

        min_dist = distances.min()
        max_dist = distances.max()
        if d_distr == 'log10':
            dist_bins = numpy.logspace(numpy.log10(min_dist),
                numpy.log10(max_dist), num=nbins)
        elif d_distr == 'linear':
            dist_bins = numpy.linspace(min_dist, max_dist, num=nbins)
        else:
            raise ValueError, "unrecognized distribution %s" % d_distr

        dist_bins[-1] *= 1.01

        # calculate the average efficiency in each distance bin
        thisEff = numpy.zeros(len(dist_bins)-1)
        numTotal = numpy.zeros(len(dist_bins)-1)
        for kk,dbin in enumerate(dist_bins[:-1]):
            nextbin = dist_bins[kk+1]
            group_idx = numpy.intersect1d(
                numpy.where(distances >= dbin)[0],
                numpy.where(distances < nextbin)[0])
            numTotal[kk] = len(group_idx)
            if len(group_idx) == 0:
                continue
            this_integrand = integrand[group_idx]
            these_weights = weights[group_idx]
            thisEff[kk] = (this_integrand * these_weights).sum() / \
                these_weights.sum()
            #this_integrand = this_integrand * these_weights / \
            #    these_weights.sum()
            #thisEff[kk] = this_integrand.mean()

        numFound = thisEff * numTotal
        numMissed = numTotal - numFound
        # if some bins have no injections in them
        # we'll combine with adjacent bins
        nzidx = numpy.nonzero(numTotal)
        numTotal = numTotal[nzidx]
        numFound = numFound[nzidx]
        numMissed = numMissed[nzidx]
        thisEff = thisEff[nzidx]
        dist_bins = numpy.array(dist_bins[nzidx].tolist() + \
            [dist_bins[-1]])

        use_frequentist_ci = True
        if use_frequentist_ci:
            # the error in the efficiency comes from eqn. 1.9 in J.T. Whalen's 
            # note cbc/protected/review/notes/poisson_errors.pdf in the
            # ligo/virgo cvs
            effPlus = (numTotal*(2.*numFound + 1.) + numpy.sqrt(
                4.*numTotal*numFound*(numTotal - numFound) + numTotal**2)) / \
                (2*numTotal*(numTotal + 1))
            effMinus = (numTotal*(2.*numFound + 1.) - numpy.sqrt(
                4.*numTotal*numFound*(numTotal - numFound) + numTotal**2)) / \
                (2*numTotal*(numTotal + 1))
            effPlus = effPlus - thisEff
            effMinus = thisEff - effMinus

        else:
            effPlus = numpy.sqrt(thisEff*(1. - thisEff)/numTotal)
            effMinus = effPlus

        return thisEff, effPlus, effMinus, dist_bins


    def _old_integrate_efficiencies(self, ranking_stat, rank_by, threshold,
            nbins=20, d_distr='linear', astro_prior=None):

        # calculate the efficiency
        efficiencies, eff_err_high, eff_err_low, distances = \
            self._old_calculate_efficiencies(ranking_stat, rank_by, threshold,
            nbins, d_distr, astro_prior)
        # now integrate
        if d_distr == 'linear':
            this_dist = distances[:-1] + numpy.diff(distances)/2.
            dr = numpy.diff(distances)
            #integrated_eff = 4*numpy.pi*(
            #    efficiencies * dr).sum() + \
            #    (4./3)*numpy.pi*distances[0]**3
            integrated_eff = 4*numpy.pi*(
                efficiencies * this_dist**2 * dr).sum() + \
                (4./3)*numpy.pi*distances[0]**3
            # error
            integrated_err_low = 4*numpy.pi*numpy.sqrt(((
                eff_err_low * this_dist**2 * dr)**2).sum())
            integrated_err_high = 4*numpy.pi*numpy.sqrt(((
                eff_err_high * this_dist**2 * dr)**2).sum())
        elif d_distr == 'log10':
            this_dist = 10**(numpy.log10(distances[:-1]) + numpy.diff(
                numpy.log10(distances))/2)
            # in this case we use:
            # int(eff) =
            # \int eff(r) r**2 \dr = ln10 \int eff(log10 r) r**3 \dlog10r
            dr = numpy.diff(numpy.log10(distances))
            integrated_eff = numpy.log(10)*(
                4.*numpy.pi*efficiencies * this_dist**3 * dr).sum() + \
                4*numpy.pi*(1./3)*distances[0]**3
            # error
            # note that in the following we're assuming the error in the
            # efficiency below and above the measured region is 0
            integrated_err_low = 4.*numpy.pi*numpy.sqrt(((
                eff_err_low * this_dist**3 * dr)**2).sum())
            integrated_err_high = 4.*numpy.pi*numpy.sqrt(((
                eff_err_high * this_dist**3 * dr)**2).sum())
        else:
            raise ValueError, 'unrecognized distance distribution %s' % d_distr

        return integrated_eff, integrated_err_high, integrated_err_low


class PHyperCubeGain:
    """
    Class to store information about a "tile" in parameter space. A tile is
    a two-dimensional slice in a higher dimensional parameter space that is
    bounded by some values. The class stores information about a "reference"
    and "test" set of results. When taking the ratio of sensitive volumes
    in the tile, the "reference" is the denominator and the "test" is the
    numerator.
    """
    def __init__(self, bounds=None):
        if bounds is None:
            bounds = {}
        self._reference_cube = PHyperCube(bounds)
        self._test_cube = PHyperCube(bounds)
        self._bounds = {}
        if bounds:
            self.set_bounds(**bounds)
        # values common to both
        self.fractional_gain = 0.
        self.gain_err_low = 0.
        self.gain_err_high = 0.
        # for grouping together phyper cubes
        self.parent = None
        self.children = []
        # for storing plots and pages
        self.volumes_vs_stat_plot = None
        self.tiles_plot = None
        self.subtiles_plot = None
        self.additional_plots = []
        self.html_page = None
        # for easy identification of self within a collection of cubes in a
        # layer; this is useful for quickly picking out what additional plots
        # are associated with this cube in a plot_cache file; see Layer for
        # more details
        self.id_in_layer = None

    @property
    def reference_cube(self):
        return self._reference_cube

    @property
    def test_cube(self):
        return self._test_cube

    def set_bound(self, param, lower, upper):
        self._bounds[param] = segments.segment(lower, upper)
        self._reference_cube.set_bound(param, lower, upper)
        self._test_cube.set_bound(param, lower, upper)

    def get_bound(self, param):
        return self._bounds[param]

    def set_bounds(self, **kwargs):
        for param,(lower, upper) in kwargs.items():
            self.set_bound(param, lower, upper)

    @property
    def bounds(self):
        return self._bounds

    @property
    def params(self):
        return self._bounds.keys()

    def __contains__(self, other):
        """
        Since the reference cube's bounds are the same as self's and 
        self.test_cube's bounds, uses the reference_cube  __contains__ function
        to do the comparison between self and other. See
        PHyperCubeGain.__contains__ for details.
        """
        return other._reference_cube in self._reference_cube

    def central_value(self, param):
        param_range = self.get_bound(param)
        return param_range[0] + abs(param_range)/2.

    def set_reference_data(self, data):
        self._reference_cube.set_data(data)

    @property
    def reference_data(self):
        return self._reference_cube.data

    def set_test_data(self, data):
        self._test_cube.set_data(data)

    @property
    def test_data(self):
        return self._test_cube.data

    @property
    def nsamples(self):
        return min(self._test_cube.nsamples, self._reference_cube.nsamples)

    def _set_ranking_params(self, ref_or_test, ranking_stat, rank_by,
            stat_label=None):
        """
        Sets the reference or test cube's ranking_stat and rank_by. See
        PHyperCube.set_ranking_params for details.
        """
        getattr(self, '_%s_cube' %(ref_or_test)).set_ranking_params(
            ranking_stat, rank_by, stat_label)

    def set_reference_ranking_params(self, ranking_stat, rank_by,
            stat_label=None):
        """
        Set's the reference cube's ranking params. See
        PHyperCube.set_ranking_params for details.
        """
        self._set_ranking_params('reference', ranking_stat, rank_by,
            stat_label)

    def set_test_ranking_params(self, ranking_stat, rank_by, stat_label=None):
        """
        Set's the test cube's ranking params.
        """
        self._set_ranking_params('test', ranking_stat, rank_by, stat_label)

    def get_fractional_gain(self, ref_threshold, test_threshold):
        """
        Gets the fractional gain and error.

        Parameters
        ----------
        ref_threshold: float
            The threshold to use for the reference data. The reference
            ranking params must be set; see set_reference_ranking_params.
        test_threshold: float
            Same, but for the test data/cube.
        """
        if self._reference_cube.ranking_stat is None:
            raise ValueError("reference ranking params not set")
        if self._test_cube.ranking_stat is None:
            raise ValueError("test ranking params not set")
        ref_volume, ref_err, _ = self._reference_cube.get_volume(ref_threshold)
        test_volume, test_err, _ = self._test_cube.get_volume(test_threshold)
        if ref_volume == 0.:
            fractional_gain, gain_err = numpy.inf, 0.
        else:
            fractional_gain = test_volume / ref_volume
            gain_err = fractional_gain * \
                numpy.sqrt((test_err / test_volume)**2. + \
                           (ref_err / ref_volume)**2.)
        return fractional_gain, gain_err

    def create_html_page(self, out_dir, html_name, test_threshold,
        ref_threshold, test_label='', ref_label='',
        test_livetime=None, ref_livetime=None, print_relative_error=False,
        mainplots_widths=1000,
        mapper=None, comments=None):
        """
        Create's self html page. See _create_html_page for details.
        """
        _create_gain_html_page(self, out_dir, html_name,
            threshold=test_threshold, ref_threshold=ref_threshold,
            test_livetime=test_livetime, ref_livetime=ref_livetime,
            test_label=test_label, ref_label=ref_label,
            mainplots_widths=mainplots_widths,
            print_relative_error=print_relative_error, mapper=mapper,
            comments=comments)
        self.html_page = '%s/%s' %(out_dir, html_name)


#
#
#   Utilities for PHyperCube(Gain)s
#
#
def _isfound(result, ranking_stat, compare_operator, threshold):
    """
    Determins whether an injection is found.

    Parameters
    ----------
    result: plot.Result
        Result class instance representing an injection.
    ranking_stat: str
        What to use for the ranking. Must be a parameter of plot.Result.
    compare_operator: operator function
        The comparison operator to use. For example, if this is operator.ge,
        then result's ranking_stat must be >= threshold to be found.
    threshold: float
        The threshold to use.

    Returns
    -------
    _isfound: bool
    """
    return compare_operator(data_utils.get_arg(result, ranking_stat),
        threshold)


def ischild(parent, child):
    """
    Checks if the parent's bounds contains the childs bounds.
    False is returned if parent.__contains__(child) is 0. If
    parent.__contains__(child) raises a KeyError, False is returned. 
    """
    try:
        return abs(child in parent)
    except KeyError:
        return False

def _construct_links(plotted_cubes, tiles):
    """
    Constructs a list of links from a list of phyper cubes and their
    corresponding tiles in a plot so that links can be passed to
    plot_utils.plot2imgmap.
    """
    links = []
    tags = []
    for cube, tile in zip(plotted_cubes, tiles):
        # if the plotted cube doesn't have an html page to link to, skip it
        if cube.html_page is None:
            continue
        # tile's coordinates go clockwise around the tile, starting from the
        # bottom left
        bl, tl, tr, br, _ = tile.xy
        # to create an image map, we need the bottom left and top right
        # corners
        links.append((bl, tr, './'+cube.html_page))
        # for the tags, we'll include the x, y values
        tags.append('x: [%f, %f)\ny: [%f, %f)' %(bl[0], br[0], bl[1], tl[1]))
    return links, tags


def format_volume_text(V, err_plus, err_minus=None, include_units=True,
        use_scientific_notation=True, use_relative_err=False):
    """
    Given a volume and it's, returns a latex string rounded to the appropriate
    number of significant figures, with units.
    """
    if err_minus is None:
        err_minus = err_plus
    if not use_scientific_notation or V == 0.:
        conversion_factor = 0.
    else:
        conversion_factor = numpy.floor(numpy.log10(V))
    V = V * 10**(-conversion_factor)
    err_plus = err_plus * 10**(-conversion_factor)
    err_minus = err_minus * 10**(-conversion_factor)
    err = min(err_minus, err_plus)
    # if the conversion factor is > 10^9, we'll change the label to Gpc
    # (V is assumed to be in Mpc^3)
    if conversion_factor >= 9 and include_units:
        conversion_factor -= 9.
        units = '\mathrm{Gpc}^3'
    elif include_units:
        units = '\mathrm{Mpc}^3'
    else:
        units = ''
    if conversion_factor == 0.:
        prefactor = ''
    elif conversion_factor == 1.:
        prefactor = r'\times 10'
    else:
        prefactor = r'\times 10^{%i}' %(int(conversion_factor))
    units_label = '%s\,%s' %(prefactor, units)

    # now round the the appropriate number of sig figs
    voltxt = plot_utils.get_signum(V, err)
    if err_plus == err_minus:
        errtxt = plot_utils.get_signum(err, err)
        if use_relative_err and float(voltxt) != 0.:
            relative_err = 100.*float(errtxt)/float(voltxt)
            # we round the relative error to the nearest 1% using get_signum;
            # Note that if the relative error is < 1%, get_signum will
            # automatically increase the number of values after the decimal
            # until it gets to the first non-zero value
            relative_err = plot_utils.get_signum(relative_err, 1.)
            return r'$%s %s \pm %s \%%$' %(voltxt, units_label, relative_err)
        else: 
            return r'$%s \pm %s %s$' %(voltxt, errtxt, units_label)
    else:
        err_minus_txt = plot_utils.get_signum(err_minus, err)
        err_plus_txt = plot_utils.get_signum(err_plus, err)
        if use_relative_err and float(voltxt) != 0.:
            # same as above, but with plus and minus
            rel_err_plus = plot_utils.get_signum(
                100.*float(err_plus_txt)/float(voltxt), 1.)
            rel_err_minus = plot_utils.get_signum(
                100.*float(err_minus_txt)/float(voltxt), 1.)
            return r'$%s %s \substack{+%s\\-%s}\%%$' %(voltxt, units_label,
                rel_err_plus, rel_err_minus)
        else:
            return r'$%s \substack{+ %s\\- %s} %s$' %(voltxt, err_plus_txt,
                err_minus_txt, units_label)


def _create_html_page(phyper_cube, out_dir, html_name, livetime=None,
        threshold=None,
        ref_threshold=None, mainplots_widths=1000, print_relative_error=False,
        mapper=None, comments=None):
    """
    Creates an html page for the given PHyperCube.
    """
    # check that the given thresholds match the phyper_cube type
    write_gain = isinstance(phyper_cube, PHyperCubeGain)
    if write_gain and ref_threshold is None:
        raise ValueError("ref_threshold needed for PHyperCubeGain pages")
    if threshold is None:
        raise ValueError("must provide a threshold")

    #
    #   Open the html page and write the header
    #
    html_page = '%s/%s' % (out_dir, html_name)
    f = open(html_page, 'w')
    print >> f, "<!DOCTYPE html>\n<html>"
    # we'll use MathJax to be able to display latex math
    print >> f, "<head>"
    print >> f, mathjax_html_header()
    if mapper is not None:
        print >> f, mapper_header(mapper)
    print >> f, "</head>"
    #
    # Now the body
    #
    print >> f, "<body>"
    # we'll put the bounds in the heading
    print >> f, "<h1>"
    for (param, (lower, upper)) in sorted(phyper_cube.bounds.items()):
        # if the param is a Parameter instance with a label, we can
        # just get the label to use from it; otherwise, we'll just use
        # the parameter name
        try:
            plbl = param.label
        except AttributeError:
            plbl = param
        print >> f, r"%s $\in [%s, %s)$<br />" %(plbl,
            plot_utils.drop_trailing_zeros(lower),
            plot_utils.drop_trailing_zeros(upper))
    print >> f, "</h1>"
    # print some basic info about this cube
    print >> f, "<h2>"
    print >> f, "Total number of injections: %i<br />" %(
        phyper_cube.nsamples)
    if phyper_cube.astro_prior is not None:
        print >> f, "Astrophysical prior: %s<br />" %(
            phyper_cube.astro_prior.description)
    if phyper_cube.nsamples > 1:
        if write_gain:
            V, err, err_minus = phyper_cube.test_cube.get_volume(threshold)
            stat_label = phyper_cube.test_cube.stat_label
        else:
            V, err, err_minus = phyper_cube.get_volume(threshold)
            stat_label = phyper_cube.stat_label
        print >> f, "Sensitive volume at %s $= %s$:<br />&nbsp&nbsp%s<br />" %(
            stat_label, plot_utils.drop_trailing_zeros(threshold),
            format_volume_text(V, err, err_minus=err_minus,
            use_relative_err=print_relative_error))
        if livetime is not None:
            print >> f, "Livetime: %.2f days <br />" %(
                livetime / (3600.*24.))
        if write_gain:
            V, err, err_minus = phyper_cube.reference_cube.get_volume(
                ref_threshold)
            print >> f, "Reference sensitive volume at " +\
                "%s $= %s$:<br />&nbsp&nbsp%s<br />" %(
                phyper_cube.reference_cube.stat_label,
                plot_utils.drop_trailing_zeros(ref_threshold),
                format_volume_text(V, err, err_minus=err_minus,
                use_relative_err=print_relative_error))
            G, err = phyper_cube.get_fractional_gain(ref_threshold,
                threshold)
            print >> f, "Relative gain:<br />&nbsp&nbsp%s<br />" %(
                format_volume_text(G, err, include_units=False,
                    use_relative_err=print_relative_error,
                    use_scientific_notation=False))
            
    print >> f, "</h2>"
    # put the V vs stat plot
    if phyper_cube.volumes_vs_stat_plot is not None:
        print >> f, "<hr />"
        mfig = phyper_cube.volumes_vs_stat_plot
        figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
        print >> f, '<img src="%s" width="%i" />' %(figname,
            mainplots_widths)
    # put the tiles plot
    if phyper_cube.tiles_plot is not None:
        print >> f, "<hr />"
        # if no clickable elements, just add the plot
        mfig = phyper_cube.tiles_plot
        # ensure links are set appropriately
        [setattr(clickable, 'link', clickable.data.html_page) \
            for clickable in mfig.clickable_elements]
        if mfig.clickable_elements == [] or not \
                all([clickable.link is not None \
                for clickable in mfig.clickable_elements]):
            figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
            print >> f, '<img src="%s" width="%i" />' %(figname,
                mainplots_widths)
        else:
            print >> f, "Click on a tile to go to the next layer. <br />"
            print >> f, mfig.create_image_map(html_page,
                view_width=mainplots_widths)
    # put the subtiles plot
    if phyper_cube.subtiles_plot is not None:
        print >> f, "<hr />"
        mfig = phyper_cube.subtiles_plot
        [setattr(clickable, 'link', clickable.data.html_page) \
            for clickable in mfig.clickable_elements]
        if mfig.clickable_elements == [] or not \
                all([clickable.link is not None \
                for clickable in mfig.clickable_elements]):
            figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
            print >> f, '<img src="%s" width="%i" />' %(figname,
                mainplots_widths)
        else:
            print >> f, "Click on a tile to go to the next layer. <br />"
            print >> f, mfig.create_image_map(html_page,
                view_width=mainplots_widths)
    # if there are additional plots, add them
    if phyper_cube.additional_plots != []:
        print >> f, "<hr />"
        print >> f, "<h3>Additional plots</h3>"
        for figname in phyper_cube.additional_plots:
            figname = os.path.relpath(os.path.abspath(figname),
                os.path.dirname(os.path.abspath(html_page)))
            print >> f, '<a href="%s"><img src="%s" width="%i" /></a>' %(
                figname, figname, int(mainplots_widths/2))
    # add any comments
    if comments is not None:
        print >> f, "<hr />"
        print >> f, comments
    # close out
    print >> f, '</body>\n</html>'
    f.close()


def _create_gain_html_page(phyper_cube, out_dir, html_name,
        test_livetime=None, ref_livetime=None,
        threshold=None,
        ref_threshold=None, test_label=None, ref_label=None,
        mainplots_widths=1000, print_relative_error=False,
        mapper=None, comments=None):
    """
    Creates an html page for the given PHyperCubeGain.
    """
    if ref_threshold is None:
        raise ValueError("must provide a threshold")
    if threshold is None:
        raise ValueError("must provide a threshold")

    if test_label is None:
        test_label = 'Test'
    if ref_label is None:
        ref_label = 'Reference'
    # make labels colored to stick out
    test_label = '<font color="#8B0000">%s</font>' %(test_label)
    ref_label = '<font color="#00008B">%s</font>' %(ref_label)

    #
    #   Open the html page and write the header
    #
    html_page = '%s/%s' % (out_dir, html_name)
    f = open(html_page, 'w')
    print >> f, "<!DOCTYPE html>\n<html>"
    # we'll use MathJax to be able to display latex math
    print >> f, "<head>"
    print >> f, mathjax_html_header()
    if mapper is not None:
        print >> f, mapper_header(mapper)
    print >> f, "</head>"
    #
    # Now the body
    #
    print >> f, "<body>"
    # we'll put the bounds in the heading
    print >> f, "<h1>"
    for (param, (lower, upper)) in sorted(phyper_cube.bounds.items()):
        # if the param is a Parameter instance with a label, we can
        # just get the label to use from it; otherwise, we'll just use
        # the parameter name
        try:
            plbl = param.label
        except AttributeError:
            plbl = param
        print >> f, r"%s $\in [%s, %s)$<br />" %(plbl,
            plot_utils.drop_trailing_zeros(lower),
            plot_utils.drop_trailing_zeros(upper))
    print >> f, "</h1>"
    # print some basic info about this cube
    print >> f, "<h2>"
    print >> f, "%s number of injections: %i<br />" %(
        test_label, phyper_cube.test_cube.nsamples)
    print >> f, "%s number of injections: %i<br />" %(
        ref_label, phyper_cube.reference_cube.nsamples)
    if phyper_cube.reference_cube.astro_prior is not None:
        print >> f, "Astrophysical prior: %s<br />" %(
            phyper_cube.reference_cube.astro_prior.description)
    if phyper_cube.nsamples > 1:
        V, err, err_minus = phyper_cube.test_cube.get_volume(threshold)
        stat_label = phyper_cube.test_cube.stat_label
        print >> f, "%s sensitive volume at " %(test_label) +\
            "%s $= %s$:<br />&nbsp&nbsp%s<br />" %(
            stat_label, plot_utils.drop_trailing_zeros(threshold),
            format_volume_text(V, err, err_minus=err_minus,
            use_relative_err=print_relative_error))
        if test_livetime is not None:
            print >> f, "%s livetime: %.2f days <br />" %(test_label,
                test_livetime / (3600.*24.))
        V, err, err_minus = phyper_cube.reference_cube.get_volume(
            ref_threshold)
        print >> f, "%s sensitive volume at " %(ref_label) +\
            "%s $= %s$:<br />&nbsp&nbsp%s<br />" %(
            phyper_cube.reference_cube.stat_label,
            plot_utils.drop_trailing_zeros(ref_threshold),
            format_volume_text(V, err, err_minus=err_minus,
            use_relative_err=print_relative_error))
        if ref_livetime is not None:
            print >> f, "%s livetime: %.2f days<br />" %(ref_label,
                ref_livetime / (3600.*24.))
        G, err = phyper_cube.get_fractional_gain(ref_threshold,
            threshold)
        if test_livetime is not None:
            G = G * test_livetime/ref_livetime
            err = err * test_livetime / ref_livetime
        print >> f, "Relative gain (%s/%s):<br />&nbsp&nbsp%s<br />" %(
            test_label, ref_label,
            format_volume_text(G, err, include_units=False,
                use_relative_err=print_relative_error,
                use_scientific_notation=False))
    print >> f, "</h2>"
    print >> f, "<hr />"
    # put the V vs stat plot
    if phyper_cube.volumes_vs_stat_plot is not None:
        mfig = phyper_cube.volumes_vs_stat_plot
        figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
        print >> f, '<img src="%s" width="%i" />' %(figname,
            mainplots_widths)
        print >> f, "<hr />"
    # put the tiles plot
    if phyper_cube.tiles_plot is not None:
        # if no clickable elements, just add the plot
        mfig = phyper_cube.tiles_plot
        # ensure links are set appropriately
        [setattr(clickable, 'link', clickable.data.html_page) \
            for clickable in mfig.clickable_elements]
        if mfig.clickable_elements == [] or not \
                all([clickable.link is not None \
                for clickable in mfig.clickable_elements]):
            figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
            print >> f, '<img src="%s" width="%i" />' %(figname,
                mainplots_widths)
        else:
            print >> f, "Click on a tile to go to the next layer. <br />"
            print >> f, mfig.create_image_map(html_page,
                view_width=mainplots_widths)
    # put the subtiles plot
    if phyper_cube.subtiles_plot is not None:
        print >> f, "<hr />"
        mfig = phyper_cube.subtiles_plot
        [setattr(clickable, 'link', clickable.data.html_page) \
            for clickable in mfig.clickable_elements]
        if mfig.clickable_elements == [] or not \
                all([clickable.link is not None \
                for clickable in mfig.clickable_elements]):
            figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
            print >> f, '<img src="%s" width="%i" />' %(figname,
                mainplots_widths)
        else:
            print >> f, "Click on a tile to go to the next layer. <br />"
            print >> f, mfig.create_image_map(html_page,
                view_width=mainplots_widths)
    # if the reference or test cubes have volume plots, add them
    if phyper_cube.test_cube.tiles_plot is not None:
        print >> f, "<hr /><h3>%s volumes</h3>" %(test_label)
        mfig = phyper_cube.test_cube.tiles_plot
        figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
            os.path.dirname(os.path.abspath(html_page)))
        print >> f, '<img src="%s" width="%i" /><br />' %(
            figname, mainplots_widths)
    if phyper_cube.reference_cube.tiles_plot is not None:
        print >> f, "<hr /><h3>%s volumes</h3>" %(ref_label)
        mfig = phyper_cube.reference_cube.tiles_plot
        figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
            os.path.dirname(os.path.abspath(html_page)))
        print >> f, '<img src="%s" width="%i" /><br />' %(
            figname, mainplots_widths)
    # if there are additional plots, add them
    if phyper_cube.additional_plots != []:
        print >> f, "<hr />"
        print >> f, "<h3>Additional plots</h3>"
        for figname in phyper_cube.additional_plots:
            figname = os.path.relpath(os.path.abspath(figname),
                os.path.dirname(os.path.abspath(html_page)))
            print >> f, '<a href="%s"><img src="%s" width="%i" /></a>' %(
                figname, figname, int(mainplots_widths/2))
    # add any comments
    if comments is not None:
        print >> f, "<hr />"
        print >> f, comments
    # close out
    print >> f, '</body>\n</html>'
    f.close()


class Parameter(str):
    """
    Class to store information about a parameter. The class inherits from str.
    The string itself is the parameter "argument"; this is what you would use
    for retrieving the parameter from, say, a Result class. This class adds a
    label attribute to the string. The label is itself a string, but it is
    what you might put onto a plot axis. The label is optional; if none
    provided, the label is set to None. For example:
    ::
        >>> p = Parameter('mchirp', '$\mathcal{M}~(\mathrm{M}_\odot)$')

        >>> p
            'mchirp'

        >>> p.label
            '$\\mathcal{M}~(\\mathrm{M}_\\odot)$'

        >>> p = Parameter('mchirp')

        >>> p
            'mchirp'

        >>> p.label

    """
    def __new__(cls, arg, label=None):
        obj = str.__new__(cls, arg)
        obj.label = label
        return obj


class Layer:
    """
    Class to store and organize parents PHyperCubes, as well as other layers.
    """
    def __init__(self, level, x_arg, y_arg, x_label=None, y_label=None,
            cube_type=None):
        self.level = level
        if x_label is None:
            x_label = x_arg
        self.x_param = Parameter(x_arg, x_label)
        if y_label is None:
            y_label = y_arg
        self.y_param = Parameter(y_arg, y_label)
        self._x_bins = segments.segmentlist([])
        self._y_bins = segments.segmentlist([])
        self._x_distr = None
        self._y_distr = None
        self._super_layer = None
        self._sub_layer = None
        self._parents = []
        self._cube_type = 'single'
        if cube_type is not None:
            self.set_cube_type(cube_type)
        # plot options
        self.plot_x_min = None
        self.plot_x_max = None
        self.plot_log_x = False
        self.plot_y_min = None
        self.plot_y_max = None
        self.plot_log_y = False
        # attributes for organizing html pages, plots
        self.root_dir = None
        self.web_dir = None
        self.images_dir = None

    def set_bins(self, x_or_y, lower_bounds, upper_bounds, distr):
        """
        Sets the bins in x or y.

        Parameters
        ----------
        x_or_y: str {'x'|'y'}
            Which axis to set the bins for.
        lower_bounds: numpy.array
            List of lower bounds of bin boundaries.
        upper_bounds: numpy.array
            List of upper bounds of bin boundaries. Length must be the same as
            lower_bounds.
        distr: str {'linear'|'log10'|'custom'}
            The distribution used to create the bins.
        """
        if len(lower_bounds) != len(upper_bounds):
            raise ValueError("length of bounds must match")
        # the actual bins will be a segments list
        boundslist = segments.segmentlist(map(segments.segment,
            zip(lower_bounds, upper_bounds)))
        # set
        setattr(self, '_%s_bins' % x_or_y, boundslist)
        setattr(self, '_%s_distr' % x_or_y, distr)

    def create_bins(self, x_or_y, minval, maxval, num, distr):
        """
        Creates and sets the bins.

        Parameters
        ----------
        x_or_y: str {'x'|'y'}
            Which axis to set the bins for.
        minval: float
            The minimum value to use.
        maxval: float
            The maximum value to use.
        num: int
            The number of bins to create.
        distr: str {'linear'|'log10'}
            How to distribute the bins between the minimum and maximum values.
        """
        if distr == 'log10':
            bins = numpy.logspace(numpy.log10(minval), numpy.log10(maxval),
                num=num+1)
        elif distr == 'linear':
            bins = numpy.linspace(minval, maxval, num=num+1)
        else:
            raise ValueError("unrecognized distribution %s; " %(distr) +\
                "options are 'linear' or 'log10'")
        lower_bounds = bins[:-1]
        upper_bounds = bins[1:]
        self.set_bins(x_or_y, lower_bounds, upper_bounds, distr)

    @property
    def x_lims(self):
        return self._x_bins.extent()

    @property
    def y_lims(self):
        return self._y_bins.extent()

    @property
    def x_nbins(self):
        return len(self._x_bins)

    @property
    def y_nbins(self):
        return len(self._y_bins)

    @property
    def x_distr(self):
        return self._x_distr

    @property
    def y_distr(self):
        return self._y_distr

    @property
    def cube_type(self):
        return self._cube_type

    def set_cube_type(self, cube_type):
        if cube_type == 'single':
            self._cube_type = PHyperCube
        elif cube_type == 'gain':
            self._cube_type = PHyperCubeGain
        else:
            raise ValueError("unrecognized cube_type %s; " %(cube_type) +\
                "must be either 'gain' or 'single'")

    @property
    def super_layer(self):
        return self._super_layer

    def set_super_layer(self, layer):
        """
        Sets the layer that is above this one.

        Parameters
        ----------
        layer: Layer instance
            The layer to set as being above this one. The level number must
            of layer must be > this layer's level.
        """
        if layer.level >= self.level:
            raise ValueError("given layer has a larger level number than self")
        if layer._cube_type != self._cube_type:
            raise ValueError("given layer has a different cube type than self")
        self._super_layer = layer


    @property
    def sub_layer(self):
        return self._sub_layer

    def set_sub_layer(self, layer):
        """
        Sets the layer that is below this one.

        Parameters
        ----------
        layer: Layer instance
            The layer to set as being above this one. The level number must
            of layer must be < this layer's level.
        """
        if layer.level <= self.level:
            raise ValueError("given layer has a smaller level number")
        if layer._cube_type != self._cube_type:
            raise ValueError("given layer has a different cube type than self")
        self._sub_layer = layer


    def create_cubes(self):
        """
        Creates all parameter hypercubes at this layer, along with the lineage
        linking cubes at this level to the children cubes of the super layer.
        If this level is > 0, super_layer must be set.
        """
        # first, tile up this layer
        if self.x_distr == 'custom' or self.y_distr == 'custom':
            this_layer_tiles = zip(self._x_bins, self._y_bins)
        else:
            this_layer_tiles = [(xbin, ybin) for ybin in self._y_bins \
                for xbin in self._x_bins]
        # now, if there is a super layer, create a child phyper cube for every
        # tile for every parent phyper cube in super layer
        if self.level > 0:
            if self._super_layer is None:
                raise ValueError("self.level is %i but super layer not set" %(
                    self.level))
            if self._super_layer._parents == []:
                raise ValueError("super layer's parents have not been " +\
                    "created yet")
            # the parents of this layer are the children of super_layer's
            self._parents = self._super_layer.all_children
        else:
            # the level 0 layer just has a single parent, which is the cube
            # that contains all of the data
            self._parents = [self._cube_type({self.x_param: self.x_lims,
                self.y_param: self.y_lims})]
        for parent in self._parents:
            # the children are all of the tiles in self
            for (xbounds, ybounds) in this_layer_tiles:
                these_bounds = copy.copy(parent.bounds)
                these_bounds[self.x_param] = xbounds
                these_bounds[self.y_param] = ybounds
                child = self._cube_type(these_bounds)
                parent.children.append(child)
                child.parent = parent


    @property
    def parents(self):
        return self._parents

    @property
    def all_children(self):
        return [child for parent in self._parents \
            for child in parent.children]

    def set_cube_layer_ids(self):
        """
        Assigns a unique id to each child phyper cube in the layer. The id is
        (level, idx), where level is the level number of the layer and idx is
        the index of the tile in the layer.all_children list. If self.level is
        0, then the (single) parent cube's id will be set to (0,-1).
        Otherwise, parent cube ids are not set, as they are the children of
        self's super_layer.
        """
        [setattr(child, 'id_in_layer', (self.level, ii)) for ii,child in \
            enumerate(self.all_children)]
        if self.level == 0:
            self.parents[0].id_in_layer = (0, -1)

    def set_cube_data(self, data, ref_or_test=None):
        """
        Bins the given data into self's parents of phyper cubes. If self's
        _cube_type is PHyperCubeGain, must also specify whether the data is
        test or reference.

        Parameters
        ----------
        data: dict
            The data to slice.
        ref_or_test: {None, 'reference'|'test'}
            If self's _cube_type is PHyperCubeGain, specifies whether the given
            data is for the reference cubes or the test cubes.
        """
        if self._parents == []:
            raise ValueError("parents have not been created yet; " +
                "run create_cubes")
        if self._cube_type == PHyperCubeGain:
            if ref_or_test is None:
                raise ValueError("_cube_type is PHyperCubeGain, but " +\
                    "ref_or_test not set")
            if not (ref_or_test == 'reference' or ref_or_test == 'test'):
                raise ValueError("unrecognized option for ref_or_test %s" %(
                    ref_or_test))
            funcstr = '_' + ref_or_test

        else:
            if ref_or_test is not None:
                raise ValueError("ref_or_test is not None, but _cube_type "+\
                    "is PHyperCube")
            funcstr = ''
        # set the parent, child data
        for parent in self._parents:
            getattr(parent, 'set%s_data' %(funcstr))(data)
            # now set the children
            for child in parent.children:
                getattr(child, 'set%s_data' %(funcstr))(data)


    def set_cube_data_using_super(self):
        """
        Uses self's super layers to slice the data for each cube in self.
        """
        if self.level == 0:
            raise ValueError("This is level 0: no super-layers!")
        if self._cube_type == PHyperCubeGain:
            data_sets = ['reference_', 'test_']
        else:
            data_sets = ['']
        for dset in data_sets:
            for parent in self._parents:
                inherited_data = getattr(parent, '%sdata' %(dset))
                if inherited_data is None:
                    raise ValueError("Parent data not set! This is probably " +
                        "because the data of this layer's super layer " +
                        "has not been set yet.")
                [getattr(child, 'set_%sdata' %(dset))(inherited_data) \
                    for child in parent.children]

    def set_ranking_params(self, ranking_stat, rank_by, stat_label=None,
            ref_or_test=None):
        """
        Sets the ranking_stat and rank_by for all parent and children cubes
        in self. If self's cube_type is PHyperCubeGain, ref_or_test must
        be either 'reference' or 'test' to specify whehther to set the ranking
        parameters of the reference or test cubes.
        """
        if self._cube_type == PHyperCubeGain:
            if not (ref_or_test == 'test' or ref_or_test == 'reference'):
                raise ValueError("_cube_type is PHyperCubeGain, but " +
                    "ref_or_test not set to 'reference' or 'test' (got %s)" %(
                    str(ref_or_test)))
            ref_or_test = '%s_' %(ref_or_test)
        if self._cube_type == PHyperCube:
            if ref_or_test != None:
                raise ValueError("ref_or_test is not None, but _cube_type "+\
                    "is PHyperCube")
            ref_or_test = ''

        [getattr(parent, 'set_%sranking_params' %(ref_or_test))(ranking_stat,
            rank_by, stat_label) for parent in self.parents]
        [getattr(child, 'set_%sranking_params' %(ref_or_test))(ranking_stat,
            rank_by, stat_label) for child in self.all_children]


def parse_custom_bins(bins_argument):
    """
    Parses a string listing custom bins to use.
    
    Parameters
    ----------
    bins_argument: str
        String giving the custom bins. Format must be:
        {xmin1, xmax1, ymin1, ymax1};{xmin2, xmax2, ymin2, ymax2};{etc.}
        (white space doesn't matter).

    Returns
    -------
    x_lower_bounds: numpy array
        Array of the x lower bounds.
    x_upper_bounds: numpy array
        Array of the x upper bounds.
    y_lower_bounds: numpy array
        Array of the y lower bounds.
    y_upper_bounds: numpy array
        Array of the y upper bounds.
    """
    bins = bins_argument.replace('{', '').replace('}', '').split(';')
    x_lower_bounds = numpy.zeros(len(bins))
    x_upper_bounds = numpy.zeros(len(bins))
    y_lower_bounds = numpy.zeros(len(bins))
    y_upper_bounds = numpy.zeros(len(bins))
    for ii,thisbin in enumerate(bins):
        xmin, xmax, ymin, ymax = map(float, thisbin.split(','))
        x_lower_bounds[ii] = xmin
        x_upper_bounds[ii] = xmax
        y_lower_bounds[ii] = ymin
        y_upper_bounds[ii] = ymax
    return x_lower_bounds, x_upper_bounds, y_lower_bounds, y_upper_bounds


def create_layers_from_config(config_file, cube_type='single'):
    """
    Constructs a hierarchy of layers from the given configuration file.

    The configuration file should have a section called 'layers'. Each option
    in this section is a layer; the details for that layer are provided by
    another section in the ini file with the same name as the option. The
    option should be set to the level number for the layer. The section with
    the details about the layer must have an x-arg, x-label, x-nbins, x-min,
    x-max, x-distr, and the same for y. For example:

    ::
        [layers]
        mtotal-q = 0

        [mtotal-q]
        ; x parameters
        x-arg = mtotal
        x-label = $M (\mathrm{M}_\odot)$
        x-min = 6.
        x-max = 50.
        x-nbins = 4
        x-distr = linear
        plot-x-min = 4.
        plot-x-max = 52.
        ;plot-log-x =
        ; y parameters
        y-arg = q
        y-label = $q$
        y-min = 1.
        y-max = 16.
        y-nbins = 4
        y-distr = log10
        plot-y-min = 0.9
        plot-y-max = 20.
        ;plot-log-y =

    Parameters
    ----------
    config_file: str
        File name of the configuration file.
    cube_type: str {'single'|'gain'}
        What cube type to create in the layers. If 'single', PHyperCubes will
        be created in each layer. If 'gain', PHyperCubeGains will be created
        in each layer.

    Returns
    -------
    layers: list
        The list of layers that are created. The order of the layers in the
        list is the same as their level numbers.
    """
    # load the layers
    if not os.path.exists(config_file):
        raise ValueError('config file %s does not exist' %(config_file))
    cp = ConfigParser.ConfigParser()
    cp.read(config_file)

    # construct the layers and parameter hyper cubes (phyper cubes)
    layer_names = dict([ [int(cp.get('layers', lname)), lname] for lname in \
        cp.options('layers')])

    layers = []
    for level, layer_name in sorted(layer_names.items()):
        x_arg = cp.get(layer_name, 'x-arg')
        y_arg = cp.get(layer_name, 'y-arg')
        x_label = cp.get(layer_name, 'x-label')
        y_label = cp.get(layer_name, 'y-label')
        this_layer = Layer(level, x_arg, y_arg, x_label, y_label, cube_type)
        # if custom bins are specified, set them
        if cp.has_option(layer_name, 'custom-bins'):
            # check that there are not distribution options also set
            if any([cp.has_option(layer_name, opt) for opt in \
                    ['x-min', 'x-max', 'x-nbins', 'x-distr',
                     'y-min', 'y-max', 'y-nbins', 'y-distr']]):
                raise ValueError("config file specifies custom-bins, but " +
                    "also includes arguments related to creating a " +
                    "creating a distribution")
            # parse the bins
            x_lower_bounds, x_upper_bounds, y_lower_bounds, y_upper_bounds = \
                parse_custom_bins(cp.get(layer_name, 'custom-bins'))
            this_layer.set_bins('x', x_lower_bounds, x_upper_bounds, 'custom')
            this_layer.set_bins('y', y_lower_bounds, y_upper_bounds, 'custom')
        else:
            # set x bins
            x_min = float(cp.get(layer_name, 'x-min'))
            x_max = float(cp.get(layer_name, 'x-max'))
            x_nbins = int(cp.get(layer_name, 'x-nbins'))
            x_distr = cp.get(layer_name, 'x-distr')
            this_layer.create_bins('x', x_min, x_max, x_nbins, x_distr)
            # ditto y
            y_min = float(cp.get(layer_name, 'y-min'))
            y_max = float(cp.get(layer_name, 'y-max'))
            y_nbins = int(cp.get(layer_name, 'y-nbins'))
            y_distr = cp.get(layer_name, 'y-distr')
            this_layer.create_bins('y', y_min, y_max, y_nbins, y_distr)
        # set plotting options
        if cp.has_option(layer_name, 'plot-x-min'):
            this_layer.plot_x_min = float(cp.get(layer_name, 'plot-x-min'))
        if cp.has_option(layer_name, 'plot-x-max'):
            this_layer.plot_x_max = float(cp.get(layer_name, 'plot-x-max'))
        if cp.has_option(layer_name, 'plot-log-x'):
            this_layer.plot_log_x = True
        if cp.has_option(layer_name, 'plot-y-min'):
            this_layer.plot_y_min = float(cp.get(layer_name, 'plot-y-min'))
        if cp.has_option(layer_name, 'plot-y-max'):
            this_layer.plot_y_max = float(cp.get(layer_name, 'plot-y-max'))
        if cp.has_option(layer_name, 'plot-log-y'):
            this_layer.plot_log_y = True
        # set super layer and create phyper cubes
        if level > 0:
            this_layer.set_super_layer(layers[level-1])
        this_layer.create_cubes()
        this_layer.set_cube_layer_ids()
        # add to the list of layers
        layers.append(this_layer)

    # set sub layers
    for this_layer in layers[:-1]:
        this_layer.set_sub_layer(layers[this_layer.level+1])

    return layers


def get_all_cubes_in_layers(top_layer):
    """
    Returns a list of all of the phyper cubes in the hierarchy layer.
    """
    if top_layer.level != 0:
        raise ValueError("given top_layer does not have level 0")
    pcubes = [top_layer.parents[0]] + top_layer.all_children
    this_layer = top_layer.sub_layer
    while this_layer is not None:
        pcubes += this_layer.all_children
        this_layer = this_layer.sub_layer
    return pcubes


def get_file_checksum(filename):
    """
    Gets the SHA 256 checksum of the given file.
    """
    return hashlib.sha256(open(filename, 'rb').read()).hexdigest()


def get_layers_checksum(top_layer):
    """
    Returns a checksum describing the layer hierarchy. This is obtained
    by creating a string out of the x-arg, x-min, x-max, x-nbins, x-distr,
    and the same for y, from every layer in the layer hierarchy.
    The order of the parameter cubes is whatever is returned by
    get_all_cubes_in_layers.
    """
    this_layer = top_layer
    description = []
    while this_layer is not None:
        x_min, x_max = this_layer.x_lims
        y_min, y_max = this_layer.y_lims
        description.extend(map(str, [this_layer.x_param, x_min, x_max,
            this_layer.x_nbins, this_layer.x_distr,
            this_layer.y_param, y_min, y_max, this_layer.y_nbins]))
        this_layer = this_layer.sub_layer
    return hashlib.sha256(','.join(description)).hexdigest()


def write_plot_cache(cachename, top_layer, layer_config_file):
    """
    Creates a cache file of plots given the top layer of a layer hierarchy.
    The structure of the cache file is phyper_cube.id_in_layer plot_filename.
    The plot filenames are retrieved from each phyper_cube's list of
    additional_plots.  At the top of the cache file the location and a
    checksum of the layer hierarchy that was created by the layer_config_file
    is written. This is done so that later programs can check that the layers
    are the same. 
    """
    f = open(cachename, 'w')
    # we'll write write an hash checksum of the config file at the top of the
    # cache file, along with the name of the cache file, so later programs
    # can check
    print >> f, "#from: %s" %(os.path.abspath(layer_config_file))
    print >> f, "#checksum: %s" %(get_layers_checksum(top_layer))
    # now print the filenames along with the index
    for pcube in get_all_cubes_in_layers(top_layer):
        level_num, child_idx = pcube.id_in_layer
        for fname in pcube.additional_plots:
            print >> f, "%i %i %s" %(level_num, child_idx,
                os.path.abspath(fname))
    f.close()


def load_plot_cache(cachename):
    """
    Loads the given plot cache. A dictionary is returned indexed by the
    parameter cubes' id_in_layer.
    """
    f = open(cachename, 'r')
    filepath = f.readline().split(' ')[1].rstrip('\n')
    checksum = f.readline().split(' ')[1].rstrip('\n')
    plot_files = {}
    for line in f:
        level_num, child_idx, fname = line.split(' ')
        fname = fname.rstrip('\n')
        dictkey = (int(level_num), int(child_idx))
        plot_files.setdefault(dictkey, [])
        plot_files[dictkey].append(fname)
    return filepath, checksum, plot_files
