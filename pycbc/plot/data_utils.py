#! /usr/bin/env python

import os, sys
import math
import numpy
import operator
import lal

from pycbc import pnutils

#############################################
#
#   Data Storage and Slicing tools
#
#############################################

class Template(object):
    """
    Class to store information about a template for plotting.
    """
    # we'll group the various parameters by type
    _intrinsic_params = [
        'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
        'spin2x', 'spin2y', 'spin2z', 'eccentricity',
        'lambda1', 'lambda2'
        ]
    _extrinsic_params = [
        'phi0', 'inclination', 'distance'
        ]
    _waveform_params = [
        'sample_rate', 'segment_length', 'duration',
        'f_min', 'f_ref', 'f_max',
        'axis_choice', 'modes_flag',
        'amp_order', 'phase_order', 'spin_order', 'tidal_order',
        'approximant', 'taper'
        ]
    _ifo_params = [
        'ifo', 'sigma'
        ]
    _tmplt_weight_params = ['weight', 'weight_function']
    _id_name = 'tmplt_id'
    __slots__ = [_id_name] + _intrinsic_params + _extrinsic_params + \
        _waveform_params + _ifo_params + _tmplt_weight_params

    def __init__(self, **kwargs):
        default = None
        [setattr(self, param, kwargs.pop(param, default)) for param in \
            self.__slots__]
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))

    # some other derived parameters
    @property
    def mtotal(self):
        return self.mass1 + self.mass2

    @property
    def mtotal_s(self):
        return lal.MTSUN_SI*self.mtotal

    @property
    def q(self):
        return self.mass1 / self.mass2

    @property
    def eta(self):
        return self.mass1*self.mass2 / self.mtotal**2.

    @property
    def mchirp(self):
        return self.eta**(3./5)*self.mtotal

    @property
    def chi(self):
        return (self.mass1*self.spin1z + self.mass2*self.spin2z) / self.mtotal

    # some short cuts
    @property
    def m1(self):
        return self.mass1

    @property
    def m2(self):
        return self.mass2

    def tau0(self, f0=None):
        """
        Returns tau0. If f0 is not specified, uses self.f_min.
        """
        if f0 is None:
            f0 = self.f_min
        return (5./(256 * numpy.pi * f0 * self.eta)) * \
            (numpy.pi * self.mtotal_s * f0)**(-5./3.)
   
    def v0(self, f0=None):
        """
        Returns the velocity at f0, as a fraction of c. If f0 is not
        specified, uses self.f_min.
        """
        if f0 is None:
            f0 = self.f_min
        return (2*numpy.pi* f0 * self.mtotal_s)**(1./3)

    @property
    def s1(self):
        return numpy.array([self.spin1x, self.spin1y, self.spin1z])

    @property
    def s1x(self):
        return self.spin1x

    @property
    def s1y(self):
        return self.spin1y

    @property
    def s1z(self):
        return self.spin1z

    @property
    def s2(self):
        return numpy.array([self.spin2x, self.spin2y, self.spin2z])

    @property
    def s2x(self):
        return self.spin2x

    @property
    def s2y(self):
        return self.spin2y

    @property
    def s2z(self):
        return self.spin2z

    @property
    def apprx(self):
        return self.approximant
    
    @property
    def final_frequency(self):
        # will get a runtime error for SpinTaylorTx; instead we'll use
        # SEOBNRpeak
        if self.approximant.startswith('SpinTaylorT'):
            return pnutils.frequency_cutoff_from_name('SEOBNRv2Peak',
                self.mass1, self.mass2, self.spin1z, self.spin2z)
        else:
            return pnutils.get_final_freq(str(self.approximant),
                self.mass1, self.mass2, self.spin1z, self.spin2z)

    @property
    def source_type(self):
        """
        Returns BNS, NSBH, or BBH depending on self's parameters.
        """
        if self.mass1 > 3. and self.mass2 > 3.:
            return 3 # must be BBH
        elif self.mass1 > 3. and self.mass2 < 2.:
            return 2 # must be NSBH
        elif self.mass1 < 2. and self.mass2 < 2.:
            return 1 # must be BNS
        else:
            return 4 # the overlapping region


class SnglIFOInjectionParams(object):
    """
    Class that stores information about an injeciton in a specific detector.
    """
    __slots__ = ['ifo', 'end_time', 'end_time_ns', 'sigma']
    def __init__(self, **kwargs):
        [setattr(self, param, kwargs.pop(param, None)) for param in \
            self.__slots__]
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))



class Injection(Template):
    """
    Class to store information about an injection for plotting. Inherits from
    Template, and adds slots for sky location.
    """
    # add information about location and time in geocentric coordinates, and
    # the distribution from which the injections were drawn
    _inj_params = [
        'geocent_end_time', 'geocent_end_time_ns', 'ra', 'dec', 'astro_prior',
        'min_vol', 'vol_weight', 'mass_distr', 'spin_distr'
        ]
    # we'll override some of the parameters in TemplateResult
    _id_name = 'simulation_id'
    # sngl_ifos will be a dictionary pointing to instances of
    # SnglIFOInjectionParams
    _ifo_params = ['sngl_ifos']
    __slots__ = [_id_name] + Template._intrinsic_params + \
        Template._extrinsic_params + Template._waveform_params + \
        _ifo_params + _inj_params
    def __init__(self, **kwargs):
        # ensure sngl_ifos is a dictionary
        self.sngl_ifos = kwargs.pop('sngl_ifos', {})
        # set the default for the rest to None
        [setattr(self, param, kwargs.pop(param, None)) for param in \
            self.__slots__ if param != 'sngl_ifos']
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))


class Result(object):
    """
    Class to store a template with an injection, along with event information
    (ranking stat, etc.) for purposes of plotting.

    Information about the template and the injection (masses, spins, etc.)
    are stored as instances of Template and Injection; since these contain
    static slots, this saves on memory if not all information is needed.

    Information about a trigger (snr, chisq, etc.) are stored as attributes
    of the class. Since the class has no __slots__, additional statistics
    may be added on the fly.

    The class can also be used to only store information about a trigger;
    i.e., adding an injection is not necessary.
    
    To make accessing intrinsic parameters easier, set the psuedoattr_class;
    this will make the __slots__ of either the injection or the template
    attributes of this class. See set_psuedoattr_class for details. 
    """
    _psuedoattr_class = None

    def __init__(self, unique_id=None, database=None, event_id=None,
            tmplt=None, injection=None):
        self.unique_id = None
        self.database = None
        self.event_id = None
        if tmplt is None:
            self.template = Template()
        else:
            self.template = tmplt
        if injection is None:
            self.injection = Injection()
        else:
            self.injection = injection
        self._psuedoattr_class = None
        self.ranking_stat = None
        self.snr = None
        self.chisq = None
        self.chisq_dof = None
        self.new_snr = None
        self.false_alarm_rate = None
        self.uncombined_far = None
        self.false_alarm_probability = None
        # experiment parameters
        self.instruments_on = None
        self.livetime = None
        # banksim parameters
        self.effectualness = None
        self.snr_std = None
        self.chisq_std = None
        self.new_snr_std = None
        self.num_samples = None

    def __getattr__(self, name):
        """
        This will get called if __getattribute__ fails. Thus, we can use
        this to access attributes of the psuedoattr_class, if it is set.
        """
        try:
            return object.__getattribute__(self,
                '_psuedoattr_class').__getattribute__(name)
        except AttributeError:
            raise AttributeError("'Result' object has no attribute '%s'" %(
                name))

    def __setattr__(self, name, value):
        """
        First tries to set the attribute in self. If name is not in self's
        dict, next tries to set the attribute in self._psuedoattr_class.
        If that fails with an AttributeError, it then adds the name to self's
        namespace with the associated value.
        """
        try:
            object.__getattribute__(self,
                '_psuedoattr_class').__setattr__(name, value)
        except AttributeError:
            object.__setattr__(self, name, value)

    @property
    def psuedoattr_class(self):
        return self._psuedoattr_class

    def set_psuedoattr_class(self, psuedo_class):
        """
        Makes the __slots__ of the given class visible to self's namespace.
        An error is raised if self and psuedo_class have any attributes
        that have the same name, as this can lead to unexpected behavior.

        Parameters
        ----------
        psuedo_class: {self.template|self.injection}
            An instance of a class to make visible. Should be either self's
            injection or template (but can be any instance of any class).
        """
        # check that there is no overlap
        attribute_overlap = [name for name in self.__dict__ \
            if name in psuedo_class.__slots__]
        if attribute_overlap != []:
            raise AttributeError(
                "attributes %s " %(', '.join(attribute_overlap)) + \
                "are common to self and the given psuedo_class. Delete " +\
                "these attributes from self if you wish to use the given " +\
                "psuedo_class.")
        self._psuedoattr_class = psuedo_class
        
    @property
    def optimal_snr(self):
        """
        Returns the quadrature sum of the inj_sigmas divided by the distance.
        """
        return numpy.sqrt((numpy.array([ifo.sigma \
            for ifo in self.injection.sngl_ifos.values()])**2.).sum()) / \
            self.injection.distance

    # some short cuts
    @property
    def tmplt(self):
        return self.template

    @property
    def inj(self):
        return self.injection


#
# Slicing utilities
#

# FIXME: dataUtils in pylal should be moved to pycbc, and the get_val in there
# used instead
def get_arg(row, arg):
    """
    Retrieves an arbitrary argument from the given row object. For speed, the
    argument will first try to be retrieved using getattr. If this fails (this
    can happen if a function of several attributes of row are requested),
    then Python's eval command is used to retrieve the argument. The argument
    can be any attribute of row, or functions of attributes of the row
    (assuming the relevant attributes are floats or ints). Allowed functions
    are anything in Python's math library. No other functions (including
    Python builtins) are allowed.

    Parameters
    ----------
    row: any instance of a Python class
        The object from which to apply the given argument to.
    arg: string
        The argument to apply.

    Returns
    -------
    value: unknown type
        The result of evaluating arg on row. The type of the returned value
        is whatever the type of the data element being retreived is.
    """
    try:
        return getattr(row, arg)
    except AttributeError:
        row_dict = dict([ [name, getattr(row,name)] for name in dir(row)])
        safe_dict = dict([ [name,val] for name,val in \
            row_dict.items()+math.__dict__.items() \
            if not name.startswith('__')])
        return eval(arg, {"__builtins__":None}, safe_dict)


def result_in_range(result, test_dict):
    cutvals = [(get_arg(result, criteria), low, high) \
        for criteria,(low,high) in test_dict.items()] 
    return not any(x < low or x >= high for (x, low, high) in cutvals)


def result_is_match(result, test_dict):
    try:
        matchvals = [(getattr(result, criteria), targetval) 
            for criteria,targetval in test_dict.items()]
    except:
        matchvals = [(get_arg(result, criteria), targetval) \
            for criteria,targetval in test_dict.items()]
    return not any(x != targetval for (x, tagetval) in matchvals)


def apply_cut(results, test_dict):
    return [x for x in results if result_in_range(x, test_dict)]

def slice_results(results, test_dict):
    return apply_cut(results, test_dict)


def parse_results_cache(cache_file):
    filenames = []
    f = open(cache_file, 'r')
    for line in f:
        thisfile = line.split('\n')[0]
        if os.path.exists(thisfile):
            filenames.append(thisfile)
        else:
            raise ValueError("file %s not found" %(thisfile))
    f.close()
    return filenames


def cull_injection_results(results, primary_arg='false_alarm_rate',
        primary_rank_by='max', secondary_arg='new_snr',
        secondary_rank_by='min'):
    """
    Given a list of injection results in which the injections are mapped to
    multiple singles, picks the more significant one based on the primary_arg.
    If the two events have the same value in the primary arg, the secondary
    arg is used.
    """
    # get the correct operator to use
    if not (primary_rank_by == 'max' or primary_rank_by == 'min'):
        raise ValueError("unrecognized primary_rank_by %s; " %(
            primary_rank_by) + 'options are "max" or "min"')
    if not (secondary_rank_by == 'max' or secondary_rank_by == 'min'):
        raise ValueError("unrecognized secondary_rank_by %s; " %(
            secondary_rank_by) + 'options are "max" or "min"')
    # find the repeated entries
    sorted_results = sorted(results,
        key=lambda x: int(x.simulation_id.split(':')[-1]))
    id_map = {}
    duplicates = {}
    this_count = 1
    for ii,this_result in enumerate(sorted_results):
        if ii+1 < len(sorted_results) and \
                sorted_results[ii+1].simulation_id == \
                this_result.simulation_id:
            this_count += 1
        elif this_count > 1:
            # pick out the loudest out of the repeated values
            this_group = sorted_results[ii-(this_count-1):ii+1]
            primaries = numpy.array([data_utils.get_arg(x, primary_arg) \
                for x in this_group])
            if primary_rank_by == 'min':
                keep_idx = numpy.where(primaries == primaries.min())[0]
            else:
                keep_idx = numpy.where(primaries == primaries.max())[0]
            if len(keep_idx) > 1:
                secondaries = numpy.array([data_utils.get_arg(this_group[jj],
                    secondary_arg) for jj in keep_idx])
                # note: this will just keep the first event if the secondaries
                # are equal
                if secondary_rank_by == 'min':
                    secondary_idx = secondaries.argmin()
                else:
                    secondary_idx = secondaries.argmax()
                keep_idx = keep_idx[secondary_idx]
            else:
                keep_idx = keep_idx[0]
            # set this_result to the desired one; also set this_count = 0 for
            # the next group
            this_result = this_group[keep_idx]
            this_count = 1
        if this_count == 1:
            id_map[this_result.database, this_result.simulation_id] = \
                this_result
    return id_map.values(), id_map

