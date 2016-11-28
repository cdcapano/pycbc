#! /usr/bin/env python

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

"""Utilities to setup a parameter estimation run.
"""

import logging
import numpy
from argparse import ArgumentParser
from pycbc.psd import from_cli_multi_ifos as psd_from_cli_multi_ifos
from pycbc.strain import gate_data
from pycbc.strain import from_cli_multi_ifos as strain_from_cli_multi_ifos
from pycbc import waveform as _waveform
from pycbc.types import FrequencySeries
from pycbc.workflow import WorkflowConfigParser
from pycbc.inference import option_utils
from pycbc import DYN_RANGE_FAC

def select_waveform_generator(approximant):
    """ Returns the generator for the approximant.
    """
    if approximant in _waveform.fd_approximants():
        return _waveform.FDomainCBCGenerator
    elif approximant in _waveform.td_approximants():
        return _waveform.TDomainCBCGenerator
    elif approximant in _waveform.ringdown_fd_approximants:
        if approximant=='FdQNM':
            return _waveform.FDomainRingdownGenerator
        elif approximant=='FdQNMmultiModes':
            return _waveform.FDomainMultiModeRingdownGenerator
    elif approximant in _waveform.ringdown_td_approximants:
        raise ValueError("Time domain ringdowns not supported")
    else:
        raise ValueError("%s is not a valid approximant."%approximant)


def convert_liststring_to_list(lstring):
    """ Checks if an argument of the configuration file is a string of a list
    and returns the corresponding list (of strings)
    """
    if lstring[0]=='[' and lstring[-1]==']':
        lvalue = [str(lstring[1:-1].split(',')[n].strip().strip("'"))
                      for n in range(len(lstring[1:-1].split(',')))]
    return lvalue


#-----------------------------------------------------------------------------
#
#                   Configuration file utilities
#
#-----------------------------------------------------------------------------

def config_parser_from_cli(opts):
    """Loads a config file from the given options, applying any overrides
    specified. Specifically, config files are loaded from the `--config-files`
    options while overrides are loaded from `--config-overrides`.
    """
    # read configuration file
    logging.info("Reading configuration file")
    if opts.config_overrides is not None:
        overrides = [override.split(":") for override in opts.config_overrides]
    else:
        overrides = None
    return WorkflowConfigParser(opts.config_files, overrides)


def config_section_to_opts(cp, section):
    """Given a section in a config file, converts the specified options into
    strings as if they were on the command line. For example:
    
    .. code::
        [section_name]
        foo =
        bar = 10

    yields: `'--foo --bar 10'`.
    """
    opts = []
    for opt in cp.options(section):
        opts.append('--{}'.format(opt))
        val = cp.get(section, opt)
        if val != '':
            opts.append(val)
    return ' '.join(opts)
            

def inference_opts_from_config(cp, section, additional_opts=None):
    """Constructs an ArgumentParser.parse_args opts instance needed to
    construct a likelihood evaluator from a config file, as if the config file
    options were specified on the command line in for `pycbc_inference`.
    """
    optstr = config_section_to_opts(cp, section)
    if additional_opts is not None:
        optstr = '{} {}'.format(opts, ' '.join(additional_opts))
    # create the dummy parser
    parser = option_utils.add_likelihood_opts_to_parser(ArgumentParser())
    return parser.parse_args(optstr.split(' '))


def read_args_from_config(cp, section_group=None):
    """Given an open config file, loads the static and variable arguments to
    use in the parameter estmation run.

    Parameters
    ----------
    cp : WorkflowConfigParser
        An open config parser to read from.
    section_group : {None, str}
        When reading the config file, only read from sections that begin with
        `{section_group}_`. For example, if `section_group='foo'`, the
        variable arguments will be retrieved from section
        `[foo_variable_args]`. If None, no prefix will be appended to section
        names.

    Returns
    -------
    variable_args : list
        The names of the parameters to vary in the PE run.
    static_args : dict
        Dictionary of names -> values giving the parameters to keep fixed.
    """
    logging.info("loading arguments")
    if section_group is not None:
        section_prefix = '{}_'.format(section_group)
    else:
        section_prefix = ''

    # sanity check that each parameter in [variable_args] has a priors section
    variable_args = cp.options("{}variable_args".format(section_prefix))
    subsections = cp.get_subsections("{}prior".format(section_prefix))
    tags = numpy.concatenate([tag.split("+") for tag in subsections])
    if not any(param in tags for param in variable_args):
        raise KeyError("You are missing a priors section in the config file.")

    # get parameters that do not change in sampler
    static_args = dict([(key,cp.get_opt_tags(
        "{}static_args".format(section_prefix), key, []))
        for key in cp.options("{}static_args".format(section_prefix))])
    for key,val in static_args.iteritems():
        try:
            static_args[key] = float(val)
            continue
        except:
            pass
        try:
            static_args[key] = convert_liststring_to_list(val) 
        except:
            pass

    return variable_args, static_args



#-----------------------------------------------------------------------------
#
#                   Data conditioning utilities
#
#-----------------------------------------------------------------------------

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
        logging.info("Gating {} strain".format(ifo))
        outdict[ifo] = gate_data(outdict[ifo], gates[ifo])
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


def data_from_cli(opts):
    """Loads data needed for an likelihood evaluator from the given
    command-line options. Gates specifed on the command line are also applied.

    Parameters
    ----------
    opts : ArgumentParse parsed args
        Argument options parsed from a command line string (the sort of thing
        returned by `parser.parse_args`).

    Returns
    -------
    strain_dict : dict
        Dictionary of instruments -> `TimeSeries` strain.
    stilde_dict : dict
        Dictionary of instruments -> `FrequencySeries` strain.
    psd_dict : dict
        Dictionary of instruments -> `FrequencySeries` psds.
    """
    # get gates to apply
    gates = gates_from_cli(opts)
    psd_gates = psd_gates_from_cli(opts)

    # get strain time series
    strain_dict = strain_from_cli_multi_ifos(opts, opts.instruments,
                                             precision="double")
    # apply gates if not waiting to overwhiten
    if not opts.gate_overwhitened:
        logging.info("Applying gates to strain data")
        strain_dict = apply_gates_to_td(strain_dict, gates)

    # get strain time series to use for PSD estimation
    # if user has not given the PSD time options then use same data as analysis
    if opts.psd_start_time and opts.psd_end_time:
        logging.info("Will generate a different time series for PSD "
                     "estimation")
        psd_opts = opts
        psd_opts.gps_start_time = psd_opts.psd_start_time
        psd_opts.gps_end_time = psd_opts.psd_end_time
        psd_strain_dict = strain_from_cli_multi_ifos(psd_opts,
                                                    opts.instruments,
                                                    precision="double")
        # apply any gates
        logging.info("Applying gates to PSD data")
        psd_strain_dict = apply_gates_to_td(psd_strain_dict, psd_gates)

    elif opts.psd_start_time or opts.psd_end_time:
        raise ValueError("Must give --psd-start-time and --psd-end-time")
    else:
        psd_strain_dict = strain_dict


    # FFT strain and save each of the length of the FFT, delta_f, and
    # low frequency cutoff to a dict
    logging.info("FFT strain")
    stilde_dict = {}
    length_dict = {}
    delta_f_dict = {}
    low_frequency_cutoff_dict = {}
    for ifo in opts.instruments:
        stilde_dict[ifo] = strain_dict[ifo].to_frequencyseries()
        length_dict[ifo] = len(stilde_dict[ifo])
        delta_f_dict[ifo] = stilde_dict[ifo].delta_f
        low_frequency_cutoff_dict[ifo] = opts.low_frequency_cutoff

    # get PSD as frequency series
    psd_dict = psd_from_cli_multi_ifos(opts, length_dict, delta_f_dict,
                               low_frequency_cutoff_dict, opts.instruments,
                               strain_dict=psd_strain_dict, precision="double")

    # apply any gates to overwhitened data, if desired
    if opts.gate_overwhitened and opts.gate is not None:
        logging.info("Applying gates to overwhitened data")
        # overwhiten the data
        for ifo in gates:
            stilde_dict[ifo] /= psd_dict[ifo]
        stilde_dict = apply_gates_to_fd(stilde_dict, gates)
        # unwhiten the data for the likelihood generator
        for ifo in gates:
            stilde_dict[ifo] *= psd_dict[ifo]

    return strain_dict, stilde_dict, psd_dict


def write_data_to_output(fp, strain_dict=None, stilde_dict=None,
                         psd_dict=None, low_frequency_cutoff_dict=None,
                         group=None):
    """Writes the strain/stilde/psd to the given output file.

    Parameters
    ----------
    fp : InferenceFile
        An open `InferenceFile` handle to save the data to.
    strain_dict : {None, dict}
        A dictionary of strains. If None, no strain will be written.
    stilde_dict : {None, dict}
        A dictionary of stilde. If None, no stilde will be written.
    psd_dict : {None, dict}
        A dictionary of psds. If None, psds will be written.
    low_freuency_cutoff_dict : {None, dict}
        A dictionary of low frequency cutoffs used for each detector in
        `psd_dict`; must be provided if `psd_dict` is not None.
    group : {None, str}
        The group to write the strain to. If None, will write to the top
        level.
    """
    # save PSD
    if psd_dict is not None:
        if low_frequency_cutoff_dict is None:
            raise ValueError("must provide low_frequency_cutoff_dict if "
                             "saving psds to output")
        # apply dynamic range factor for saving PSDs since
        # plotting code expects it
        logging.info("Saving PSDs")
        psd_dyn_dict = {}
        for key,val in psd_dict.iteritems():
             psd_dyn_dict[key] = FrequencySeries(
                                        psd_dict[key] * DYN_RANGE_FAC**2,
                                        delta_f=psd_dict[key].delta_f)
        fp.write_psd(psds=psd_dyn_dict,
                     low_frequency_cutoff=low_frequency_cutoff_dict,
                     group=group)

    # save stilde
    if stilde_dict is not None:
        fp.write_stilde(stilde_dict, group=group)

    # save strain if desired
    if strain_dict is not None:
        fp.write_strain(strain_dict, group=group)

