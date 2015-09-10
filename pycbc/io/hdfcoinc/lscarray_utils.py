# Copyright (C) 2015  Collin Capano
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
This modules provides functions for creating lscarrays from hdf files created
by the hdfcoinc pipeline.
"""


import os, sys
import h5py
import re
import numpy
from pycbc import events
from pycbc.io import lscarrays

#
# =============================================================================
#
#                           Helper functions
#
# =============================================================================
#

def _identity(data):
    """Just passes the given data.
    """
    return data

def _invertfield(data):
    """Inverts the given field data.
    """
    # if data is an h5py dataset, we need to get the value in order to do
    # math on it
    if isinstance(data, h5py.Dataset):
        data = data.value
    return 1./data

def _sqrt(data):
    """Takes the square root of the given field data.
    """
    # if data is an h5py dataset, we need to get the value in order to do
    # math on it
    if isinstance(data, h5py.Dataset):
        data = data.value
    return numpy.sqrt(data)

def _chisqbins_to_dof(data):
    """Converts chisq number of bins to degrees of freedom.
    """
    # if data is an h5py dataset, we need to get the value in order to do
    # math on it
    if isinstance(data, h5py.Dataset):
        data = data.value
    return 2*data - 2

def _end_time_from_float(data):
    """Converts the end time float data into seconds and nanoseconds.
    """
    # if data is an h5py dataset, we need to get the value in order to do
    # math on it
    if isinstance(data, h5py.Dataset):
        data = data.value
    return lscarrays.end_time_from_float(data)

#
# Ranking stats wrappers: for populating the SnglEvent array, below
#
def _effsnr(snr, chisq, chisq_dof, fac=250.):
    """Wrapper around events.py's effsnr that takes chisq and chisq_dof
    separately. Note: chisq_dof is assumed to be actual degrees of freedom,
    not the number of bins.
    """
    return events.effsnr(snr, chisq/chisq_dof, fac=fac)

def _newsnr(snr, chisq, chisq_dof, q=6., n=2.):
    """Wrapper around events.py's newsnr that takes chisq and chisq_dof
    separately. Note: chisq_dof is assumed to be actual degrees of freedom,
    not the number of bins.
    """
    return events.newsnr(snr, chisq/chisq_dof, q=q, n=n)

# the known ranking stats, their requred arguments, and the function that
# computes them
known_ranking_stats = {
    'effsnr': (('snr', 'chisq', 'chisq_dof'), _effsnr),
    'newsnr': (('snr', 'chisq', 'chisq_dof'), _newsnr)
}


class DictWithAttrs(dict):
    """A dictionary that allows additional attributes to be set.
    """
    pass

def hdf_as_empty_dict(hdffile):
    """Given an open hdf file, constructs a dictionary in memory that has the
    same structure as the hdf file, but in which all of the data arrays have
    a single element set to 0. All groups present in the hdf file are searched
    recursively until the data sets are found.
    """
    data = DictWithAttrs()
    for key in hdffile.keys():
        if isinstance(hdffile[key], h5py.Group):
            data[key] = hdf_as_empty_dict(hdffile[key])
        else:
            arr = numpy.zeros(1, dtype=hdffile[key].dtype)
            data[key] = arr
    return data

def construct_inmemory_hdfstruct(hdffile):
    """Given an open hdf file, constructs a dictionary in memory that has the
    same structure as the hdf file, but in which all of the data arrays have
    length 1 and 0. Also adds the hdffile's attrs as an attribute of the dict.
    """
    data = hdf_as_empty_dict(hdffile)
    data.attrs = hdffile.attrs
    data.filename = hdffile.filename
    return data


#
# =============================================================================
#
#                           Arrays from files
#
# =============================================================================
#

#
#
#       TmpltInspiral <--> BANKHDF
#
#
def tmplt_inspiral_from_bankhdf(bankhdf, names=None):
    """
    Given a bank hdf file, converts into a TmpltInspiral array.

    Parameters
    ---------
    bankhdf : h5py.File or similar
        An open hdf file containing the bank to load, or a dictionary of
        similar structure.
    names : {None | (list of) strings}
        Only get the fields with the specified names.

    Returns
    -------
    lscarrays.TmpltInspiral
        An instance of an lscarrays.TmpltInspiral array, with the desired data
        loaded from the bankhdf file.
    """
    if names is None:
        names = bankhdf.keys()
    elif isinstance(names, str) or isinstance(names, unicode):
        names = [names]
    # ensure template_id is included
    if 'template_id' not in names:
        names.append('template_id')
    arrays = []
    for name in names:
        # if the bank file has no template_id, add it by hand
        if name == 'template_id' and name not in bankhdf.keys():
            arrays.append(numpy.arange(len(bankhdf.values()[0])))
        else:
            arrays.append(bankhdf[name])
    templates = lscarrays.TmpltInspiral.from_arrays(arrays, names=names)
    # add the source filename
    templates.add_source_file(bankhdf.filename, 'template_id')
    return templates


def dummy_tmplt_inspiral_from_bankhdf(bankhdf):
    """Given a bank hdf file, creates an empty TmpltInspiral array of length
    1. The array will have all of the possible fields that can be loaded from
    the bank file. This is useful for determining what fields (including
    virtual fields) are available for manipulation, and for parsing what
    fields need to be loaded given some input arguments (c.f.
    ```lscarrays.get_needed_fields```).

    Parameters
    ---------
    bankhdf : h5py.File
        An open BANKHDF file.

    Returns
    -------
    lscarrays.TmpltInspiral
        An instance of an lscarrays.TmpltInspiral array, with all of the
        possible fields that can be loaded.
    """
    bankhdf = construct_inmemory_hdfstruct(bankhdf)
    return tmplt_inspiral_from_bankhdf(bankhdf)


#
#
#       SnglEvent <--> TRIGGER_MERGE
#
#

# The following dictionary gives the mapping between SnglEvent field names
# and the field names in TRIGGER_MERGE files, along with the function needed to
# convert between the two
#
sev_triggermerge_fieldmap = {
    'snr': (_identity, 'snr'),
    'chisq': (_identity, 'chisq'),
    'chisq_dof': (_chisqbins_to_dof, 'chisq_dof'),
    'bank_chisq': (_identity, 'bank_chisq'),
    'bank_chisq_dof': (_identity, 'bank_chisq_dof'),
    'cont_chisq': (_identity, 'cont_chisq'),
    'cont_chisq_dof': (_identity, 'cont_chisq_dof'),
    'end_time': (_end_time_from_float, 'end_time'),
    'coa_phase': (_identity, 'coa_phase'),
    'template_duration': (_identity, 'template_duration'),
    'template_id': (_identity, 'template_id'),
}
# the following is just the inverse of the above dictionary; we drop the
# functions, as it is only intended to indicate what cev field can be created
# given a statmap field
triggermerge_sev_fieldmap = dict([[val[1], key] \
    for key,val in sev_triggermerge_fieldmap.items()])

def statmap_data_by_cev_name(name, statmapdata):
    """
    Given the name of a field in a CoincEvent array, retrieves the associated
    value from a data group in a statmap file.
    """
    try:
        func, statmapname = cev_statmap_fieldmap[name]
        return func(statmapdata[statmapname])
    except KeyError:
        if name == 'event_id':
            return numpy.arange(len(statmapdata[statmapdata.keys()[0]]))
        else:
            # unknown name, just try to retrieve it from the statmap data
            return statmapdata[name]

def triggermerge_data_by_sev_name(name, mergedata):
    """
    Given the name of a field in a SnglEvent array, retrieves the associated
    value from a data group in a trigger merge file.
    """
    try:
        func, triggermapname = sev_triggermerge_fieldmap[name]
        return func(mergedata[triggermapname])
    except KeyError:
        if name == 'event_id':
            return numpy.arange(len(mergedata['snr']))
        else:
            # unknown name, just try to retrieve it from the triggermerge data
            return mergedata[name]


def sngl_events_from_triggermerge(triggermergehdf, detectors=None,
        ranking_stat='newsnr', bankhdf=None, names=None):
    """
    Given a TRIGGER_MERGE file, converts into a SnglEvent array.

    Parameters
    ----------
    triggermergehdf : h5py.File or similar
        An open TRIGGER_MERGE file, or a dictionary of similar structure.
    detectors : {None | (list of) strings}
        The names of the detectors to load. If None, all of the detectors found
        in the TRIGGER_MERGE file will be loaded.
    ranking_stat : {'newsnr' | string}
        The ranking stat to compute for the ranking-stat column. Only used if
        names is None or 'ranking_stat' is in the list of names. See
        ```known_ranking_stats``` for possible options; default is 'newsnr'.
    bankhdf : {None | h5py.File or similar}
        An open hdf file, or a dictionary of similar structure, containing the 
        template bank that was used to generate the events. If provided, the
        template information stored in the file will be added to the SnglEvent
        array (via the ```expand_templates``` method). 
    names : {None | (list of) strings}
        Only get fields with the specified names.

    Returns
    -------
    lscarrays.SnglEvent
        A SnglEvent array with the trigger merge data.
    """
    # parse the detectors
    if isinstance(detectors, str) or isinstance(detectors, unicode):
        detectors = [detectors]
    if detectors is None:
        detectors = triggermergehdf.keys()
    else:
        # check that the specified detectors are in the file
        missing_detectors = [det for det in detectors \
            if det not in triggermergehdf.keys()]
        if any(missing_detectors):
            raise ValueError("detector(s) %s not found " %(
                ','.join(missing_detectors)) + 'in the trigger merge file')
    # parse the names
    if isinstance(names, str) or isinstance(names, unicode):
        names = [names]
    if names is None:
        names = [name for name,hdfname in sev_triggermerge_fieldmap.items() \
            if hdfname[1] in triggermergehdf[detectors[0]]] + 'ranking_stat'
        if bankhdf is not None:
            banknames = bankhdf.keys()
    else:
        # parse what names correspond to what data file
        if bankhdf is not None:
            banknames = [name for name in names if name in bankhdf.keys()]
            # remove the bank names from names
            names = [name for name in names if name not in banknames]
            # ensure that template_id is in both the banknames and the names
            if 'template_id' not in names:
                names.append('template_id')
            if 'template_id' not in banknames:
                banknames.append('template_id')
    # ensure event_id and detectors is created
    if 'event_id' not in names:
        names.append('event_id')
    if 'detector' not in names:
        names.append('detector')
    # convert end_time to end_time_s, end_time_ns
    if 'end_time' in names:
        names.pop(names.index('end_time'))
    names.append('end_time_s')
    names.append('end_time_ns')
    # load the data
    sngls = None
    for detector in detectors:
        mergedata = triggermergehdf[detector]
        these_sngls = lscarrays.SnglEvent(len(mergedata[names[0]]),
            ranking_stat_alias=ranking_stat, names=names) 
        for name in names:
            if name == 'detector' or name == 'ifo':
                these_sngls[name] = detector
            # end time needs to be treated separately
            #elif name == 'end_time':
            #    secs, nanosecs = triggermerge_data_by_sev_name(name, mergedata)
            #    these_sngls['end_time_s'] = secs
            #    these_sngls['end_time_ns'] = nanosecs
            elif name == 'end_time_s':
                secs, _ = triggermerge_data_by_sev_name('end_time', mergedata)
                these_sngls[name] = secs
            elif name == 'end_time_ns':
                _, ns = triggermerge_data_by_sev_name('end_time', mergedata)
                these_sngls[name] = ns
            elif name != 'ranking_stat': # we'll calculate this later
                these_sngls[name] = triggermerge_data_by_sev_name(name,
                    mergedata)
        # compute the ranking stat
        if 'ranking_stat' in names:
            try:
                needed_args, statfunc = known_ranking_stats[ranking_stat]
            except KeyError:
                raise ValueError("unrecognized ranking-stat %s" % ranking_stat)
            # populate the needed args
            for arg in needed_args:
                if arg in names:
                    # can get from the array
                    needed_args[arg] = these_sngls[arg]
                else:
                    # have to get from the file
                    needed_args[arg] = triggermerge_data_by_sev_name(name,
                        mergedata)
            # populate the array
            these_sngls[name] = statfunc(**neededargs)
        if sngls is None:
            sngls = these_sngls
        else:
            # flatten into a single array
            sngls = sngls.append(these_sngls, remap_indices='event_id')
    # add the source filename
    sngls.add_source_file(triggermergehdf.filename, 'event_id')
    # add the template information if desired
    if bankhdf is not None and banknames != []:
        templates = tmplt_inspiral_from_bankhdf(bankhdf, names=banknames)
        sngls = sngls.expand_templates(templates)
        sngls.add_source_file(bankhdf.filename, 'template_id')
    return sngls


def dummy_sngl_events_from_triggermerge(triggermergehdf, bankhdf=None):
    """Given a TRIGGER_MERGE hdf file, creates an empty SnglEvent array of
    length 1. The array will have all of the possible fields that can be
    loaded from the triggermerge file and (if provided) the bank file. This is
    useful for determining what fields (including virtual fields) are
    available for manipulation, and for parsing what fields need to be loaded
    given some input arguments (c.f.  ```lscarrays.get_needed_fields```).

    Parameters
    ---------
    triggermergehdf : h5py.File
        An open TRIGGER_MERGE file.
    bankhdf : {None | h5py.File}
        An open BANKHDF file.

    Returns
    -------
    lscarrays.SnglEvent
        An instance of an lscarrays.SnglEvent array with all of the possible
        fields that can be loaded.
    """
    if bankhdf is not None:
        bankhdf = construct_inmemory_hdfstruct(bankhdf)
    triggermergehdf = construct_inmemory_hdfstruct(triggermergehdf)
    return sngl_events_from_triggermerge(triggermergehdf, bankhdf=bankhdf)

#
#
#       CoincEvent <--> STATMAP
#
#

# The following dictionary gives the mapping between CoincEvent field names
# and the field names in STATMAP files, along with the function needed to
# convert between the two
#
cev_statmap_fieldmap = {
    'ifap': (_invertfield, 'fap'),
    'ifap_exc': (_invertfield, 'fap_exc'),
    'ifar': (_identity, 'ifar'),
    'ifar_exc': (_identity, 'ifar_exc'),
    'ranking_stat': (_identity, 'stat'),
    'template_id': (_identity, 'template_id')
}
# the following is just the inverse of the above dictionary; we drop the
# functions, as it is only intended to indicate what cev field can be created
# given a statmap field
statmap_cev_fieldmap = dict([[val[1], key] \
    for key,val in cev_statmap_fieldmap.items()])

def statmap_data_by_cev_name(name, statmapdata):
    """
    Given the name of a field in a CoincEvent array, retrieves the associated
    value from a data group in a statmap file.
    """
    try:
        func, statmapname = cev_statmap_fieldmap[name]
        return func(statmapdata[statmapname])
    except KeyError:
        if name == 'event_id':
            return numpy.arange(len(statmapdata[statmapdata.keys()[0]]))
        else:
            # unknown name, just try to retrieve it from the statmap data
            return statmapdata[name]

def coinc_events_from_statmap(statmaphdf, datatype, bankhdf=None,
        detectors=None, triggermergehdfs=None, names=None):
    """
    Given a STATMAP hdf file, converts into a CoincEvent array.

    Parameters
    ----------
    statmaphdf : h5py.File or similar
        An open STATMAP hdf file, or a dictionary of similar structure.
    datatype : string 
        Which type of data to get. For full data STATMAP files this can be set
        to either "foreground", "background", or "background_exc". For
        injection STATMAP files, only "foreground" is available.
    bankhdf : {None | h5py.File or similar}
        If an open BANK hdf file is provided, the template information will be
        added to the array. Default is None, in which case no template
        information will be added.
    detectors : {None | (list of) strings}
        What detectors to get single detector information for. All of the
        specified detectors must be in the ``statmaphdf``'s ```attrs```. If
        None, will retrieve information for all detectors specified in the
        ``statmaphdf``'s ```attrs```.
    triggermergehdfs : {None | list of open TRIGGER_MERGE hdf files}
        If provided, will retrieve additional single-detector information from
        the provided TRIGGER_MERGE hdf files. If None provided, only the
        single detector information present in the STATMAP file will be
        included. If a TRIGGER_MERGE file has information for a detector that
        is not in STATMAP's attrs, it is ignored. If ```detectors``` is not
        None, additional information will only be retrieved for the specified
        detectors.
    names : {None | (list of) strings}
        Only get fields with the specified names. The names should be names of
        the coinc_event fields. If names includes template fields, must
        provide a ```bankhdf``` file. Default is to load the fields specified
        in the keys of ```cev_statmap_fieldmap``` and (if it is provided) all
        of the fields from the bankhdf file.
    """
    data = statmaphdf[datatype]
    # parse the detectors
    all_detectors = {detname: det \
        for (det, detname) in statmaphdf.attrs.items() \
        if det.startswith('detector')}
    if detectors is not None:
        # check that all of the detectors are available
        missing_dets = [det for det in detectors \
            if det not in all_detectors]
        if any(missing_dets):
            raise ValueError("detector(s) %s not found in statmap file" %(
                ','.join(missing_dets)))
        detectors = {det: all_detectors[det] for det in detectors}
    else:
        detectors = all_detectors
    # if triggermerge hdfs are provided, figure out which file goes with which
    # detector
    if triggermergehdfs is not None:
        sngls_filemap = {det: thisfile \
            for det in detectors \
            for thisfile in triggermergehdfs if det in thisfile.keys()}
    # parse the names
    if isinstance(names, str) or isinstance(names, unicode):
        names = [names]
    if names is None:
        names = [name for name,hdfname in cev_statmap_fieldmap.items() \
            if hdfname[1] in data]
        if bankhdf is not None:
            banknames = bankhdf.keys()
        if triggermergehdfs is not None:
            snglsnames = {det: sev_triggermerge_fieldmap.keys() \
                for det in detectors}
    else:
        # check for names that start with detector names;
        # detectors that are not specified will not be retrieved
        detectors = {det: detectors[det] for det in detectors \
            if any([name.startswith('%s.' %(det)) for name in names])}
        # parse what names correspond to what data file
        if bankhdf is not None:
            banknames = [name for name in names if name in bankhdf.keys()]
            # remove the bank names from names
            names = [name for name in names if name not in banknames]
            # ensure that template_id is in both the banknames and the names
            # if we will be retrieving anything from the bank
            if banknames != []:
                if 'template_id' not in names:
                    names.append('template_id')
                if 'template_id' not in banknames:
                    banknames.append('template_id')
        if triggermergehdfs is not None:
            snglsnames = {det: name[3:] \
                for det in detectors \
                for name in names if name.startswith('%s.' %(det))}
            # remove
            names = [name for name in names \
                if not any([name.startswith('%s.' %(det))
                            for det in detectors])]
            # ensure that the detectors' event_id column is included
            for det in snglsnames:
                if 'event_id' not in snglsnames[det]:
                    snglsnames[det].append('event_id')
                if '%s.event_id' %(det) not in names:
                    names.append('%s.event_id' %(det))
    # ensure event_id is created
    if 'event_id' not in names:
        names.append('event_id')
    coincs = lscarrays.CoincEvent(len(data[data.keys()[0]]), name=datatype,
        detectors=detectors.keys(), names=names)
    # copy the data over
    for name in names:
        coincs[name] = statmap_data_by_cev_name(name, data)
    # copy detector data over
    for detname,det in detectors.items():
        coincs[detname]['event_id'] = data['trigger_id%s' % det[-1]]
        if 'end_time_s' or 'end_time_ns' in coincs[detname].fieldnames:
            end_time_s, end_time_ns = _end_time_from_float(
                data['time%s' % det[-1]])
            if 'end_time_s' in coincs[detname].fieldnames:
                coincs[detname]['end_time_s'] = end_time_s
            if 'end_time_ns' in coincs[detname].fieldnames:
                coincs[detname]['end_time_ns'] = end_time_ns
    # add template info
    if bankhdf is not None and banknames != []:
        templates = tmplt_inspiral_from_bankhdf(bankhdf, names=banknames)
        coincs = coincs.expand_templates(templates)
        coincs.add_source_file(bankhdf.filename, 'template_id')
    # add singles info
    if triggermergehdfs is not None and snglsnames != {}:
        for det in sngls_filemap:
            # note that we do not include the bankfile here, since it is
            # already expanded
            sngls = sngl_events_from_triggermerge(sngls_filemap[det],
                detectors=det, names=snglsnames[det])
            coincs = coincs.expand_sngls(sngls)
            coincs.add_source_file(sngls_filemap[det].filename,
                '%s.event_id' %(det))
    # add the source filename
    coincs.add_source_file(statmaphdf.filename, 'event_id')
    return coincs


def dummy_coinc_events_from_statmap(statmaphdf, datatype, bankhdf=None,
        triggermergehdfs=None):
    """Given a STATMAP hdf file, creates an empty CoincEvent array of length
    1. The array will have all of the possible fields that can be loaded from
    the STATMAP file and (if provided) the bank file and trigger-merge files.
    This is useful for determining what fields (including virtual fields) are
    available for manipulation, and for parsing what fields need to be loaded
    given some input arguments (c.f.  ```lscarrays.get_needed_fields```).

    Parameters
    ---------
    statmaphdf : h5py.File
        An open STATMAP file.
    datatype : string
        What datatype to check.
    bankhdf : {None | h5py.File}
        An open BANKHDF file.
    triggermergehdf : {None | list of h5py.Files}
        A list of open TRIGGER_MERGE files.

    Returns
    -------
    lscarrays.CoincEvent
        An instance of an lscarrays.CoincEvent array with all of the possible
        fields that can be loaded.
    """
    if bankhdf is not None:
        bankhdf = construct_inmemory_hdfstruct(bankhdf)
    if triggermergehdfs is not None:
        triggermergehdfs = [construct_inmemory_hdfstruct(thishdf) \
            for thishdf in triggermergehdfs]
    statmaphdf = construct_inmemory_hdfstruct(statmaphdf)
    return coinc_events_from_statmap(statmaphdf, datatype, bankhdf=bankhdf,
        triggermergehdfs=triggermergehdfs)
