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
import logging
from pycbc import events
from pycbc.io import lscarrays

#
# =============================================================================
#
#                           Helper functions
#
# =============================================================================
#
def load_hdf(hdffile):
    """If a file path is provied, opens the file for reading.

    Parameters
    ----------
    hdffile : string | unicode | other
        If a string or unicode, assumed to be a file path to an hdf file. In
        that case, the file is opened for reading. If anything else, just
        passes.
    
    Returns
    -------
    hdffile : open h5py.File | other
        If the input was a string or unicode, an open hdf file. Otherwise, just
        returns whatever the input was.
    ispath : bool
        Whether or not the input was a string or unicode.
    """
    ispath = isinstance(hdffile, str) or isinstance(hdffile, unicode)
    if ispath:
        hdffile = h5py.File(hdffile, 'r')
    return hdffile, ispath

def get_fields_to_load(arr, names):
    """Given an LSCArray or similar and a list of names, determines what fields
    need to be loaded. Names may be a list of fields, virtual fields, or method
    fields. If virtual fields or method fields, will determine what fields are
    needed for them.
    """
    logging.info('determining what fields are needed')
    needed_fields = set()
    for name in names:
        if name in arr.all_fieldnames:
            needed_fields.update([name])
        else:
            needed_fields.update(lscarrays.get_needed_fieldnames(arr,
                    lscarrays.get_fields_from_arg(name)))
    logging.info('loading fields %s' %(', '.join(names)))
    return list(needed_fields)

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
        if isinstance(hdffile[key], h5py.Group) or \
                isinstance(hdffile[key], dict):
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
    bankhdf : file path, open h5py.File or similar
        A file path to a BANKHDF file, an open BANKHDF file, or a dictionary of
        similar structure, containing the template bank that was used to
        generate the events. If a file path, the file will be opened and
        closed.
    names : {None | (list of) strings}
        Only get fields with the specified names. May be fields, virtual fields,
        or method fields. If any virtual fields or method fields are listed,
        the fields needed for them will be loaded.

    Returns
    -------
    lscarrays.TmpltInspiral
        An instance of an lscarrays.TmpltInspiral array, with the desired data
        loaded from the bankhdf file.
    """
    bankhdf, bankispath = load_hdf(bankhdf)
    if names is None:
        names = bankhdf.keys()
    else:
        if isinstance(names, str) or isinstance(names, unicode):
            names = [names]
        # determine what fields to load; we'll need a dummy blank array to parse
        dummy_arr = dummy_tmplt_inspiral_from_bankhdf(bankhdf)
        names = get_fields_to_load(dummy_arr, names)
    # ensure template_id is included
    if 'template_id' not in names:
        names.append('template_id')
    arrays = []
    logging.info('loading templates')
    for name in names:
        # if the bank file has no template_id, add it by hand
        if name == 'template_id' and name not in bankhdf.keys():
            arrays.append(numpy.arange(len(bankhdf.values()[0])))
        else:
            arrays.append(bankhdf[name])
    templates = lscarrays.TmpltInspiral.from_arrays(arrays, names=names)
    # add the source filename
    templates.add_source_file(bankhdf.filename, 'template_id')
    if bankispath:
        bankhdf.close()
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
    bankhdf : file path | h5py.File
        A file path to a BANKHDF file or an open BANKHDF file. If a file path,
        the file will be opened and closed.

    Returns
    -------
    lscarrays.TmpltInspiral
        An instance of an lscarrays.TmpltInspiral array, with all of the
        possible fields that can be loaded.
    """
    logging.info('creating dummy tmplt_inspiral array')
    # temporarily silence logging
    logger = logging.getLogger()
    loglevel = logger.level
    logger.level = logging.WARN
    bankhdf, bankispath = load_hdf(bankhdf)
    _bankhdf = construct_inmemory_hdfstruct(bankhdf)
    if bankispath:
        bankhdf.close()
    dummy_arr = tmplt_inspiral_from_bankhdf(_bankhdf)
    # restore logging
    logger.level = loglevel
    return dummy_arr


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
        ranking_stat='newsnr', bankhdf=None, veto_file=None, segment_name=None,
        names=None):
    """
    Given a TRIGGER_MERGE file, converts into a SnglEvent array.

    Parameters
    ----------
    triggermergehdf : file path, open h5py.File or similar
        Either the file path to a TRIGGER_MERGE file, an open TRIGGER_MERGE
        file, or a dictionary of similar structure. If a file path, the file
        will be opened and closed. If an open file, the file will not be closed.
    detectors : {None | (list of) strings}
        The names of the detectors to load. If None, all of the detectors found
        in the TRIGGER_MERGE file will be loaded.
    ranking_stat : {'newsnr' | string}
        The ranking stat to compute for the ranking-stat column. Only used if
        names is None or 'ranking_stat' is in the list of names. See
        ```known_ranking_stats``` for possible options; default is 'newsnr'.
    bankhdf : {None | file path, open h5py.File or similar}
        A file path to a BANKHDF file, an open BANKHDF file, or a dictionary of
        similar structure, containing the template bank that was used to
        generate the events. If a file path, the file will be opened and
        closed. If provided, the template information stored in the file will
        be added to the SnglEvent array (via the ```expand_templates```
        method). 
    names : {None | (list of) strings}
        Only get fields with the specified names. May be fields, virtual fields,
        or method fields. If any virtual fields or method fields are listed,
        the fields needed for them will be loaded.

    Returns
    -------
    lscarrays.SnglEvent
        A SnglEvent array with the trigger merge data.
    """
    triggermergehdf, ispath = load_hdf(triggermergehdf)
    if bankhdf is not None:
        bankhdf, bankispath = load_hdf(bankhdf)
    # load vetoes
    if veto_file is not None and segment_name is None:
        raise ValueError('veto_file requires segment_name')
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
            if hdfname[1] in triggermergehdf[detectors[0]]] + ['ranking_stat']
        if bankhdf is not None:
            banknames = bankhdf.keys()
    else:
        # determine what fields to load; we'll need a dummy blank array to parse
        dummy_arr = dummy_sngl_events_from_triggermerge(triggermergehdf,
            bankhdf=bankhdf)
        names = get_fields_to_load(dummy_arr, names)
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
    # ensure end_time is in names if a veto file is specified
    if veto_file is not None and 'end_time' not in names:
        names.append('end_time')
    # convert end_time to end_time_s, end_time_ns
    if 'end_time' in names:
        names.pop(names.index('end_time'))
    names.append('end_time_s')
    names.append('end_time_ns')
    # load the data
    logging.info('loading events')
    sngls = None
    for detector in detectors:
        mergedata = triggermergehdf[detector]
        these_sngls = lscarrays.SnglEvent(len(mergedata[mergedata.keys()[0]]),
            ranking_stat_alias=ranking_stat, names=names)
        for name in names:
            if name == 'detector' or name == 'ifo':
                these_sngls[name] = detector
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
                needed_names, statfunc = known_ranking_stats[ranking_stat]
            except KeyError:
                raise ValueError("unrecognized ranking-stat %s" % ranking_stat)
            # populate the needed args
            needed_args = {}
            for arg in needed_names:
                if arg in names:
                    # can get from the array
                    needed_args[arg] = these_sngls[arg]
                else:
                    # have to get from the file
                    needed_args[arg] = triggermerge_data_by_sev_name(name,
                        mergedata)
            # populate the array
            these_sngls[name] = statfunc(**needed_args)
        logging.info('loaded %i %s events' %(these_sngls.size, detector))
        # apply vetos if specified
        if veto_file is not None:
            logging.info('applying vetoes')
            mask, _ = events.veto.indices_outside_segments(
                these_sngls['end_time'],
                [veto_file], ifo=detector, segment_name=segment_name)
            these_sngls = these_sngls[mask]
            logging.info('%i events survive vetoes' %(these_sngls.size))
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
        logging.info('expanding templates')
        sngls = sngls.expand_templates(templates,
            assume_this_array_sorted=True, assume_templates_sorted=True)
        sngls.add_source_file(bankhdf.filename, 'template_id')
    # close files if paths were provided
    if ispath:
        triggermergehdf.close()
    if bankhdf is not None and bankispath:
        bankhdf.close()
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
    logging.info('creating dummy sngl_events array')
    # temporarily silence logging
    logger = logging.getLogger()
    loglevel = logger.level
    logger.level = logging.WARN
    # load structures
    triggermergehdf, ispath = load_hdf(triggermergehdf)
    _triggermergehdf = construct_inmemory_hdfstruct(triggermergehdf)
    if ispath:
        triggermergehdf.close()
    if bankhdf is not None:
        bankhdf, bankispath = load_hdf(bankhdf)
        _bankhdf = construct_inmemory_hdfstruct(bankhdf)
        if bankispath:
            bankhdf.close()
    else:
        _bankhdf = None
    dummy_arr = sngl_events_from_triggermerge(_triggermergehdf,
        bankhdf=_bankhdf)
    # restore logging
    logger.level = loglevel
    return dummy_arr

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
    bankhdf : {None | file path, open h5py.File or similar}
        A file path to a BANKHDF file, an open BANKHDF file, or a dictionary of
        similar structure, containing the template bank that was used to
        generate the events. If a file path, the file will be opened and
        closed. If provided, the template information stored in the file will
        be added to the CoincEvent array (via the ```expand_templates```
        method). Default is None, in which case no template information will
        be added.
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
    # load data
    statmaphdf, ispath = load_hdf(statmaphdf)
    if bankhdf is not None:
        bankhdf, bankispath = load_hdf(bankhdf)
    if triggermergehdfs is not None:
        trigs_are_paths = [False]*len(triggermergehdfs)
        for ii,thishdf in enumerate(triggermergehdfs):
            triggermergehdfs[ii], trigs_are_paths[ii] = load_hdf(thishdf)
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
    # if triggermerge filess are provided, figure out which file goes with which
    # detector
    if triggermergehdfs is not None:
        sngls_filemap = {det: thisfile \
            for det in detectors \
            for thisfile in triggermergehdfs if det in thisfile.keys()}
    # parse the names
    if names is None:
        names = set([name for name,hdfname in cev_statmap_fieldmap.items() \
            if hdfname[1] in data])
        if bankhdf is not None:
            banknames = bankhdf.keys()
        if triggermergehdfs is not None:
            snglsnames = {det: sev_triggermerge_fieldmap.keys() \
                for det in detectors}
    else:
        if isinstance(names, str) or isinstance(names, unicode):
            names = [names]
        # determine what fields to load; we'll need a dummy blank array to parse
        dummy_arr = dummy_coinc_events_from_statmap(statmaphdf, datatype,
            bankhdf=bankhdf, triggermergehdfs=triggermergehdfs)
        names = set(get_fields_to_load(dummy_arr, names))
        # check for names that start with detector names;
        # detectors that are not specified will not be retrieved
        detectors = {det: detectors[det] for det in detectors \
            if any([name.startswith('%s.' %(det)) for name in names])}
        # parse what names correspond to what data file
        if bankhdf is not None:
            dummy_arr = dummy_tmplt_inspiral_from_bankhdf(bankhdf)
            banknames = set(dummy_arr.all_names) & names
            # remove the bank names from names
            names -= banknames
            # ensure that template_id is in both the banknames and the names
            # if we will be retrieving anything from the bank
            if banknames:
                names.update(['template_id'])
                banknames.update(['template_id'])
                banknames = list(banknames)
        if triggermergehdfs is not None:
            snglsnames = {
                det: set([name[3:] for name in names \
                            if name.startswith('%s.' %(det))])\
                for det in detectors
                }
            # remove and ensure that the detectors' event_id column is included
            for det in snglsnames:
                names -= set(['%s.%s' %(det, name) for name in snglsnames[det]])
                snglsnames[det].update(['event_id'])
                snglsnames[det] = list(snglsnames[det])
                names.update(['%s.event_id' % det])
    # ensure event_id is created
    names.update(['event_id'])
    # LSCArrays expects a list of names
    names = list(names)
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
        coincs = coincs.expand_templates(templates,
            assume_this_array_sorted=False, assume_templates_sorted=True)
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
    logging.info('creating dummy coinc_events array')
    # temporarily silence logging
    logger = logging.getLogger()
    loglevel = logger.level
    logger.level = logging.WARN
    # load structures
    statmaphdf, ispath = load_hdf(statmaphdf)
    _statmaphdf = construct_inmemory_hdfstruct(statmaphdf)
    if ispath:
        statmaphdf.close()
    if triggermergehdfs is not None:
        _triggermergehdfs = []
        for thishdf in triggermergehdfs:
            thishdf, ispath = load_hdf(thishdf)
            _triggermergehdfs.append(construct_inmemory_hdfstruct(thishdf))
            if ispath:
                thishdf.close()
    else:
        _triggermergehdfs = None
    if bankhdf is not None:
        bankhdf, bankispath = load_hdf(bankhdf)
        _bankhdf = construct_inmemory_hdfstruct(bankhdf)
        if bankispath:
            bankhdf.close()
    else:
        _bankhdf = None
    dummy_arr = coinc_events_from_statmap(_statmaphdf, datatype,
        bankhdf=_bankhdf, triggermergehdfs=_triggermergehdfs)
    # restore logging
    logger.level = loglevel
    return dummy_arr
