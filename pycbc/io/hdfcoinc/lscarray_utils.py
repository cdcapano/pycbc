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

from pycbc.io import lscarrays
from pycbc.io.ligolw import lscarray_utils as ligolw_utils
from pycbc import distributions
from pycbc.events import newsnr

from glue import segments


#
# =============================================================================
#
#                           File-list manipulation
#
# =============================================================================
#
def parse_cache_file(cache_file):
    """
    Parses the given cache file for file names.

    Parameters
    ----------
    cache_file: string
        The name of the cache file. Should be text file containing a simple
        list of the filenames.

    Returns
    -------
    filenames: list
        The list of files found in the cache file.
    """
    f = open(cache_file, 'r')
    filenames = [line.rstrip('\n') for line in f]
    f.close()
    return filenames


def get_filetypes_from_filelist(filenames, filetype, tag=None,
        include_ifos=False):
    """
    Searches filenames for the given filetype and tag. If ``tag`` is
    specified, the pattern searched for is:
    ``'([A-Z0-9]*)-(filetype).*(tag).*'``. In that case there are three
    groups: the ifos (first), the filetype (second), and the tag (third). If
    ``tag`` is ``None``, the tag is dropped, in which case there are only two
    groups matched.
    """
    if tag is not None:
        pattern = '([A-Z0-9]*)-(%s).*(%s).*' %(filetype, tag)
    else:
        pattern = '([A-Z0-9]*)-(%s).*' %(filetype)
    match = re.compile(pattern)
    results = {}
    for filename in filenames:
        fnmatch = match.match(os.path.basename(filename))
        if fnmatch is not None and tag is not None and not tag.startswith('?'):
            ifos, ftype, ftag = fnmatch.groups()
            try:
                typedict = results[ftype]
            except KeyError:
                typedict = results[ftype] = {}
            try:
                flist = results[ftype][ftag]
            except KeyError:
                flist = results[ftype][ftag] = []
            if include_ifos:
                flist.append((ifos, filename))
            else:
                flist.append(filename)
        elif fnmatch is not None:
            ifos, ftype = fnmatch.groups()
            try:
                flist = results[ftype]
            except KeyError:
                flist = results[ftype] = []
            if include_ifos:
                flist.append((ifos, filename))
            else:
                flist.append(filename)
    return results

def get_bank_files_from_filelist(filenames, tag=None):
    """
    Gets all BANK2HDF files in the given filelist. If a tag is provided, this
    will only search for the given tag.
    """
    ftype = 'BANK2HDF'
    return get_filetypes_from_filelist(filenames, ftype, tag=tag)[ftype]


def get_statmap_files_from_filelist(filenames, tag=None):
    """
    Gets all STATMAP files in the given filelist. If a tag is provided, this
    will only search for the given tag.
    """
    ftype = 'STATMAP'
    return get_filetypes_from_filelist(filenames, ftype, tag=tag)[ftype]


def get_hdfinjfind_files_from_filelist(filenames, tag=None):
    """
    Gets all the HDFINJFIND files from a list of files.  If a tag is provided,
    this will only search for the given tag.
    """
    ftype = 'HDFINJFIND'
    return get_filetypes_from_filelist(filenames, ftype, tag=tag)[ftype]


def get_injection_files_from_filelist(filenames, tag=None):
    """
    Gets all the HDFINJFIND files from a list of files.  If a tag is provided,
    this will only search for the given tag.
    """
    ftype = 'INJECTIONS'
    return get_filetypes_from_filelist(filenames, ftype, tag=tag)[ftype]


def map_hdfinjfind_to_injfiles(hdfinjfind_files, injfiles):
    """
    FIXME: Right now this has to use file names to do the mapping, which is
    very rickety.
    """
    filemap = {}
    # cycle over the injfiles, finding the match
    for this_injfile in injfiles:
        simtag = os.path.basename(this_injfile).split('-')[1].replace(
            'INJECTIONS_', '')
        pattern = '_'+simtag
        # now find the match in the list of hdfinjfind files
        matching_file = [hdf_file for hdf_file in hdfinjfind_files \
            if re.search(pattern, hdf_file) is not None]
        if len(matching_file) == 0:
            raise ValueError('no hdf file found for injection file %s' %(
                this_injfile))
        # make sure mappings are one-to-one
        if len(matching_file) > 1:
            raise ValueError('more than one hdf file found that matches ' + \
                'injection file %s' %(this_injfile))
        matching_file = matching_file[0]
        if matching_file in filemap:
            raise ValueError('more than injection file matches hdf file %s' %(
                matching_file))
        filemap[matching_file] = this_injfile
    # check that every hdf file has a match
    for hdffile in hdfinjfind_files:
        # we'll only raise an error for non-ALLINJ files
        if hdffile not in filemap and re.search('ALLINJ', hdffile) is not None:
            raise ValueError('could not find an injection file for %s' %(
                hdffile))
    return filemap

def map_injfiles_to_hdfinjfind(hdfinjfind_files, injfiles):
    """
    The inverse of map_hdfinjfind_to_injfiles.
    """
    filemap = map_hdfinjfind_to_injfiles(hdfinjfind_files, injfiles)
    return dict(zip(filemap.values(), filemap.keys()))

#
# =============================================================================
#
#                           Arrays from files
#
# =============================================================================
#
def tmplt_inspiral_from_bankhdf(bankhdf, names=None):
    """
    Given a bank hdf file, converts into a TmpltInspiral array.

    Paramters
    ---------
    bankhdf: h5py File
        An open hdf file containing the bank to load.
    names: {None | (list of) strings}
        Only get the fields with the specified names.
    """
    if names is None:
        names = bankhdf.keys()
    elif isinstance(names, str) or isinstance(names, unicode):
        names = [names]
    arrays = [bankhdf[field] for field in names]
    # if the bank file has no tmplt_id, add it
    if 'template_id' not in bankhdf.keys():
        names.append('template_id')
        arrays.append(numpy.arange(len(bankhdf.values()[0])))
    templates = lscarrays.TmpltInspiral.from_arrays(arrays, names=names)
    # add the source filename
    templates.add_source_file(bankhdf.filename, 'template_id')
    return templates


def statmap_data_by_cev_name(name, statmapdata):
    """
    Given the name of a field in a CoincEvent array, retrieves the associated
    value from a data group in a statmap file.
    """
    if name == 'ifap':
        return 1./statmapdata['fap'].value
    elif name == 'ifap_exc':
        return 1./statmapdata['fap_exc'].value
    elif name == 'ranking_stat':
        return statmapdata['stat']
    elif name == 'event_id':
        return numpy.arange(len(statmapdata['fap']))
    else:
        # all of the others are one-to-one mappings
        return statmapdata[name]
    

def coinc_events_from_statmap(statmaphdf, datatype='foreground', names=None):
    """
    Given a STATMAP hdf file, converts into a CoincEvent array.

    Parameters
    ----------
    statmaphdf: h5py File
        An open STATMAP hdf file.
    datatype: {'foreground' | 'background'}
        Which type of data to get. For full data STATMAP files this can be set
        to either "foreground" or "background". For injection STATMAP files,
        only "foreground" is available. Default is "foreground".
    names: {None | (list of) strings}
        Only get fields with the specified names. The names should be names
        of the coinc_event fields.
    """
    data = statmaphdf[datatype]
    detectors = {det: detname for (det, detname) in statmaphdf.attrs.items() \
        if det.startswith('detector')}
    if names is None:
        names = ['ifap', 'ifar', 'ifap_exc', 'ifar_exc', 'ranking_stat',
            'template_id', 'event_id']
    if isinstance(names, str) or isinstance(names, unicode):
        names = [names]
    # ensure event_id is created
    if 'event_id' not in names:
        names.append('event_id')
    coincs = lscarrays.CoincEvent(len(data['fap']), name=datatype,
        detectors=detectors.values(), names=names)
    # copy the data over
    for name in names:
        coincs[name] = statmap_data_by_cev_name(name, data)
    # copy detector data over
    for det,detname in detectors.items():
        coincs[detname]['event_id'] = data['trigger_id%s' % det[-1]]
        coincs[detname]['end_time_s'], coincs[detname]['end_time_ns'] = \
            lscarrays.end_time_from_float(data['time%s' % det[-1]].value)
    # add the source filename
    coincs.add_source_file(statmaphdf.filename, 'event_id')
    return coincs
    

def triggermerge_data_by_sev_name(name, mergedata):
    """
    Given the name of a field in a SnglEvent array, retrieves the associated
    value from a data group in a trigger merge file.
    """
    if name == "sigmasq":
        return numpy.sqrt(mergedata['sigmasq'])
    elif name == "chisq_dof":
        # we'll store actual degrees of freedom, not the number of bins
        return 2*mergedata[name] - 2
    elif name == "ranking_stat":
        return newsnr(mergedata['snr'],
            mergedata['chisq']/(2*mergedata['chisq_dof']-2))
    elif name == "event_id":
        return numpy.arange(len(mergedata["snr"]))
    else:
        return mergedata[name]


def sngl_events_from_triggermerge(mergehdf, names=None):
    """
    Given a TRIGGER_MERGE file, converts into a SnglEvent array.

    Parameters
    ----------
    mergehdf: h5py File
        An open TRIGGER_MERGE file.
    names: {None | (list of) strings}
        Only get fields with the specified names.

    Returns
    -------
    sngl_event_array: SnglEvent array
        A SnglEvent array with the trigger merge data.
    """
    if names is None:
        names = [name for name in lscarrays.SnglEvent.default_fields().keys() \
            if name != 'process_id']
    if isinstance(names, str) or isinstance(names, unicode):
        names = [names]
    # ensure event_id is created
    if 'event_id' not in names:
        names.append('event_id')
    sngls = None
    for detector in mergehdf.keys():
        mergedata = mergehdf[detector]
        these_sngls = lscarrays.SnglEvent(len(mergedata[names[0]]),
            names=names) 
        for name in names:
            if name == 'detector' or name == 'ifo':
                these_sngls[name] = detector
            else:
                these_sngls[name] = triggermerge_data_by_sev_name(name,
                    mergedata)
        if sngls is None:
            sngls = these_sngls
        else:
            # flatten into a single array
            sngls = sngls.append(these_sngls, remap_indices='event_id')
    # add the source filename
    sngls.add_source_file(mergehdf.filename, 'event_id')
    return sngls


def sim_inspiral_from_filelist(filenames, tag=None,
        injection_fields=None, detectors=['H1', 'L1', 'V1'],
        load_recovered_fields=True, load_recovered_param_fields=True,
        load_recovered_sngl_fields=False,
        load_volume_weights=True, load_param_distribution=True,
        verbose=False):
    """
    Given a list of filenames, loads the injections as a sim_inspiral array.
    """
    # we'll get the injection information from injection xml files, since
    # these also have information about the distributions used
    if tag is not None:
        tag = '?%s' %(tag)
    injection_files = get_injection_files_from_filelist(filenames,
        tag=tag)
    if injection_fields is not None:
        if isinstance(injection_fields, str) or \
                isinstance(injection_fields, unicode):
            injection_fields = [injection_fields]
        # make sure simulation_id is included in injection fields
        if 'simulation_id' not in injection_fields:
            injection_fields.append('simulation_id')
    # if we're loading found events, get the corresponding hdfinjfind file
    if load_recovered_fields:
        # get the names of the fields to load
        if load_recovered_fields == True:
            # get everything
            recovered_names = None
        else:
            # just get the desired
            recovered_names = load_recovered_fields
        # get the files, exclude ALLINJ
        hdfinjfind_files = get_hdfinjfind_files_from_filelist(filenames,
            tag=tag)
        hdfinjfind_files = get_hdfinjfind_files_from_filelist(hdfinjfind_files,
            tag='?!ALLINJ')
        # map the two together
        inj_filemap = map_injfiles_to_hdfinjfind(hdfinjfind_files,
            injection_files)
    # if the found params are desired, we'll also need the bank file
    if load_recovered_param_fields:
        if not load_recovered_fields:
            raise ValueError("load_recovered_param_fields requires " +
                "load_recovered_fields")
        # ensure template id is in the recovered names
        if recovered_names is not None and \
                'template_id' not in recovered_names:
            recovered_names.append('template_id')
        if load_recovered_param_fields == True:
            param_names = None
        else:
            param_names = load_recovered_param_fields
        bank_files = get_bank_files_from_filelist(filenames)
        # FIXME: Can only handle a single bank file at the moment
        if len(bank_files) > 1:
            raise ValueError("found multiple bank files")
        # load the file as a tmplt_inspiral array
        bankhdf = h5py.File(bank_files[0], 'r')
        bank_array = tmplt_inspiral_from_bankhdf(bankhdf, names=param_names)
    # cycle over the injection files, loading into the sim array
    sim_array = None

    if verbose:
        print >> sys.stdout, "Loading from file:"
    for ii,inj_file in enumerate(injection_files):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(injection_files)),
            sys.stdout.flush()
        xmldoc, _ = ligolw_utils.load_xmldoc(inj_file)
        this_sim = ligolw_utils.sim_inspiral_from_xmldoc(xmldoc,
            names=injection_fields, detectors=detectors)
        if load_volume_weights:
            # get the volume weights from the process params
            pparams = ligolw_utils.process_params_from_xmldoc(xmldoc)
            this_sim = ligolw_utils.add_vol_weights_from_process_params(
                this_sim, pparams)
        if load_param_distribution:
            # get the parameters from the process params 
            pdistrs = ligolw_utils.load_distributions_from_process_params(
                inj_file)
            this_sim = ligolw_utils.add_parameter_distributions(this_sim,
                pdistrs)
        if load_recovered_fields:
            hdfinjfind_file = inj_filemap[inj_file]
            hdfinjfind = h5py.File(hdfinjfind_file, 'r')
            recovered_events = coinc_events_from_statmap(hdfinjfind,
                datatype='found', names=recovered_names)
            # if params are desired, load them
            if load_recovered_param_fields:
                recovered_events = recovered_events.expand_templates(
                    bank_array, get_fields=param_names)
            # add to this sim
            this_sim = this_sim.add_default_fields(['recovered'])
            # set what was recovered...
            recidx = hdfinjfind['found']['injection_index']
            this_sim['recovered']['event_id'][recidx] = \
                numpy.arange(recidx.size)
            # ...and expand
            this_sim = this_sim.expand_recovered(recovered_events)
        # add information about source files and set the indices to remap
        this_sim.add_source_file(inj_file, 'simulation_id')
        remap_indices = ['simulation_id', 'process_id']
        if load_recovered_fields:
            this_sim.add_source_file(hdfinjfind_file, 'recovered.event_id')
            remap_indices.append('recovered.event_id')
        if load_recovered_param_fields:
            # XXX: we do not try to remap the recovered.template_id, as it
            # is assumed that everything comes from the same bank; change this
            # when allow multiple banks
            this_sim.add_source_file(bankhdf.filename, 'recovered.template_id')
        # add to the global array
        if sim_array is None:
            sim_array = this_sim
        else:
            sim_array = sim_array.append(this_sim, remap_ids=remap_indices)
            if load_recovered_fields:
                hdfinjfind.close()
    if verbose:
        print >> sys.stdout, ""
    # close the bank file
    if load_recovered_param_fields:
        bankhdf.close()
    return sim_array


def get_livetime(statmap_files):
    """
    Cycles over the given statmap files, adding livetimes as it goes.
    """
    livetimes = segments.segmentlist([])
    for this_file in statmap_files:
        data = h5py.File(this_file, 'r')
        data_segs = numpy.zeros((data['segments']['coinc']['start'].len(), 2))
        data_segs[:,0] = data['segments']['coinc']['start']
        data_segs[:,1] = data['segments']['coinc']['end']
        seg_list = segments.segmentlist(map(tuple, data_segs))
        seg_list.coalesce()
        if seg_list.intersects(livetimes):
            raise ValueError("files have overlapping segment times")
        livetimes.extend(seg_list)
        livetimes.coalesce()
    return abs(livetimes)
