#! /usr/bin/env python

import os, sys
import h5py
import re
import numpy

from pycbc import distributions
from pycbc.plot import data_utils

from glue import segments

# stuff for converting xml into a sqlite database
import sqlite3
from glue.ligolw.utils import ligolw_sqlite
from glue.ligolw import dbtables
from glue.ligolw import ligolw
class ContentHandler(ligolw.LIGOLWContentHandler):
	pass
dbtables.use_in(ContentHandler)
# we'll use the utils from the sqlite module for loading the distance distr.
from pycbc.plot.pycbc_sqlite import get_r_distribution_from_inspinj, \
    get_dist_weights_from_inspinj_distr


def get_filetypes_from_filelist(filenames, match_pattern,
        exclude_pattern=None):
    """
    Pulls out files matching the given pattern, excluding ones that match the
    exlude_pattern (if provided).
    """
    matching_files = []
    for this_file in filenames:
        if re.search(match_pattern, this_file) is not None:
            if exclude_pattern is None or \
                    re.search(exclude_pattern, this_file) is None:
                matching_files.append(this_file)
    if len(matching_files) == 0:
        raise ValueError("found no files matching %s" %(match_pattern) +
            " and not matching %s" %(exclude_pattern) \
            if exclude_pattern is not None else "")
    return matching_files
    
def get_hdfinjfind_files_from_filelist(filenames):
    """
    Gets all the hdfinjfind files from a list of files, excluding ALLINJ.
    """
    pattern = '-HDFINJFIND'
    exclude_pattern = '-HDFINJFIND_ALLINJ'
    return get_filetypes_from_filelist(filenames, pattern, exclude_pattern)

def get_injection_files_from_filelist(filenames):
    """
    Gets all of the injection files from a list of files.
    """
    pattern = "HL-INJECTIONS"
    return get_filetypes_from_filelist(filenames, pattern)

def get_fulldata_statmap_file_from_filelist(filenames):
    """
    Gets the full data statmap files.
    """
    pattern = 'STATMAP_FULL_DATA'
    return get_filetypes_from_filelist(filenames, pattern)

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


def load_xml_as_memorydb(xmlfile):
    """
    Loads an xml file into memory as a sqlite database, and returns a
    connection to it. This allows the file to be used with functions written
    for sqlite.
    """
    connection = sqlite3.connect(':memory:')
    ContentHandler.connection = connection
    ligolw_sqlite.insert_from_url(xmlfile, contenthandler=ContentHandler,
        preserve_ids=True, verbose=False)
    dbtables.build_indexes(connection, False)
    return connection


def get_injection_results(injfind_filenames,
        include_missed_injections=True, sim_inspiral_files=[],
        load_inj_distribution=True, load_vol_weights_from_inspinj=True,
        verbose=False):
    results = []
    id_map = {}
    idx = 0
    inj_dists = {}
    # if loading injection distributions, create a map between the injfind
    # files and the injection files
    if load_inj_distribution or load_vol_weights_from_inspinj:
        filemap = map_hdfinjfind_to_injfiles(injfind_filenames,
            sim_inspiral_files)
    for ii,thisfile in enumerate(injfind_filenames):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(injfind_filenames)),
            sys.stdout.flush()
        if not thisfile.endswith('.hdf'):
            continue
        # if loading distributions, load the injection file into memory 
        if load_inj_distribution or load_vol_weights_from_inspinj:
            injfile = filemap[thisfile]
            connection = load_xml_as_memorydb(injfile)
            rdistrs = get_r_distribution_from_inspinj(connection)
            # there's only one process_id, so get it
            proc_id = rdistrs.keys()[0] 
            distr_type, distribution, r1, r2 = rdistrs[proc_id]
            # since we have it, we'll get the waveform approximant too
            approxs = dict([[int(simid.split(':')[-1]), apprx] \
                for simid, apprx in connection.cursor().execute(
                'select simulation_id, waveform from sim_inspiral')])
        # get the injection distribution information
        if load_inj_distribution:
            # we'll use the process_id retrieved from rdistrs
            inj_distribution = \
                distributions.get_inspinj_distribution(connection, proc_id)

        data = h5py.File(thisfile, 'r')
        # get the found injections
        injections = data['injections']
        found_inj = data['found_after_vetoes']
        missed_inj = data['missed/after_vetoes']
        all_inj = numpy.append(found_inj['injection_index'], missed_inj)
        transition_point = found_inj['injection_index'].len()
        for statidx,injidx in enumerate(all_inj):
            thisRes = data_utils.Result()
            # id information
            thisRes.unique_id = idx
            thisRes.database = thisfile 
            id_map[thisfile, injidx] = idx
            thisRes.event_id = None
            idx += 1
            # Set the injection parameters: we'll make the injection
            # the psuedo class so we can access its attributes directly
            thisRes.set_psuedoattr_class(thisRes.injection)
            thisRes.injection.simulation_id = injidx
            # FIXME: add the following info to hdfinjfind files
            if load_inj_distribution or load_vol_weights_from_inspinj:
                thisRes.injection.approximant = approxs[injidx]
            # ensure that m1 is always > m2
            mass1 = injections['mass1'][injidx]
            mass2 = injections['mass2'][injidx]
            if mass2 > mass1:
                thisRes.injection.mass1 = mass2
                thisRes.injection.mass2 = mass1
                thisRes.injection.spin1x = injections['spin2x'][injidx]
                thisRes.injection.spin1y = injections['spin2y'][injidx]
                thisRes.injection.spin1z = injections['spin2z'][injidx]
                thisRes.injection.spin2x = injections['spin1x'][injidx]
                thisRes.injection.spin2y = injections['spin1y'][injidx]
                thisRes.injection.spin2z = injections['spin1z'][injidx]
            else:
                thisRes.injection.mass1 = mass1
                thisRes.injection.mass2 = mass2
                thisRes.injection.spin1x = injections['spin1x'][injidx]
                thisRes.injection.spin1y = injections['spin1y'][injidx]
                thisRes.injection.spin1z = injections['spin1z'][injidx]
                thisRes.injection.spin2x = injections['spin2x'][injidx]
                thisRes.injection.spin2y = injections['spin2y'][injidx]
                thisRes.injection.spin2z = injections['spin2z'][injidx]
            thisRes.injection.distance = injections['distance'][injidx]
            thisRes.injection.inclination = injections['inclination'][injidx]
            thisRes.injection.ra = injections['longitude'][injidx]
            thisRes.injection.dec = injections['latitude'][injidx]
            # get the injection weights
            if load_vol_weights_from_inspinj:
                min_vol, vol_weight = get_dist_weights_from_inspinj_distr(
                    thisRes, distr_type, distribution,
                    r1, r2)
                thisRes.injection.min_vol = min_vol
                thisRes.injection.vol_weight = vol_weight
            # set the mass distribution if desired
            if load_inj_distribution:
                thisRes.injection.mass_distr = inj_distribution
            # FIXME: set the template parameters

            # statistics
            if statidx < transition_point:
                thisRes.new_snr = found_inj['stat'][statidx]
                thisRes.false_alarm_rate = 1./found_inj['ifar'][statidx]
                thisRes.false_alarm_rate_exc = 1./found_inj['ifar_exc'][
                    statidx]
                thisRes.false_alarm_probability = found_inj['fap'][statidx]
                thisRes.false_alarm_probability_exc = found_inj['fap_exc'][
                    statidx]
            else:
                thisRes.new_snr = 0.
                thisRes.false_alarm_rate = numpy.inf
                thisRes.false_alarm_rate_exc = numpy.inf
                thisRes.false_alarm_probability = numpy.inf
                thisRes.false_alarm_probability_exc = numpy.inf

            results.append(thisRes)

        data.close()
        if load_inj_distribution or load_vol_weights_from_inspinj:
            connection.close()

    return results, id_map


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
