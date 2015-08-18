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
This modules provides functions for creating lscarrays from ligolw tables.
"""

import sqlite3
import numpy

from pycbc import distributions
from pycbc.io import lscarrays

from glue.ligolw import ligolw
from glue.ligolw import utils
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from glue.ligolw.utils import ligolw_sqlite

class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass
class dbContentHandler(ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(LIGOLWContentHandler)
dbtables.use_in(dbContentHandler)


#
# =============================================================================
#
#                           Helper functions 
#
# =============================================================================
#
def create_xmldoc(tables):
    """
    Creates an xmldoc from the list of given LIGOLW tables.
    """
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    for table in tables:
        xmldoc.childNodes[0].appendChild(table)
    return xmldoc


def load_xmldoc(filename):
    """
    Loads an xml document into memory from the given filename.

    Parameters
    ----------
    filename: string
        Should end in "xml[.gz]" or ".sqlite".

    Returns
    -------
    xmldoc: ligolw XML doc
        A LIGOLW xml document tree.
    connection: {None|SQLite3 connection}
        If filename ended in '.sqlite', the connection to the sqlite database.
        Otherwise, None.
    """
    if filename.endswith('.sqlite'):
        connection = sqlite3.connect(filename)
        xmldoc = dbtables.get_xml(connection)
    else:
        xmldoc = utils.load_filename(filename,
            contenthandler=LIGOLWContentHandler,  gz=filename.endswith('.gz'))
        connection = None
    return xmldoc, connection 


def load_xml_as_memorydb(xmlfile):
    """
    Loads an xml file into memory as a sqlite database, and returns a
    connection to it. This allows the file to be used with functions written
    for sqlite.
    """
    connection = sqlite3.connect(':memory:')
    dbContentHandler.connection = connection
    ligolw_sqlite.insert_from_url(xmlfile, contenthandler=dbContentHandler,
        preserve_ids=True, verbose=False)
    dbtables.build_indexes(connection, False)
    return connection


def lscarray_from_ligolw(ligolw_table, ArrayClass, names2columns, names=None,
        default_fields=None):
    """
    Converts a ligolw table into an LSCArrayWithDefaults class.
    """
    if names is None:
        # don't try to create any names of subarrays
        names = [name for name in names2columns \
            if len(name.split('.')) == 1]
    elif isinstance(names, str) or isinstance(names, unicode):
        names = [names]
    # cast to the default fields
    if default_fields is not None:
        cast_to_dtypes = {
            names2columns[name]: \
            lscarrays.fields_from_names(default_fields, names=name).items()[0]
            for name in names}
    else:
        cast_to_dtypes = None
    return ArrayClass.from_ligolw_table(ligolw_table,
        columns=[names2columns[name] for name in names],
        cast_to_dtypes=cast_to_dtypes)



#
# =============================================================================
#
#                       ProcessParams <--> LSCArray
#
# =============================================================================
#
def process_params_from_table(process_params_table):
    """
    Loads a process_params table as an LSCArray.
    """
    return lscarrays.LSCArray.from_ligolw_table(process_params_table)

def process_params_from_xmldoc(xmldoc):
    """
    Loads the process_params table in the given xmldoc as an LSCArray.
    """
    pptable = table.get_table(xmldoc, 'process_params')
    return process_params_from_table(pptable)

def get_r_distribution_from_process_params(process_params):
    """
    Gets the distance distribution that was given to inspinj from a
    process_params array.
    """
    inspinj_params = process_params[numpy.where(
        process_params['program'] == 'inspinj')]
    proc_ids = numpy.unique(inspinj_params['process_id'])
    # create an array to store the distributions to
    distr_dtype = {
        'process_id': 'ilwd:char',
        'distance_type': 'S10', # whether chirp distance or distance
        'distribution': 'S20', # uniform, log, etc.
        'min_dist': float,
        'max_dist': float,
        }.items()
    rdistrs = lscarrays.LSCArray(proc_ids.size, dtype=distr_dtype)
    rdistrs['process_id'] = proc_ids
    for ii,proc_id in enumerate(rdistrs['process_id']):
        inspinj_args = inspinj_params[numpy.where(
            inspinj_params['process_id'] == proc_id)]
        # find the distance distribution
        dist_idx = numpy.where(inspinj_args['param'] == '--d-distr')[0]
        chirp_idx = numpy.where(inspinj_args['param'] == '--dchirp-distr')[0]
        if dist_idx.size != 0:
            rdistrs[ii]['distance_type'] = 'distance'
            rdistrs[ii]['distribution'] = inspinj_args['value'][dist_idx]
        elif chirp_idx.size != 0:
            rdistrs[ii]['distance_type'] = 'chirp_dist'
            rdistrs[ii]['distribution'] = 'uniform'
        else:
            raise ValueError("could not find --d-distr nor --dchirp-distr " +
                "for process_id %s" %(proc_id))
        # get the min/max distances
        get_idx = numpy.where(inspinj_args['param'] == '--min-distance')[0]
        if get_idx.size != 0:
            # convert kpc to Mpc
            rdistrs[ii]['min_dist'] = \
                inspinj_args['value'][get_idx].astype(float)/1000.
        get_idx = numpy.where(inspinj_args['param'] == '--max-distance')[0]
        if get_idx.size != 0:
            rdistrs[ii]['max_dist'] = \
                inspinj_args['value'][get_idx].astype(float)/1000.
    return rdistrs


def load_distributions_from_process_params(xmlfile):
    """
    Loads the mass distribution as a CBC distribution from the process_params
    table.

    FIXME: Currently this takes an xmldoc filename and loads it into memory
    as a sqlite database. This is because all of the loading functions in
    distributions.py expect sqlite databases. These should be updated to
    read a process_params LSCArray.
    """
    # load the xml document as a sqlite database
    connection = load_xml_as_memorydb(xmlfile)
    # get all of the process ids
    sqlquery = """
        select distinct process_id
        from process_params
        where program == "inspinj"
        """
    distrs = {}
    for (proc_id,) in connection.cursor().execute(sqlquery).fetchall():
        distrs[int(proc_id.split(':')[-1])] = \
            distributions.get_inspinj_distribution(connection, proc_id)
    connection.close()
    return distrs



#
# =============================================================================
#
#                           SimInspiral <--> SimInspiral
#
# =============================================================================
#

# dictionary mapping ligolw SimInspiral table valid column names to
# lscarray SimInspiral array fields
# Note: Any field in the ligolw SimInspiral table listed here has no direct
# counterpart in the SimInspiral array
SimInspiralColumns2ArrayNames = {
    "process_id": "process_id",
    "simulation_id": "simulation_id",
    "waveform": "approximant",
    "geocent_end_time": "geocent_end_time_s",
    "geocent_end_time_ns": "geocent_end_time_ns",
    "mass1": "mass1",
    "mass2": "mass2",
    "distance": "distance",
    "longitude": "ra",
    "latitude": "dec",
    "inclination": "inclination",
    "coa_phase": "phi_ref",
    "polarization": "polarization",
    "spin1x": "spin1x",
    "spin1y": "spin1y",
    "spin1z": "spin1z",
    "spin2x": "spin2x",
    "spin2y": "spin2y",
    "spin2z": "spin2z",
    "f_lower": "f_min",
    "f_final": "f_max",
    "eff_dist_h": "H1.eff_dist",
    "eff_dist_l": "L1.eff_dist",
    "eff_dist_v": "V1.eff_dist",
    "amp_order": "amp_order",
    "taper": "taper",
    }
# the inverse
SimInspiralArrayNames2Columns = dict(zip(
    SimInspiralColumns2ArrayNames.values(),
    SimInspiralColumns2ArrayNames.keys()))


def sim_inspiral_from_table(sim_inspiral_table, names=None,
        detectors=['H1', 'L1', 'V1']):
    """
    Given a LIGOLW sim_inspiral table, converts to a SimInspiral LSCArray.

    Parameters
    ----------
    sim_inspiral_table: LIGOLW SimInspiral table
        The sim_inspiral table to convert.
    names: {None | (list of) string(s)}
        The names of fields (according to the LSCArray) to retrieve. If None,
        will retrieve all fields given in the SimInspiralColumnsTable2Array
        dictionary.
    detectors: {['H1', 'L1', 'V1'] | (list of) detector(s)}
        Only retrieve site information for the given detectors. If None, will
        retrieve information for H1, L1, and V1.

    Returns
    -------
    sim_inspiral_array: SimInspiral LSCArray
    """
    if isinstance(detectors, str) or isinstance(detectors, unicode):
        detectors = [detectors]
    if detectors is None:
        detectors = []
    sim_inspiral = lscarray_from_ligolw(sim_inspiral_table,
        lscarrays.SimInspiral, SimInspiralArrayNames2Columns, names=names,
        default_fields=lscarrays.SimInspiral.default_fields(
            detectors=detectors))
    subarrays = []
    subarray_names = []
    for detector in detectors:
        param_name = '%s.eff_dist' % detector
        param_data = numpy.array([
            getattr(row, SimInspiralArrayNames2Columns[param_name]) \
            for row in sim_inspiral_table])
        subarrays.append(param_data)
        subarray_names.append(param_name)
    if subarrays != []:
        sim_inspiral = sim_inspiral.add_fields(subarrays, subarray_names)
        sim_inspiral.detectors = detectors
    return sim_inspiral


def sim_inspiral_from_xmldoc(xmldoc, **kwargs):
    """
    Loads the sim_inspiral table in the given xmldoc as an LSCArray. See
    sim_inspiral_from_table for information about other keyword
    arguments.
    """
    simtable = table.get_table(xmldoc, 'sim_inspiral')
    return sim_inspiral_from_table(simtable, **kwargs)


def add_vol_weights_from_process_params(sim_inspiral_array,
        process_params_array):
    """
    Given a SimInspiral array and a process params table as an LSCArray,
    finds the min_vol and vol_weight needed for each injection. This
    information is added to distributions.min_vol and
    distributions.volume_weight, respectively.
    """
    # check for required fields
    required_cols = ['distance', 'process_id']
    not_present = [col for col in required_cols \
        if col not in sim_inspiral_array.fieldnames]
    if any(not_present):
        raise ValueError("sim_inspiral_array is missing required fields %s" %(
            ', '.join(not_present)))
    # load the distance distributions
    rdistrs = get_r_distribution_from_process_params(process_params_array)
    # if more than one distribution present, expand to the sim_inspiral_array
    if rdistrs.size > 1:
        rdistrs = sim_inspiral_array.with_fields(
            ['process_id'], copy=True).join(rdistrs, 'process_id')
        dist_types = rdistrs['distance_type']
        distrs = rdistrs['distribution']
        min_dists = rdistrs['min_dist']
        max_dists = rdistrs['max_dist']
        # vectorize distribution's vol weights
        get_weights = numpy.vectorize(
            distributions.get_dist_weights_from_inspinj_params)
    else:
        # can just get values from the first element
        dist_types = rdistrs['distance_type'][0]
        distrs = rdistrs['distribution'][0]
        min_dists = rdistrs['min_dist'][0]
        max_dists = rdistrs['max_dist'][0]
        get_weights = numpy.vectorize(
            distributions.get_dist_weights_from_inspinj_params,
            excluded=[1,2,3,4])
    # get the weights
    min_vol, vweights = get_weights(sim_inspiral_array['distance'],
        dist_types, distrs, min_dists, max_dists,
        mchirp=sim_inspiral_array.mchirp)
    # add the weights to the sim_inspiral array
    return sim_inspiral_array.add_fields([min_vol, vweights],
        ['min_vol', 'volume_weight'])


def add_parameter_distributions(sim_inspiral_array, distribution_dict):
    """
    Given a sim_inspiral_array and a dictionary of CBCDistribution instances,
    maps each injection to the appropriate distribution based on process id.

    Parameters
    ----------
    sim_inspiral_array: SimInspiral LSCArray instance
        An instance of a sim_inspiral array. Must have a process_id column.
    distribution_dict: dict
        Dictionary of distributions keyed by the process ids. To generate this
        dictionary, see ``load_distributions_from_process_params``.

    Returns
    -------
    sim_inspiral_array: SimInspiral LSCArray instance
        A copy of the sim_inspiral_array with the distributions added as
        objects to the column param_distributions.
    """
    pdistrs = lscarrays.default_empty(sim_inspiral_array.size, dtype=object)
    for proc_id in numpy.unique(sim_inspiral_array['process_id']):
        getidx = numpy.where(sim_inspiral_array['process_id'] == proc_id)
        pdistrs[getidx] = distribution_dict[proc_id]
    return sim_inspiral_array.add_fields(pdistrs, ['param_distribution'])
