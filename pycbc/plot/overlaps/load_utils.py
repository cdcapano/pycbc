#! /usr/bin/env python

import sys
import numpy
import sqlite3

from pycbc import distributions
from pycbc.plot import data_utils

def get_injection_results(filenames, load_inj_distribution=False,
        weight_function='uniform', result_table_name='overlap_results',
        ifo=None, verbose=False):
    sqlquery = """
        SELECT
            sim.process_id, sim.waveform, sim.simulation_id,
            sim.mass1, sim.mass2, sim.spin1x, sim.spin1y, sim.spin1z,
            sim.spin2x, sim.spin2y, sim.spin2z,
            sim.distance, sim.inclination, sim.alpha1, sim.alpha2,
            tmplt.event_id, tmplt.mass1, tmplt.mass2,
            tmplt.spin1x, tmplt.spin1y, tmplt.spin1z,
            tmplt.spin2x, tmplt.spin2y, tmplt.spin2z,
            res.effectualness, res.snr, res.snr_std,
            tw.weight, res.chisq, res.chisq_std, res.chisq_dof, res.new_snr,
            res.new_snr_std, res.num_successes, res.sample_rate,
            res.coinc_event_id
        FROM
            %s AS res""" %(result_table_name) + """
        JOIN
            sim_inspiral as sim, coinc_event_map as map
        ON
            sim.simulation_id == map.event_id AND
            map.coinc_event_id == res.coinc_event_id
        JOIN
            sngl_inspiral AS tmplt, coinc_event_map AS mapB
        ON
            mapB.coinc_event_id == map.coinc_event_id AND
            mapB.event_id == tmplt.event_id
        JOIN
            coinc_definer AS cdef, coinc_event AS cev
        ON
            cev.coinc_event_id == res.coinc_event_id AND
            cdef.coinc_def_id == cev.coinc_def_id
        JOIN
            tmplt_weights as tw
        ON
            tw.tmplt_id == tmplt.event_id AND
            tw.weight_function == cdef.description
        WHERE
            cdef.description == ?
    """
    if ifo is not None:
        sqlquery += 'AND res.ifo == ?'
        get_args = (weight_function, ifo)
    else:
        get_args = (weight_function,)
    results = []
    id_map = {}
    idx = 0
    inj_dists = {}
    for ii,thisfile in enumerate(filenames):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(filenames)),
            sys.stdout.flush()
        if not thisfile.endswith('.sqlite'):
            continue
        connection = sqlite3.connect(thisfile)
        cursor = connection.cursor()
        try:
            for (sim_proc_id, apprx, sim_id, m1, m2, s1x, s1y, s1z,
                    s2x, s2y, s2z, dist, inc, min_vol, inj_weight,
                    tmplt_evid, tmplt_m1, tmplt_m2,
                    tmplt_s1x, tmplt_s1y, tmplt_s1z,
                    tmplt_s2x, tmplt_s2y, tmplt_s2z, ff, snr,
                    snr_std, weight, chisq, chisq_std, chisq_dof, new_snr,
                    new_snr_std, nsamp, sample_rate, ceid) in \
                    cursor.execute(sqlquery, get_args):
                thisRes = data_utils.Result()
                # id information
                thisRes.unique_id = idx
                thisRes.database = thisfile 
                thisRes.event_id = ceid
                id_map[thisfile, sim_id] = idx
                idx += 1
                # Set the injection parameters: we'll make the injection
                # the psuedo class so we can access its attributes directly
                thisRes.set_psuedoattr_class(thisRes.injection)
                thisRes.injection.simulation_id = sim_id
                thisRes.injection.approximant = apprx
                # ensure that m1 is always > m2
                if m2 > m1:
                    thisRes.injection.mass1 = m2
                    thisRes.injection.mass2 = m1
                    thisRes.injection.spin1x = s2x
                    thisRes.injection.spin1y = s2y
                    thisRes.injection.spin1z = s2z
                    thisRes.injection.spin2x = s1x
                    thisRes.injection.spin2y = s1y
                    thisRes.injection.spin2z = s1z
                else:
                    thisRes.injection.mass1 = m1
                    thisRes.injection.mass2 = m2
                    thisRes.injection.spin1x = s1x
                    thisRes.injection.spin1y = s1y
                    thisRes.injection.spin1z = s1z
                    thisRes.injection.spin2x = s2x
                    thisRes.injection.spin2y = s2y
                    thisRes.injection.spin2z = s2z
                thisRes.injection.distance = dist
                thisRes.injection.inclination = inc
                thisRes.injection.min_vol = min_vol
                thisRes.injection.vol_weight = inj_weight
                thisRes.injection.sample_rate = sample_rate
                # set the template parameters
                thisRes.template.tmplt_id = tmplt_evid
                thisRes.template.mass1 = tmplt_m1
                thisRes.template.mass2 = tmplt_m2
                thisRes.template.spin1x = tmplt_s1x
                thisRes.template.spin1y = tmplt_s1y
                thisRes.template.spin1z = tmplt_s1z
                thisRes.template.spin2x = tmplt_s2x
                thisRes.template.spin2y = tmplt_s2y
                thisRes.template.spin2z = tmplt_s2z
                thisRes.template.weight_function = weight_function
                thisRes.template.weight = weight
                # statistics
                thisRes.effectualness = ff
                thisRes.snr = snr
                thisRes.snr_std = snr_std
                thisRes.chisq = chisq
                thisRes.chisq_sts = chisq_std
                thisRes.chisq_dof = chisq_dof
                thisRes.new_snr = new_snr
                thisRes.new_snr_std = new_snr_std
                thisRes.num_samples = nsamp
                # get the injection distribution information
                if load_inj_distribution:
                    try:
                        thisRes.injection.mass_distr = inj_dists[thisfile,
                            sim_proc_id]
                    except KeyError:
                        # need to load the distribution
                        inj_dists[thisfile, sim_proc_id] = \
                            distributions.get_inspinj_distribution(connection,
                            sim_proc_id)
                        thisRes.injection.mass_distr = inj_dists[thisfile,
                            sim_proc_id]
                results.append(thisRes)

            # try to get the sim_inspiral_params table
            tables = cursor.execute(
                'SELECT name FROM sqlite_master WHERE type == "table"'
                ).fetchall()
            if ('sim_inspiral_params',) in tables:
                # older codes stored the minimum volume and injection weight in
                # the sim_inspiral_params table. If we find that column, we'll
                # get the min_vol and vol_weight from there. Otherwise, the
                # min_vol and the vol_weight are set above. 
                column_names = [name[1] for name in cursor.execute(
                    "PRAGMA table_info(sim_inspiral_params)").fetchall()]
                if "min_vol" in column_names and "weight" in column_names:
                    sipquery = """
                        SELECT
                            sip.simulation_id, sip.ifo, sip.sigmasq,
                            sip.min_vol, sip.weight
                        FROM
                            sim_inspiral_params AS sip
                        """
                else:
                    sipquery = """
                        SELECT
                            sip.simulation_id, sip.ifo, sip.sigmasq,
                            NULL, NULL
                        FROM
                            sim_inspiral_params AS sip
                        """
                for simid, ifo, sigmasq, min_vol, vol_weight in cursor.execute(
                        sipquery):
                    try:
                        thisRes = results[id_map[thisfile, simid]]
                    except KeyError:
                        continue
                    sngl_params = data_utils.SnglIFOInjectionParams(ifo=ifo,
                        sigma=numpy.sqrt(sigmasq))
                    thisRes.injection.sngl_ifos[ifo] = sngl_params
                    if min_vol is not None:
                        thisRes.injection.min_vol = min_vol
                    if inj_weight is not None:
                        thisRes.injection.vol_weight = vol_weight

        except sqlite3.OperationalError:
            cursor.close()
            connection.close()
            continue
        except sqlite3.DatabaseError:
            cursor.close()
            connection.close()
            print "Database Error: %s" % thisfile
            continue

        connection.close()

    if verbose:
        print >> sys.stdout, ""

    return results, id_map


def get_templates(filename, old_format=False):
    templates = []
    connection = sqlite3.connect(filename)
    if old_format:
        sqlquery = 'select sngl.mass1, sngl.mass2, sngl.alpha3, sngl.alpha6 from sngl_inspiral as sngl'
    else:
        sqlquery = 'select sngl.mass1, sngl.mass2, sngl.spin1z, sngl.spin2z from sngl_inspiral as sngl'
    for m1, m2, s1z, s2z in connection.cursor().execute(sqlquery):
        thisRes = data_utils.Result()
        thisRes.set_psuedoattr_class(thisRes.template)
        thisRes.m1 = m1
        thisRes.m2 = m2
        thisRes.s1z = s1z
        thisRes.s2z = s2z
        templates.append(thisRes)
    connection.close()
    return templates
