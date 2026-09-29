# Copyright (C) 2026 Collin Capano
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
"""Tests the ``gated_gaussian_multimargphasepol`` model.

A QNM template (220 + 221 modes) is applied to the post-merger part of an
IMR injection in simulated Gaussian noise. The likelihood marginalized over
polarization and the 220 phase is compared to brute-force marginalizing
the ``gated_gaussian_margpol`` model over the phase and the
``gated_gaussian_multimargphase`` model over polarization.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy
import scipy.linalg
from scipy.special import logsumexp

from pycbc.inference import models
from pycbc.strain.gate import (batch_gate_and_paint_fd, invert_covariance,
                               toeplitz_inverse, toeplitz_inverses)
from pycbc.workflow import WorkflowConfigParser
from utils import simple_exit

TESTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(TESTDIR, 'data', 'gated_margphasepol')
CREATE_INJECTIONS = os.path.join(TESTDIR, '..', 'bin',
                                 'pycbc_create_injections')

# The ringdown__ parameters in
# examples/inference/time_marg_bhspec/expected_maxl-gw150914.json
MAXL_PHI220 = 2.8333365224081235
TEMPLATE_PARAMS = {
    'ra': 0.9122465496513537,
    'dec': -0.6969462844463007,
    'delta_tc': -0.006457389096415349,
    'inclination': 1.3573258236406052,
    'final_mass': 72.2337559752136,
    'final_spin': 0.7206958553185714,
    'amp220': 1.652209702951859e-20,
    'amp221': 1.034137226179199,
    # the phase of the 221 is relative to the 220 in the margphasepol model
    'phi221': 5.5393363189773375 - MAXL_PHI220,
}


class TestGatedMargPhasePol(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.injfile = os.path.join(cls.tmpdir, 'injection.hdf')
        subprocess.run(
            [sys.executable, CREATE_INJECTIONS,
             '--config-files', os.path.join(DATADIR, 'injection.ini'),
             '--ninjections', '1', '--seed', '10',
             '--output-file', cls.injfile,
             '--variable-params-section', 'variable_params',
             '--static-params-section', 'static_params',
             '--dist-section', 'prior', '--force'],
            check=True)
        # the model to test
        cls.model = models.read_from_config(cls.config())
        cls.model.update(**TEMPLATE_PARAMS)
        cls.marglogl = cls.model.loglikelihood
        cls.stats = cls.model.current_stats

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    @classmethod
    def config(cls):
        """Loads the model config file, using the test's injection file."""
        cp = WorkflowConfigParser([os.path.join(DATADIR,
                                                'ringdown_margphasepol.ini')])
        cp.set('data', 'injection-file', cls.injfile)
        return cp

    def test_marglogl_is_finite(self):
        self.assertTrue(numpy.isfinite(self.marglogl))
        self.assertTrue(self.stats['maxl_logl'] >= self.marglogl)
        # the loglr should be large, since there's a loud signal
        self.assertTrue(self.model.loglr > 100)

    def test_batch_gating(self):
        """Checks that batched gating matches gating each series."""
        wfs = self.model.get_waveforms()
        gate_times = self.model.get_gate_times()
        for paint_method in ['matmul', 'toeplitz', 'gs']:
            for det, modes in wfs.items():
                terms = [x for mode in modes.values() for x in mode]
                invpsd = self.model._invpsds[det]
                gatestart, dgate = gate_times[det]
                invmat = None
                if paint_method == 'matmul':
                    invmat = self.model.invert_covariance(det)
                batched = batch_gate_and_paint_fd(
                    terms, gatestart + dgate/2, dgate/2, invpsd,
                    paint_method=paint_method, invmat=invmat)
                for h, bh in zip(terms, batched):
                    expected = h.to_timeseries().gate(
                        gatestart + dgate/2, window=dgate/2, invpsd=invpsd,
                        method='paint', paint_method=paint_method,
                        paint_invmat=invmat).to_frequencyseries()
                    self.assertEqual(len(bh), len(expected))
                    self.assertEqual(bh.epoch, expected.epoch)
                    err = abs(bh.numpy() - expected.numpy()).max()
                    self.assertTrue(err < 1e-12 * abs(expected.numpy()).max())

    def test_toeplitz_inverse(self):
        """Checks that the Gohberg-Semencul inverse matches the explicit
        inverse of the covariance matrix."""
        det = self.model.det_names[0]
        invpsd = self.model._invpsds[det]
        lindex, rindex = self.model.gate_indices(det)
        invmat = invert_covariance(invpsd, lindex, rindex)
        tinv = toeplitz_inverse(invpsd, lindex, rindex)
        vecs = numpy.random.default_rng(0).standard_normal(
            (3, rindex - lindex))
        expected = vecs @ invmat.T
        # the inverses of several sizes found with a single pass of the
        # Levinson-Durbin recursion should be the same
        n = rindex - lindex
        tinvs = toeplitz_inverses(invpsd, [n - 1, n])
        for got in [tinv.apply(vecs),
                    numpy.array([tinv.apply(v) for v in vecs]),
                    tinvs[n].apply(vecs)]:
            err = abs(got - expected).max()
            self.assertTrue(err < 1e-9 * abs(expected).max())
        expected = vecs[:, :n-1] @ invert_covariance(
            invpsd, lindex, rindex - 1).T
        err = abs(tinvs[n-1].apply(vecs[:, :n-1]) - expected).max()
        self.assertTrue(err < 1e-9 * abs(expected).max())

    def test_gs_paint_method(self):
        """Checks that the model gives the same likelihood when in-painting
        with the Gohberg-Semencul method as with the explicit inverse."""
        cp = self.config()
        cp.set('model', 'paint-method', 'gs')
        model = models.read_from_config(cp)
        model.update(**TEMPLATE_PARAMS)
        self.assertAlmostEqual(model.loglikelihood, self.marglogl,
                               delta=1e-8 * abs(self.marglogl))
        self.assertAlmostEqual(model.current_stats['maxl_logl'],
                               self.stats['maxl_logl'],
                               delta=1e-8 * abs(self.stats['maxl_logl']))

    def test_fill_gate_cache(self):
        """Checks that the gate cache is filled by the warmup when using the
        gs paint method, and that the cached inverses are used."""
        cp = self.config()
        cp.set('model', 'paint-method', 'gs')
        cp.set('model', 'gate-cache-samples', '1000')
        model = models.read_from_config(cp)
        # nothing should be cached until the warmup
        self.assertEqual(len(model._cov_matrices), 0)
        model.warmup()
        cached = {(k, det) for k, dets in model._cov_matrices.items()
                  for det in dets}
        self.assertTrue(len(cached) > 0)
        # the same inverses should be set up (up to round-off) when using a
        # pool
        pmodel = models.read_from_config(cp)
        numpy.random.seed(0)
        model._cov_matrices.clear()
        model.warmup()
        numpy.random.seed(0)
        pmodel.warmup(nprocesses=2)
        self.assertEqual(
            {(k, det) for k, dets in pmodel._cov_matrices.items()
             for det in dets},
            {(k, det) for k, dets in model._cov_matrices.items()
             for det in dets})
        maxdiff = 0.
        for k, dets in model._cov_matrices.items():
            for det, tinv in dets.items():
                x = pmodel._cov_matrices[k][det].x
                maxdiff = max(maxdiff, abs(tinv.x - x).max() / abs(x).max())
        self.assertTrue(maxdiff < 1e-10)
        cached = {(k, det) for k, dets in model._cov_matrices.items()
                  for det in dets}
        model.update(**TEMPLATE_PARAMS)
        logl = model.loglikelihood
        # filling the cache should not change the output of the model's
        # waveform transforms, which should still be scalars
        for p in ['tc', 't_gate_start', 't_gate_end']:
            self.assertEqual(numpy.ndim(model.current_params[p]), 0)
        # the gate sizes needed for this point should have been in the cache
        for det in model.det_names:
            lindex, rindex = model.gate_indices(det)
            self.assertIn((rindex - lindex, det), cached)
        self.assertAlmostEqual(logl, self.marglogl,
                               delta=1e-8 * abs(self.marglogl))
        # no cache when disabled
        cp.set('model', 'gate-cache-samples', '0')
        model = models.read_from_config(cp)
        model.warmup()
        self.assertEqual(len(model._cov_matrices), 0)

    def test_hierarchical_warmup(self):
        """Checks that the submodels of a hierarchical model draw their
        parameters from the hierarchical prior, and that the hierarchical
        warmup fills the submodels' gate caches."""
        cp = self.config()
        # make a hierarchical model with the model as its only submodel
        lbl = 'rd'
        for sec in ['model', 'data']:
            cp.add_section(f'{lbl}__{sec}')
            for opt, val in cp.items(sec):
                cp.set(f'{lbl}__{sec}', opt, val)
            cp.remove_section(sec)
        cp.add_section('model')
        cp.set('model', 'name', 'hierarchical')
        cp.set('model', 'submodels', lbl)
        cp.set(f'{lbl}__model', 'paint-method', 'gs')
        cp.set(f'{lbl}__model', 'gate-cache-samples', '1000')
        model = models.read_from_config(cp)
        submodel = model.submodels[lbl]
        # the submodel's prior samples should include the outputs of the
        # waveform transforms
        samples = submodel.prior_rvs(size=10)
        self.assertEqual(len(samples), 10)
        for p in list(submodel.variable_params) + ['t_gate_start',
                                                   't_gate_end']:
            self.assertIn(p, samples.fieldnames)
            self.assertEqual(samples[p].shape, (10,))
        # the warmup should fill the submodel's cache
        self.assertEqual(len(submodel._cov_matrices), 0)
        model.warmup()
        self.assertTrue(len(submodel._cov_matrices) > 0)
        model.update(**TEMPLATE_PARAMS)
        logl = model.loglikelihood
        for det in submodel.det_names:
            lindex, rindex = submodel.gate_indices(det)
            self.assertIn(det, submodel._cov_matrices[rindex - lindex])
        self.assertAlmostEqual(logl, self.marglogl,
                               delta=1e-8 * abs(self.marglogl))

    def test_logdet_fit(self):
        """Checks that the determinants used for the normalization, which
        are computed from truncated covariance matrices constructed directly
        from the autocorrelation, match those of the matrices obtained by
        removing rows and columns from the full covariance matrix."""
        det = self.model.det_names[0]
        psd = self.model.psds[det]
        # use a short (decaying) autocorrelation so the full matrix is small
        col = self.model._Rss[det].numpy()[:256] / 2
        col = col * numpy.exp(-numpy.arange(len(col)) / 32.)
        (sizes, logdets), _ = self.model.logdet_fit(col, psd)
        cov = scipy.linalg.toeplitz(col)
        s = len(col)
        for size, logdet in zip(sizes[1:], logdets[1:]):
            start = size // 2
            end = start + s - size
            tc = numpy.delete(numpy.delete(cov, slice(start, end), 0),
                              slice(start, end), 1)
            self.assertAlmostEqual(logdet, numpy.linalg.slogdet(tc)[1],
                                   delta=1e-8 * abs(logdet))

    def test_gate_times(self):
        """Checks that the gates are shifted into each detector by the delay
        at tc (preserving the gate width), unless the model has a time
        varying response, in which case each edge is shifted by the delay at
        that time."""
        from pycbc.detector import Detector
        model = self.model
        samples = {p: numpy.full(1000, v) for p, v in TEMPLATE_PARAMS.items()}
        rng = numpy.random.default_rng(1)
        samples['ra'] = rng.uniform(0, 2 * numpy.pi, 1000)
        samples['dec'] = numpy.arcsin(rng.uniform(-1, 1, 1000))
        tc = model.static_params['trigger_time'] + rng.uniform(-0.05, 0.05,
                                                               1000)
        start, end = tc - 1., tc
        refframe = model.static_params.get('tc_ref_frame', 'geocentric')
        gatetimes = model._get_gate_times(start, end, samples['ra'],
                                          samples['dec'], refframe=refframe,
                                          tc=tc)
        for det, (gstart, gwidth) in gatetimes.items():
            thisdet = Detector(det)
            delay = thisdet.arrival_time_delay(tc, samples['ra'],
                                               samples['dec'], refframe)
            numpy.testing.assert_array_equal(gstart, start + delay)
            numpy.testing.assert_array_equal(gwidth, end - start)
            # the gate is always the same number of samples
            ts = model.td_data[det]
            st0 = float(ts.start_time)
            dt = float(ts.delta_t)
            nsamples = (numpy.trunc((gstart + gwidth - st0) / dt)
                        - numpy.trunc((gstart - st0) / dt))
            self.assertEqual(len(numpy.unique(nsamples)), 1)
        # with a time varying response, each edge is shifted by the delay at
        # that time
        model._time_varying_response = True
        try:
            gatetimes = model._get_gate_times(start, end, samples['ra'],
                                              samples['dec'],
                                              refframe=refframe)
        finally:
            model._time_varying_response = False
        for det, (gstart, gwidth) in gatetimes.items():
            thisdet = Detector(det)
            astart = thisdet.arrival_time(start, samples['ra'],
                                          samples['dec'], refframe)
            aend = thisdet.arrival_time(end, samples['ra'], samples['dec'],
                                        refframe)
            numpy.testing.assert_array_equal(gstart, astart)
            numpy.testing.assert_array_equal(gwidth, aend - astart)

    def test_margpol_brute_phase(self):
        """Marginalizes gated_gaussian_margpol over the 220 phase."""
        cp = self.config()
        cp.set('model', 'name', 'gated_gaussian_margpol')
        for opt in ['ref_phase', 'phase_names', 'phase_samples']:
            cp.remove_option('model', opt)
        # use the (summed-mode) version of the approximant
        cp.set('static_params', 'approximant', 'TdQNMfromFinalMassSpin')
        cp.set('variable_params', 'phi220', '')
        cp.add_section('prior-phi220')
        cp.set('prior-phi220', 'name', 'uniform_angle')
        model = models.read_from_config(cp)
        numpy.testing.assert_array_equal(model.pol, self.model.pol)
        logls = []
        maxl = -numpy.inf
        for phi in self.model.phases:
            # every mode's phase is shifted by the same amount
            model.update(**dict(TEMPLATE_PARAMS, phi220=phi,
                                phi221=TEMPLATE_PARAMS['phi221'] + phi))
            logls.append(model.loglikelihood)
            maxl = max(maxl, model.current_stats['maxl_logl'])
        marglogl = logsumexp(logls) - numpy.log(len(logls))
        self.assertAlmostEqual(marglogl, self.marglogl, delta=1e-8)
        self.assertAlmostEqual(maxl, self.stats['maxl_logl'], delta=1e-8)

    def test_multimargphase_brute_pol(self):
        """Marginalizes gated_gaussian_multimargphase over polarization."""
        cp = self.config()
        cp.set('model', 'name', 'gated_gaussian_multimargphase')
        cp.remove_option('model', 'polarization_samples')
        # the multimode margphase model requires the reference phase to be 0
        cp.set('static_params', 'phi220', '0')
        cp.set('variable_params', 'polarization', '')
        cp.add_section('prior-polarization')
        cp.set('prior-polarization', 'name', 'uniform_angle')
        model = models.read_from_config(cp)
        numpy.testing.assert_array_equal(model.phases, self.model.phases)
        logls = []
        maxl = -numpy.inf
        for pol in self.model.pol:
            model.update(**dict(TEMPLATE_PARAMS, polarization=pol))
            logls.append(model.loglikelihood)
            maxl = max(maxl, model.current_stats['maxl_logl'])
        marglogl = logsumexp(logls) - numpy.log(len(logls))
        self.assertAlmostEqual(marglogl, self.marglogl, delta=1e-8)
        self.assertAlmostEqual(maxl, self.stats['maxl_logl'], delta=1e-8)

    @classmethod
    def snr_config(cls, model_name):
        """Config for sampling in the SNR of the 220 mode."""
        cp = cls.config()
        cp.set('model', 'name', model_name)
        cp.remove_option('variable_params', 'amp220')
        cp.remove_section('prior-amp220')
        cp.set('variable_params', 'amp220_snr', '')
        cp.add_section('prior-amp220_snr')
        cp.set('prior-amp220_snr', 'name', 'uniform')
        cp.set('prior-amp220_snr', 'min-amp220_snr', '0')
        cp.set('prior-amp220_snr', 'max-amp220_snr', '50')
        cp.set('model', 'sample-snrs', '')
        cp.set('model', 'snr-mode-map', '220:amp220_snr:amp220')
        # the 221 amplitude is relative to the 220
        cp.set('model', 'ref-mode', '')
        return cp

    def test_multimargphase_brute_pol_snr(self):
        """Marginalizes gated_gaussian_multimargphase over polarization when
        sampling in the SNR of the 220 mode."""
        params = {p: val for p, val in TEMPLATE_PARAMS.items()
                  if p != 'amp220'}
        params['amp220_snr'] = 15.
        # use fewer polarizations to keep the run time down
        cp = self.snr_config('gated_gaussian_multimargphasepol')
        cp.set('model', 'polarization_samples', '250')
        model = models.read_from_config(cp)
        model.update(**params)
        expected = model.loglikelihood
        stats = model.current_stats
        self.assertTrue(numpy.isfinite(expected))
        # brute force
        cp = self.snr_config('gated_gaussian_multimargphase')
        cp.remove_option('model', 'polarization_samples')
        cp.set('static_params', 'phi220', '0')
        cp.set('variable_params', 'polarization', '')
        cp.add_section('prior-polarization')
        cp.set('prior-polarization', 'name', 'uniform_angle')
        brute = models.read_from_config(cp)
        numpy.testing.assert_array_equal(brute.phases, model.phases)
        logls = []
        maxls = []
        scales = []
        for pol in model.pol:
            brute.update(**dict(params, polarization=pol))
            logls.append(brute.loglikelihood)
            maxls.append(brute.current_stats['maxl_logl'])
            scales.append(brute.current_stats['scale_factor_220'])
        marglogl = logsumexp(logls) - numpy.log(len(logls))
        self.assertAlmostEqual(marglogl, expected, delta=1e-8)
        self.assertAlmostEqual(max(maxls), stats['maxl_logl'], delta=1e-8)
        # the scale factor at the maxL polarization should be the same; the
        # antenna patterns are periodic in polarization with period pi, so
        # the likelihood at pol and pol + pi is the same up to round-off,
        # and which of the two is the max is arbitrary
        idx = numpy.argmax(maxls)
        dpol = (model.pol[idx] - stats['maxl_polarization']) % numpy.pi
        self.assertAlmostEqual(min(dpol, numpy.pi - dpol), 0., delta=1e-10)
        self.assertAlmostEqual(scales[idx] / stats['scale_factor_220'], 1.,
                               delta=1e-10)


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
    TestGatedMargPhasePol))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
