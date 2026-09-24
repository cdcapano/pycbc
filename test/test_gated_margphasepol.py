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
from scipy.special import logsumexp

from pycbc.inference import models
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


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
    TestGatedMargPhasePol))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
