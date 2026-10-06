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
"""Tests the ``t0`` argument of the ringdown waveforms, which starts the
ringdown ``t0`` seconds after the waveform's reference time."""

import unittest

import numpy

from pycbc.waveform import generator
from pycbc.waveform.ringdown import (get_td_from_final_mass_spin,
                                     get_td_modes_from_final_mass_spin)
from utils import simple_exit

PARAMS = {'final_mass': 70., 'final_spin': 0.7, 'lmns': ['221', '331'],
          'inclination': 0.5, 'amp220': 1e-21, 'phi220': 0.3,
          'amp330': 0.2, 'phi330': 1.}
# the values of t0 to test, after and before the reference time
T0S = [0.01, -0.01]


class TestRingdownT0(unittest.TestCase):

    def test_td_epoch(self):
        """The time-domain ringdown is the same, but starts at t0, which
        may be before (t0 < 0) or after (t0 > 0) the reference time."""
        for func in [get_td_from_final_mass_spin,
                     get_td_modes_from_final_mass_spin]:
            ref = func(delta_t=1./2048, **PARAMS)
            if isinstance(ref, dict):
                ref = [x for hpc in ref.values() for x in hpc]
            for t0 in T0S:
                out = func(delta_t=1./2048, t0=t0, **PARAMS)
                if isinstance(out, dict):
                    out = [x for hpc in out.values() for x in hpc]
                for r, x in zip(ref, out):
                    self.assertEqual(float(x.start_time),
                                     float(r.start_time) + t0)
                    numpy.testing.assert_array_equal(x.numpy(), r.numpy())

    def test_generator(self):
        """A ringdown with t0 is placed in the same way as a ringdown
        with t0 added to its tc, in both domains."""
        for approximant, domain in [('TdModesfromFinalMassSpin', 'td'),
                                    ('FdModesfromFinalMassSpin', 'fd')]:
            rframe = generator.select_waveform_modes_generator(approximant,
                                                               domain)
            gen = generator.FDomainDetFrameModesGenerator(
                rframe, 0., variable_args=['tc', 't0'], delta_f=1./4,
                delta_t=1./2048, f_lower=10., approximant=approximant,
                **PARAMS)
            for t0 in T0S:
                out = gen.generate(tc=2., t0=t0)['RF']
                ref = gen.generate(tc=2.+t0, t0=0.)['RF']
                for mode in ref:
                    for x, r in zip(out[mode], ref[mode]):
                        r = r.numpy()
                        err = abs(x.numpy() - r).max()
                        self.assertTrue(err < 1e-10 * abs(r).max(),
                                        f"{approximant} {mode} t0={t0}: "
                                        f"{err}")


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRingdownT0))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
