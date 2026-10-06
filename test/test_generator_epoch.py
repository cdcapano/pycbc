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
"""Tests that the frequency-domain generators place time-domain waveforms
at ``tc`` plus the waveform's epoch, whether the epoch is negative (as for
most time-domain waveforms) or positive."""

import unittest

import numpy

from pycbc.types import TimeSeries
from pycbc.waveform import generator
from pycbc.waveform.plugin import add_custom_waveform
from utils import simple_exit

APPROX = 'test_generator_epoch_sinegauss'


def _sinegauss(toffset=0., delta_t=1./2048, **kwargs):
    """A sine-Gaussian that starts at ``toffset``."""
    times = numpy.arange(512) * delta_t
    hp = numpy.exp(-((times - 0.125) / 0.02)**2) * \
        numpy.cos(2 * numpy.pi * 100 * times)
    hc = numpy.exp(-((times - 0.125) / 0.02)**2) * \
        numpy.sin(2 * numpy.pi * 100 * times)
    return (TimeSeries(hp, delta_t=delta_t, epoch=toffset),
            TimeSeries(hc, delta_t=delta_t, epoch=toffset))


add_custom_waveform(APPROX, _sinegauss, 'time', force=True)


class TestGeneratorEpoch(unittest.TestCase):

    def test_epoch(self):
        gen = generator.FDomainDetFrameGenerator(
            generator.TDomainCBCGenerator, 0.,
            variable_args=['tc', 'toffset'], delta_f=1./8, delta_t=1./2048,
            f_lower=10., approximant=APPROX)
        # the same waveform with a positive and a negative epoch, placed at
        # the same time
        tc = 2.
        pos = gen.generate(tc=tc, toffset=0.25)['RF']
        neg = gen.generate(tc=tc + 0.5, toffset=-0.25)['RF']
        self.assertTrue(numpy.allclose(pos.numpy(), neg.numpy(), rtol=0,
                                       atol=1e-12 * abs(neg.numpy()).max()))
        # the peak is at tc + epoch + 0.125
        ht = pos.to_timeseries()
        tpeak = float(ht.sample_times[numpy.argmax(abs(ht.numpy()))])
        self.assertAlmostEqual(tpeak, tc + 0.25 + 0.125, delta=0.005)


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestGeneratorEpoch))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
