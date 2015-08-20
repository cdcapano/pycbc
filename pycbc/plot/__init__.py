import matplotlib
matplotlib.use('Agg')

from pycbc.plot.data_utils import *
from pycbc.plot.plot_utils import *

from pycbc.plot import overlaps
from pycbc.plot import pycbc_sqlite
from pycbc.plot import hdfcoinc
from pycbc.plot import gstlal

# the known pipeline types that we can parse
known_pipelines = {
    'overlaps': overlaps,
    'pycbc_sqlite': pycbc_sqlite,
    'hdfcoinc': hdfcoinc,
    'gstlal': gstlal
}
