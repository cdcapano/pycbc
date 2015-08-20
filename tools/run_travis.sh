#!/bin/bash

set -v

INST=${HOME}/inst
source ${INST}/etc/lal-user-env.sh
export LD_LIBRARY_PATH=${INST}/lib:${INST}/lib64
export PKG_CONFIG_PATH=${INST}/lib/pkgconfig:${PKG_CONFIG_PATH}
export PATH=/usr/lib/ccache:${PATH}:${INST}/bin

# Using python setup.py test has two issues:
#     Some tests fail for reasons not necessarily related to PyCBC
#     Setup.py seems to returns 0 even when tests fail
# So we rather run specific tests manually

RESULT=0

python test/test_array_lal.py
test $? -ne 0 && RESULT=1

python test/test_array.py
test $? -ne 0 && RESULT=1

#python test/test_autochisq.py
#test $? -ne 0 && RESULT=1

python test/test_chisq.py
test $? -ne 0 && RESULT=1

python test/test_correlate.py
test $? -ne 0 && RESULT=1

#python test/test_fft_unthreaded.py
#test $? -ne 0 && RESULT=1

#python test/test_frame.py
#test $? -ne 0 && RESULT=1

python test/test_frequencyseries.py
test $? -ne 0 && RESULT=1

#python test/test_injection.py
#test $? -ne 0 && RESULT=1

python test/test_matchedfilter.py
test $? -ne 0 && RESULT=1

python test/test_pnutils.py
test $? -ne 0 && RESULT=1

python test/test_psd.py
test $? -ne 0 && RESULT=1

python test/test_resample.py
test $? -ne 0 && RESULT=1

#python test/test_schemes.py
#test $? -ne 0 && RESULT=1

python test/test_threshold.py
test $? -ne 0 && RESULT=1

python test/test_timeseries.py
test $? -ne 0 && RESULT=1

python test/test_tmpltbank.py
test $? -ne 0 && RESULT=1

# check for trivial failures of important executables

pycbc_inspiral --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_geom_nonspinbank --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_aligned_stoch_bank --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_geom_aligned_bank --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_coinc_mergetrigs --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_coinc_findtrigs --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_coinc_hdfinjfind --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_coinc_statmap --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_coinc_statmap_inj --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_plot_singles_vs_params --help > /dev/null
test $? -ne 0 && RESULT=1

pycbc_plot_psd_file --help > /dev/null
test $? -ne 0 && RESULT=1

exit ${RESULT}
