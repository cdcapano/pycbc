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
#                           Preamble
#
# =============================================================================
#
"""
This modules provides definitions of, and helper functions for, LSCArrays.
LSCArrays are wrappers of numpy recarrays with additional functionality useful for storing and retrieving data created by a search for gravitational waves.
"""

import os, sys
import inspect
import numpy
import operator
import numpy.lib.recfunctions as recfunctions

import lal
import lalsimulation as lalsim
from glue.ligolw import types as ligolw_types

#
# =============================================================================
#
#                           Data type mappings
#
# =============================================================================
#
# add ligolw_types to numpy typeDict
numpy.typeDict.update(ligolw_types.ToNumPyType)

# Annoyingly, numpy has no way to store NaNs in an integer field to indicate
# the equivalent of None. This can be problematic for fields that store ids:
# if an array has an id field with value 0, it isn't clear if this is because
# the id is the first element, or if no id was set. To clear up the ambiguity,
# we define here an integer to indicate 'id not set'. 
ID_NOT_SET = -1
EMPTY_OBJECT = None

def set_default_empty(array):
    if array.dtype.names is None:
        # scalar dtype, just set
        if array.dtype.str[1] == 'i':
            # integer, set to ID_NOT_SET
            array[:] = ID_NOT_SET
        elif array.dtype.str[1] == 'O':
            # object, set to EMPTY_OBJECT
            array[:] = EMPTY_OBJECT
    else:
        for name in array.dtype.names:
            set_default_empty(array[name])

def default_empty(shape, dtype):
    """
    Numpy's empty array can have random values in it. To prevent that, we
    define here a default emtpy array. This default empty is a numpy.zeros
    array, except that objects are set to None, and all ints to ID_NOT_SET.
    """
    default = numpy.zeros(shape, dtype=dtype)
    set_default_empty(default)
    return default

# set default data types
_default_types_status = {
    'default_strlen': 50,
    'ilwd_as_int': True,
    'lstring_as_obj': False
}

def lstring_as_obj(true_or_false=None):
    """
    Toggles whether lstrings should be treated as strings or as objects.
    When lscarrays is first loaded, the default is True.

    Parameters
    ----------
    true_or_false: {None|bool}
        Pass True to map lstrings to objects; False otherwise. If None
        provided, just returns the current state.

    Return
    ------
    current_stat: bool
        The current state of lstring_as_obj.

    Example
    -------
    ``
>>> from pycbc.io import lscarrays

>>> lscarrays.lstring_as_obj()
    True

>>> lscarrays.LSCArray.from_arrays([numpy.zeros(10)], dtype=[('foo', 'lstring')])
LSCArray([(0.0,), (0.0,), (0.0,), (0.0,), (0.0,), (0.0,), (0.0,), (0.0,),
       (0.0,), (0.0,)], 
      dtype=[('foo', 'O')])

>>> lscarrays.lstring_as_obj(False)
    False

>>> lscarrays.LSCArray.from_arrays([numpy.zeros(10)], dtype=[('foo', 'lstring')])
LSCArray([('0.0',), ('0.0',), ('0.0',), ('0.0',), ('0.0',), ('0.0',),
       ('0.0',), ('0.0',), ('0.0',), ('0.0',)], 
      dtype=[('foo', 'S50')])
``
    """
    if true_or_false is not None:
        _default_types_status['lstring_as_obj'] = true_or_false
        # update the typeDict
        numpy.typeDict[u'lstring'] = numpy.object_ \
            if _default_types_status['lstring_as_obj'] \
            else 'S%i' % _default_types_status['default_strlen']
    return _default_types_status['lstring_as_obj']

def ilwd_as_int(true_or_false=None):
    """
    Similar to lstring_as_obj, sets whether or not ilwd:chars should be
    treated as strings or as ints. Default is True.
    """
    if true_or_false is not None:
        _default_types_status['ilwd_as_int'] = true_or_false
        numpy.typeDict[u'ilwd:char'] = int \
            if _default_types_status['ilwd_as_int'] \
            else 'S%i' % default_strlen
    return _default_types_status['ilwd_as_int']


def default_strlen(strlen=None):
    """
    Sets the default string length for lstring and ilwd:char, if they are
    treated as strings. Default is 50.
    """
    if strlen is not None:
        _default_types_status['default_strlen'] = strlen
        # update the typeDicts as needed
        lstring_as_obj(_default_types_status['lstring_as_obj'])
        set_ilwd_as_int(_default_types_status['ilwd_as_int'])
    return _default_types_status['default_strlen']

# set the defaults
lstring_as_obj(True)
ilwd_as_int(True)


#
# =============================================================================
#
#                           Helper functions 
#
# =============================================================================
#
def get_dtype_descr(dtype):
    """
    Numpy's dtype.descr will return empty void fields if a dtype has
    offsets specified. This function tries to fix that by not including
    fields that have no names and are void types.
    """
    return [dt for dt in dtype.descr if not (dt[0] == '' and dt[1][1] == 'V')]


def combine_fields(dtypes, names=None):
    """Combines the fields in the list of given dtypes into a single dtype.

    Parameters
    ----------
    dtypes: (list of) numpy.dtype(s)
        Either a numpy.dtype, or a list of numpy.dtypes.
    names: {None | (list of) strings}
        Only get the fields in the dtypes that have the given names. Default
        is to get all the names from all of the dtypes.

    Returns
    -------
    combined_dtype: numpy.dtype
        A new dtype combining the fields in the list of dtypes.
    """
    if not isinstance(dtypes, list):
        dtypes = [dtypes]
    # Note: incase any of the dtypes have offsets, we won't include any fields
    # that have no names and are void
    new_dt = numpy.dtype([dt for dtype in dtypes \
        for dt in get_dtype_descr(dtype)])
    if names is not None:
        if isinstance(names, str) or isinstance(names, unicode):
            names = [names]
        # ensure that all of the names are fields in the dtypes
        missing_names = [name for name in names if name not in new_dt.names]
        if any(missing_names):
            raise ValueError("requested name(s) %s are not fields in the " %(
                ', '.join(missing_names)) + "given dtypes")
        # retrieve only the desired names
        new_dt = numpy.dtype([dt for dt in new_dt.descr if dt[0] in names])
    return new_dt


def merge_arrays(merge_list, names=None, flatten=True, outtype=None):
    """
    Merges the given arrays into a single array. The arrays must all have
    the same shape. If one or more of the given arrays has multiple fields,
    all of the fields will be included as separate fields in the new array.

    Parameters
    ----------
    merge_list: list of arrays
        The list of arrays to merge.
    names: {None | sequence of strings}
        Optional, the names of the fields in the output array. If flatten is
        True, must be the same length as the total number of fields in
        merge_list.  Otherise, must be the same length as the number of
        arrays in merge_list.  If None provided, and flatten is True, names
        used will be the same as the name of the fields in the given arrays.
        If the datatype has no name, or flatten is False, the new field will
        be ``'fi'`` where i is the index of the array in arrays.
    flatten: bool
        Make all of the fields in the given arrays separate fields in the
        new array. Otherwise, each array will be added as a field. If an
        array has fields, they will be subfields in the output array. Default
        is True.
    outtype: {None | class}
        Cast the new array to the given type. Default is to return a
        numpy structured array.

    Returns
    -------
    new array: {numpy.ndarray | outtype}
        A new array with all of the fields in all of the arrays merged into
        a single array.
    """
    if not all(merge_list[0].shape == arr.shape for arr in merge_list):
        raise ValueError("all of the arrays in merge_list must have the " +
            "same shape")
    try:
        new_arr = recfunctions.merge_arrays(merge_list, flatten=flatten)
    # merge arrays will fail if there are objects; in that case, we'll do
    # it manually
    except ValueError:
        if flatten:
            new_dt = combine_fields([arr.dtype for arr in merge_list])
        else:
            new_dt = numpy.dtype([('f%i' %ii, arr.dtype.descr) \
                for ii,arr in enumerate(merge_list)])
        new_arr = numpy.empty(merge_list[0].shape, dtype=new_dt)
        for ii,arr in enumerate(merge_list):
            if arr.dtype.names is None:
                new_arr[new_dt.names[ii]] = arr
            else:
                for field in arr.dtype.names:
                    new_arr[field] = arr[field]
    # set the names if desired
    if names is not None:
        new_arr.dtype.names = names
    # ditto the outtype
    if outtype is not None:
        new_arr = new_arr.view(type=outtype)
    return new_arr


def add_fields(input_array, arrays, names=None, assubarray=False):
    """
    Adds the given array(s) as new field(s) to the given input array.
    Returns a new instance of the input_array with the new fields added.

    Parameters
    ----------
    input_array: instance of a numpy.ndarray or numpy recarray
        The array to to add the fields to.
    arrays: (list of) numpy array(s)
        The arrays to add. If adding multiple arrays, must be a list;
        if adding a single array, can just be that array.
    names: (list of) strings
        Optional, the name(s) of the new fields in the output array. If
        adding multiple fields, must be a list of strings with the same
        length as the list of arrays. If None provided, names used will
        be the same as the name of the datatype in the given arrays.
        If the datatype has no name, the new field will be ``'fi'`` where
        i is the index of the array in arrays.
    assubarray: bool
        Add the list of arrays as a single subarray field. If True, and names
        provided, names should be a string or a length-1 sequence. Default is
        False, in which case each array will be added as a separate field.

    Returns
    -------
    new_array: new instance of ``input_array``
        A copy of the ``input_array`` with the desired fields added.
    """
    if not isinstance(arrays, list):
        arrays = [arrays]
    # set the names
    if names is not None:
        if isinstance(names, str) or isinstance(names, unicode):
            names = [names]
        # check if any names are subarray names; if so, we have to add them
        # separately
        subarray_names = [name for name in names if len(name.split('.')) > 1]
    else:
        subarray_names = []
    if any(subarray_names):
        subarrays = [arrays[ii] for ii,name in enumerate(names) \
            if name in subarray_names]
        # group together by subarray
        groups = {}
        for name,arr in zip(subarray_names, subarrays):
            key = name.split('.')[0]
            subkey = '.'.join(name.split('.')[1:])
            try:
                groups[key].append((subkey, arr))
            except KeyError:
                groups[key] = [(subkey, arr)]
        # now cycle over the groups, adding all of the fields in each group
        # as a subarray
        for group_name in groups:
            # we'll create a dictionary out of the subarray field names ->
            # subarrays
            thisdict = dict(groups[group_name])
            # check if the input array has this field; if so, remove it, then
            # add it back with the other new arrays
            if group_name in input_array.columns:
                # get the data
                new_subarray = input_array[group_name]
                # add the new fields to the subarray
                new_subarray = add_fields(new_subarray, thisdict.values(),
                    thisdict.keys())
                # remove the original from the input array
                input_array = input_array.without_fields(group_name)
            else:
                new_subarray = thisdict.values()
            # add the new subarray to input_array as a subarray
            input_array = add_fields(input_array, new_subarray,
                names=group_name, assubarray=True)
        # remove the subarray names from names 
        keep_idx = [ii for ii,name in enumerate(names) \
            if name not in subarray_names]
        names = [names[ii] for ii in keep_idx]
        # if there's nothing left, just return
        if names == []:
            return input_array
        # also remove the subarray arrays
        arrays = [arrays[ii] for ii in keep_idx] 
    if assubarray:
        # merge all of the arrays into a single array
        if len(arrays) > 1:
            arrays = [merge_arrays(arrays, flatten=True)]
        # now merge all the fields as a single subarray
        merged_arr = numpy.empty(len(arrays[0]),
            dtype=[('f0', arrays[0].dtype.descr)])
        merged_arr['f0'] = arrays[0]
        arrays = [merged_arr]
    merge_list = [input_array] + arrays
    if names is not None:
        names = list(input_array.dtype.names) + names
    # merge into a single array
    return merge_arrays(merge_list, names=names, flatten=True,
        outtype=type(input_array))


def get_fields(input_array, names, copy=False, outtype=None):
    """
    Given an array with named fields, creates a new LSCArray with the given
    columns. Only fields in input_array that are in the list of names will be
    extracted. All the fields listed in names must be present in the
    input_array. The method for creating a view is from:
    <http://stackoverflow.com/a/21819324/1366472>

    Parameters
    ----------
    input_array: array-like
        Anything that can be cast as a numpy array. The resulting array
        must have a dtype with at least one field in columns.
    names: {strings|list of strings}
        List of column names for the returned array; may be either a single
        name or a list of names. All of the names must be a field in the input
        array.
    copy: bool, optional
        If True, will force a copy of the input array rather than a view.
    outtype: optional
        Make the output array be the given type. If None, the type of the
        output will be the same as the input.

    Returns
    -------
    output: {type(input_array)|outtype}
        An view or copy of the input array with the given columns.
    """
    if outtype is None:
        outtype = type(input_array)
    if isinstance(names, str) or isinstance(names, unicode):
        names = [names]
    if copy:
        drop_fields = [name for name in input_array.dtype.names \
            if name not in names]
        return recfunctions.rec_drop_fields(input_array, drop_fields).view(
            type=outtype)
    else:
        new_dtype = numpy.dtype({name: input_array.dtype.fields[name] \
            for name in names})
        return numpy.ndarray(input_array.shape, new_dtype,
            input_array, 0, input_array.strides).view(type=outtype)
    

def build_lookup_table(input_array):
    """
    Given an array, builds a dictionary of the values of the array.
    """
    unique_vals, unique_idx, map_idx = numpy.unique(input_array,
        return_index=True, return_inverse=True)
    # if the number of unique values is the same as the number
    # of elements in self, this is a one-to-one mapping
    if unique_vals.shape == input_array.shape:
        try:
            return dict(zip(unique_vals, unique_idx))
        # if unique vals is a list (as in the case when looking up multiple
        # fields), we'll get a TypeError; in that case, cast to tuple
        except TypeError:
            return dict(zip(tuple(unique_vals.tolist()), unique_idx))
    # else, we need to be set each key value pointing to the
    # appropriate indices
    else:
        try:
            return dict([[unique_vals[ii], numpy.where(map_idx == ii)] \
                for ii in range(len(unique_vals))
                ])
        except TypeError:
            return dict([[tuple(unique_vals[ii]), numpy.where(map_idx == ii)]\
                for ii in range(len(unique_vals))
                ])

def join_arrays(this_array, other_array, map_field, expand_field_name=None,
        other_map_field=None, get_fields=None, map_indices=None):
    """

    Joins ``other_array`` to ``this_array`` using the provided map fields.
    The information from ``other_array`` is added to ``this_array`` as a
    subarray. For a given element in ``this_array``, all elements in
    ``other_array`` with
    ``this_array[map_field] == other_array[(other_)map_field]`` are
    retrieved. If multiple elements in ``other_array`` map to a single
    element in ``this_array``, the expanded sub-array will have a shape equal
    to the maximum number of elements in ``other_array`` that map to a single
    element in ``this_array``.

    Parameters
    ----------
    this_array: any subclass of numpy array
        The array to add the fields to.
    other_array: LSCArray or similar
        The array that information is retrieved from. Must have ``lookup`` and
        ``with_fields`` methods.
    map_field: string
        The name of the field in ``this_array`` to use for mapping.
    expand_field_name: string
        The name of the field that will be added to ``this_array``. The
        information from ``other_array`` will be contained as a subarray
        under this field.
    other_map_field: {None | string}
        The name of the field in ``other_array`` to use for mapping. If None,
        ``map_field`` will be used.
    get_fields: {None | (list of) strings}
        Optionally specify what fields to retrieve from ``other_array``. If
        None provided, will get all the fields in ``other_array``.
    map_indices: {None | array of ints}
        If provided, will only map rows in ``this_array`` that have indices in
        the given array of indices. Any rows that are skipped will have a
        zeroed element in the expand field of the returned array. If None (the
        default), all rows in ``this_array`` are mapped.

    Returns
    -------
    new_array: type(this_array)
        A copy of ``this_array`` with the mapped information added to
        ``this_array[expand_field_name]``.
    """
    if other_map_field is None:
        other_map_field = map_field
    # strip off the expand field in this_array, if is in the array
    if expand_field_name is not None:
        new_array = this_array.without_fields(expand_field_name)
    # otherwise, strip off the map field, if it has the same name as the map
    # field in the other array
    elif map_field == other_map_field:
        new_array = this_array.without_fields(map_field)
    else:
        new_array = this_array
    # get a view of the other_array with just the desired fields;
    # note: this will also give a clean lookup table
    if isinstance(get_fields, str) or isinstance(get_fields, unicode):
        get_fields = [get_fields]
    elif get_fields is None:
        get_fields = list(other_array.dtype.names)
    # ensure the map field is included
    if other_map_field not in get_fields:
        get_fields.append(other_map_field)
    # XXX: I've found that copying other array is necessary here, otherwise
    # I get corrupted values in expanded_info, below.
    other_array = other_array.with_fields(get_fields, copy=True)
    # set an empty default in case one or map values in this array is not
    # found in other array
    # Note: for some strange reason, running dtype.descr will yield void fields
    # if with_fields is not a copy, so we'll strip those out
    other_dtdescr = get_dtype_descr(other_array.dtype)
    default = default_empty(1, dtype=other_dtdescr).view(
        type=type(other_array))
    if map_indices is not None:
        # only map rows whose indices are listed in map inidices
        mask = numpy.zeros(this_array.size, dtype=bool)
        mask[map_indices] = True
        expanded_info = [
            other_array.lookup(
                other_map_field, this_array[map_field][ii], default
            ) if applymap else default \
            for ii,applymap in enumerate(mask)]
    else:
        expanded_info = [other_array.lookup(other_map_field, mapval, default) \
            for mapval in this_array[map_field]]
    # need to know the maximum size of the subarray
    maxlen = numpy.array([x.size for x in expanded_info]).max()
    # convert to LSCArray
    if expand_field_name is not None:
        expanded_info = LSCArray.from_records(expanded_info,
            dtype=[(expand_field_name, other_dtdescr, maxlen)])
    else:
        expanded_info = LSCArray.from_records(expanded_info,
            dtype=[(name, dt, maxlen) for (name,dt) in other_dtdescr])
    # add to this_array
    return new_array.add_fields(expanded_info)


#
# =============================================================================
#
#                           Base LSCArray definitions
#
# =============================================================================
#
class LSCArray(numpy.recarray):
    """
    Subclass of numpy.recarray that adds additional functionality.

    Initialization is done the same way as numpy.recarray, with the addition
    that a "name" attribute can be passed to name the output array. When you
    initialize an array it creates a new zeroed array. This is similar to
    numpy.recarray, except that ``numpy.recarray(shape)`` will create an empty
    array, whereas here the default is to zero all of the elements. If you
    prefer an empty array, set ``zero=False`` when initializing.
    
    You cannot pass an array or sequence as input as you do with numpy.array.
    To initialize an LSCArray from an already existing arrays, use the
    ``LSCArray.from_arrays`` class method. To initialize from a list of
    tuples, use ``LSCArray.from_records``. See the docstring for those methods
    for details. For more information on initalizing an empty array, see
    ``numpy.recarray`` help.

    Parameters
    ----------
    shape: int | tuple
        The shape of the new array.
    name: {None | str}
        Optional, what to name the new array. The array's ``name`` attribute
        is set to this.

    For details on other keyword arguments, see ``numpy.recarray`` help.

    Attributes
    ----------
    name: str
        Instance attribute. The name of the array.

    Additional Features
    -------------------
    Arbitrary functions
    +++++++++++++++++++
    You can retrive functions on fields in the same manner that you access
    individual fields. For example, if you have an LSCArray `x` with fields
    'a' and 'b', you can access each field with `x['a'], x['b']`.
    You can also do `x['a*b/(a+b)**2.']`, `x[cos(a)*sin(b)]`, etc. Logical
    operations are also possible, e.g., `x['(a < 3) & (b < 2)']. Syntax
    for functions is python, and any numpy ufunc can be used to operate
    on the fields. Note that while fields may be accessed as
    attributes (e.g, field `a` can be accessed via `x['a']` or `x.a`),
    functions on multiple fields may not; e.g. `x.a+b` does not work, for
    obvious reasons.

    Lookup tables
    +++++++++++++
    A lookup function is provided that allows you to quickly get all rows in
    the array for which a paricular field matches a particular value, e.g.,
    `x.lookup('a', 10.)` will return all rows in `x` for which
    `x['a'] == 10.`.  This is done by building an internal dictionary using
    the requested column as a key the first time it is requested. Since this
    relies on the order of the results, the internal lookup table is not
    passed to views or copies of the array, and it cleared whenever a sort is
    carried out.  The lookup table does increase memory overhead. Also, if
    you change a value in the column that is used as key after the lookup
    table is created, you will get spurious results. For these reasons, a
    clear_lookup method is also provided. See `lookup` and `clear_lookup` for
    details.

    Notes
    -----
    Input arrays with variable-length strings in one or more fields can be
    tricky to deal with. Numpy arrays are designed to use fixed-length
    datasets, so that quick memory access can be achieved. To deal with
    variable-length strings, there are two options: 1. set the data type to
    object, or 2. set the data type to a string with a fixed length larger
    than the longest string in the input array.
    
    The first option, using objects, essentially causes the array to store a
    pointer to the string.  This is the most flexible option, as it allows
    strings in the array to be updated to any length. However, operations on
    object fields are slower, as numpy cannot take advantage of its fast
    memory striding abilities (see `this question/answer on stackoverflow
    <http://stackoverflow.com/a/14639568/1366472>`_ for details). Also,
    numpy's support of object arrays is more limited.  In particular, prior
    to version 1.9.2, you cannot create a view of an array in which the dtype
    is changed that has any fields that are object data types, even if the
    view does not touch the object fields. (This has since been relaxed.)

    The second option, using strings of a fixed length, solves the issues
    with object fields. However, if you try to change one of the strings
    after the array is created, the string will be truncated at whatever
    string length is used. Additionally, if you choose too large of a string
    length, you can substantially increase the memory overhead for large
    arrays.

    This class offers the option to set strings designated as 'lstring' as
    objects, or as fixed-length strings. To toggle what it does use
    lscarrays.set_lstring_as_obj (see the docstring for that function for
    more details).

    Examples
    --------
    * Create an empty array with four rows and two columns called ``'foo'`` and
    ``'bar'``, both of which are floats:
``
>>> x = LSCArray(4, dtype=[('foo', float), ('bar', float)])
``

    * Set/retrieve a column using index or attribute syntax:
``
>>> x['foo'] = [1.,2.,3.,4.]

>>> x.bar = [5.,6.,7.,8.]

>>> x
    
LSCArray([(1.0, 5.0), (2.0, 6.0), (3.0, 7.0), (4.0, 8.0)], 
      dtype=[('foo', '<f8'), ('bar', '<f8')])

>>> x.foo
    array([ 1.,  2.,  3.,  4.])

>>> x['bar']
    array([ 5.,  6.,  7.,  8.])
``
    
    Get the names of the columns:
``
>>> x.columns
    ('foo', 'bar')
``

    * Rename the columns to ``a`` and ``b``:
``
>>> x.dtype.names = ['a', 'b']

>>> x.columns
    ('a', 'b')
``

    * Retrieve a function of the columns as if it were a column:
``
>>> x['sin(a/b)']
array([ 0.19866933,  0.3271947 ,  0.41557185,  0.47942554])
``

    * Load from a list of arrays (in this case, from an hdf5 file):
``
>>> f = h5py.File('bank/H1L1-BANK2HDF-1117400416-928800.hdf', 'r')

>>> f.keys()
    [u'mass1', u'mass2', u'spin1z', u'spin2z', u'template_hash']

>>> templates = LSCArray.from_arrays(f.values(), names=f.keys())

>>> templates.mass1
array([ 1.71731389,  1.10231435,  2.99999857, ...,  1.67488706,
        1.00531888,  2.11106491], dtype=float32)

>>> templates[['mass1', 'spin1z']]
array([(1.7173138856887817, 0.0), (1.1023143529891968, 0.0),
       (2.9999985694885254, 0.0), ..., (1.6748870611190796, 0.0),
       (1.0053188800811768, 0.0), (2.111064910888672, 0.0)], 
      dtype=[('mass1', '<f4'), ('spin1z', '<f4')])
``

    * Convert a LIGOLW xml table:
``
>>> type(sim_table)
glue.ligolw.lsctables.SimInspiralTable

>>> sim_array = LSCArray.from_ligolw_table(sim_table)

>>> sim_array.mass1
array([ 5.94345808,  3.34226608,  3.73025393, ...,  4.75996208,
        4.23756123,  4.62750006], dtype=float32)

>>> sim_array.waveform
chararray(['SEOBNRv2pseudoFourPN', 'SEOBNRv2pseudoFourPN',
       'SEOBNRv2pseudoFourPN', ..., 'SEOBNRv2pseudoFourPN',
       'SEOBNRv2pseudoFourPN', 'SEOBNRv2pseudoFourPN'], 
      dtype='|S22')
``
    
    * Make lstrings be objects instead of strings:
``
>>> lscarrays.set_lstring_as_obj(True)

>>> sim_array = LSCArray.from_ligolw_table(sim_table)

>>> sim_array.waveform
array([u'SEOBNRv2pseudoFourPN', u'SEOBNRv2pseudoFourPN',
       u'SEOBNRv2pseudoFourPN', ..., u'SEOBNRv2pseudoFourPN',
       u'SEOBNRv2pseudoFourPN', u'SEOBNRv2pseudoFourPN'], dtype=object)
``
    
    * Only view a few of the columns:
``
>>> sim_array.with_fields(['simulation_id', 'mass1', 'mass2'])
LSCArray([(0, 5.943458080291748, 3.614427089691162),
       (1, 3.342266082763672, 3.338679075241089),
       (2, 3.7302539348602295, 6.2023820877075195), ...,
       (9236, 4.75996208190918, 4.678255081176758),
       (9237, 4.237561225891113, 2.1022119522094727),
       (9238, 4.627500057220459, 3.251383066177368)], 
      dtype={'names':['simulation_id','mass1','mass2'], 'formats':['<i8','<f4','<f4'], 'offsets':[200,236,240], 'itemsize':244})
``

    ...or just retrieve a few of the columns to begin with:
``
>>> sim_array = LSCArray.from_ligolw_table(sim_table, columns=['simulation_id', 'mass1', 'mass2'])

>>> sim_array
LSCArray([(0, 5.943458080291748, 3.614427089691162),
       (1, 3.342266082763672, 3.338679075241089),
       (2, 3.7302539348602295, 6.2023820877075195), ...,
       (9236, 4.75996208190918, 4.678255081176758),
       (9237, 4.237561225891113, 2.1022119522094727),
       (9238, 4.627500057220459, 3.251383066177368)], 
      dtype=[('simulation_id', '<i8'), ('mass1', '<f4'), ('mass2', '<f4')])
``

    * Add a field to the array:
``
>>> optimal_snrs = numpy.random.uniform(4.,40., size=len(sim_array))

>>> sim_array.optimal_snr
array([ 11.16132242,  21.05959836,  27.543917  , ...,  35.48635821,
        23.8457928 ,  14.93213006])
``

    * Create an array with nested fields:
``
>>> more_info = LSCArray(len(sim_array), dtype=[('site_params', [('ifos', 'S2'), ('eff_dists', float)], 2)])

>>> more_info.site_params.ifos = ['H1', 'L1']

>>> more_info.site_params.eff_dists[:,0] = [row.eff_dist_h for row in sim_table]

>>> more_info.site_params.eff_dists[:,1] = [row.eff_dist_l for row in sim_table]

>>> more_info.site_params.ifos
chararray([['H1', 'L1'],
       ['H1', 'L1'],
       ['H1', 'L1'],
       ..., 
       ['H1', 'L1'],
       ['H1', 'L1'],
       ['H1', 'L1']], 
      dtype='|S2')

>>> more_info.site_params.eff_dists
array([[  395.0081,   405.4235],
       [  467.1236,   333.3516],
       [  470.7249,   543.3748],
       ..., 
       [ 2424.947 ,  2011.257 ],
       [  354.6583,   483.3423],
       [ 1148.264 ,  1093.221 ]])
``
    
    * Add the nested fields to another array:
``
>>> sim_array = sim_array.add_fields(more_info)

>>> sim_array.site_params
LSCArray([[('H1', 395.0081), ('L1', 405.4235)],
       [('H1', 467.1236), ('L1', 333.3516)],
       [('H1', 470.7249), ('L1', 543.3748)],
       ..., 
       [('H1', 2424.947), ('L1', 2011.257)],
       [('H1', 354.6583), ('L1', 483.3423)],
       [('H1', 1148.264), ('L1', 1093.221)]], 
      dtype=[('ifos', 'S2'), ('eff_dists', '<f8')])
``

    * Sort by one of the fields (in this case, decisive distance):
``
>>> sort_idx = numpy.argsort(sim_array['site_params.eff_dists'].min(axis=1))

>>> sim_array[sort_idx]
LSCArray([ (6812, 2.0183539390563965, 2.2284369468688965, 28.000550054717195, [('H1', 14.89811), ('L1', 21.8963)]),
       (1387, 2.7829830646514893, 2.214693069458008, 29.092840816763346, [('H1', 18.14495), ('L1', 23.38128)]),
       (2928, 3.3857719898223877, 2.5695199966430664, 31.27512113467619, [('H1', 19.6899), ('L1', 20.47152)]),
       ...,
       (858, 3.4494340419769287, 5.6560869216918945, 39.45165010161026, [('H1', 9966.197), ('L1', 11871.26)]),
       (261, 2.9730420112609863, 4.851998805999756, 19.983317475299707, [('H1', 12378.11), ('L1', 10544.48)]),
       (8619, 2.506516933441162, 2.3052260875701904, 33.229039459442454, [('H1', 20041.22), ('L1', 17056.87)])], 
      dtype=[('simulation_id', '<i8'), ('mass1', '<f4'), ('mass2', '<f4'), ('optimal_snr', '<f8'), ('site_params', [('ifos', 'S2'), ('eff_dists', '<f8')], (2,))])
``
    """
    str_as_obj = False
    __persistent_attributes__ = ['name']

    def __new__(cls, shape, name=None, set_default_empty=True, **kwargs):
        """
        Initializes a new empty array.
        """
        obj = super(LSCArray, cls).__new__(cls, shape, **kwargs).view(
            type=cls)
        obj.name = name
        obj.__persistent_attributes__ = cls.__persistent_attributes__
        # zero out the array if desired
        if set_default_empty:
            default = default_empty(1, dtype=obj.dtype)
            obj[:] = default
        return obj


    def __array_finalize__(self, obj):
        """
        Default values are set here.
        """
        if obj is None:
            return
        # copy persisitent attributes
        [setattr(self, attr, getattr(obj, attr, None)) for attr in \
            self.__persistent_attributes__]
        # numpy has some issues with dtype field names that are unicode,
        # so we'll force them to strings here
        if obj.dtype.names is not None and \
                any([isinstance(name, unicode) for name in obj.dtype.names]):
            obj.dtype.names = map(str, obj.dtype.names)
        self.__lookuptable__ = {}


    def __copy_attributes__(self, other, default=None):
        """
        Copies the values of all of the attributes listed in
        self.__persistent_attributes__ to other.
        """
        [setattr(other, attr, getattr(self, attr, default)) \
            for attr in self.__persistent_attributes__]


    def __setitem__(self, item, values):
        """
        Wrap's recarray's setitem to allow attribute-like indexing when
        setting values.
        """
        try:
            return super(LSCArray, self).__setitem__(item, values)
        except ValueError:
            # we'll get a ValueError if a subarray is being referenced using
            # '.'; so we'll try to parse it out here
            fields = item.split('.')
            if len(fields) > 1:
                for field in fields[:-1]:
                    self = self[field]
                item = fields[-1]
            # now try again
            return super(LSCArray, self).__setitem__(item, values)


    def __getitem__(self, item):
        """
        Wraps recarray's  __getitem__ so that math functions on columns can be
        retrieved. Any function in numpy's library may be used.
        """
        try:
            return super(LSCArray, self).__getitem__(item)
        except ValueError:
            # arg isn't a simple argument of row, so we'll have to eval it
            item_dict = dict([ [col, super(LSCArray, self).__getitem__(col)]\
                for col in self.dtype.names])
            # add any aliases
            item_dict.update({alias: item_dict[name] \
                for alias,name in self.aliases.items()})
            safe_dict = {}
            safe_dict.update(item_dict)
            # add self's funciton dict
            safe_dict.update(numpy.__dict__)
            return eval(item, {"__builtins__": None}, safe_dict)

    def addattr(self, attrname, value=None, persistent=True):
        """
        Adds an attribute to self. If persistent is True, the attribute will
        be made a persistent attribute. Persistent attributes are copied
        whenever a view or copy of this array is created. Otherwise, new views
        or copies of this will not have the attribute.
        """
        setattr(self, attrname, value)
        # add as persistent
        if persistent and attrname not in self.__persistent_attributes__:
            self.__persistent_attributes__.append(attrname)

    @classmethod
    def from_arrays(cls, arrays, name=None, **kwargs):
        """
        Creates a new instance of self from the given (list of) array(s). This
        is done by calling numpy.rec.fromarrays on the given arrays with the
        given kwargs. The type of the returned array is cast to this class,
        and the name (if provided) is set.

        Parameters
        ----------
        arrays: (list of) numpy array(s)
            A list of the arrays to create the LSCArray from.
        name: {None|str}
            What the output array should be named.

        For other keyword parameters, see the numpy.rec.fromarrays help.

        Returns
        -------
        array: instance of this class
            An array that is an instance of this class in which the field
            data is from the given array(s).
        """
        obj = numpy.rec.fromarrays(arrays, **kwargs).view(type=cls)
        obj.name = name
        return obj

    @classmethod
    def from_records(cls, records, name=None, **kwargs):
        """
        Creates a new instance of self from the given (list of) record(s). A
        "record" is a tuple in which each element is the value of one field
        in the resulting record array. This is done by calling
        numpy.rec.fromrecordss on the given records with the given kwargs.
        The type of the returned array is cast to this class, and the name
        (if provided) is set.

        Parameters
        ----------
        records: (list of) tuple(s)
            A list of the tuples to create the LSCArray from.
        name: {None|str}
            What the output array should be named.

        For other keyword parameters, see the numpy.rec.fromrecords help.

        Returns
        -------
        array: instance of this class
            An array that is an instance of this class in which the field
            data is from the given record(s).
        """
        obj = numpy.rec.fromrecords(records, **kwargs).view(
            type=cls)
        obj.name = name
        return obj

    @classmethod
    def from_ligolw_table(cls, table, columns=None, dtype=None):
        """
        Converts the given ligolw table into an LSCArray. The tableName
        attribute is copied to the array's name.

        Parameters
        ----------
        table: LIGOLw table instance
            The table to convert.
        columns: {None|list}
            Optionally specify a list of columns to retrieve. All of the
            columns must be in the table's validcolumns attribute. If None
            provided, all the columns in the table will be converted.
        dtype: {None | numpy.dtype readable}
            Override the column's dtype to dtype.

        Returns
        -------
        array: LSCArray
            The input table as an LSCArray.
        """
        name = table.tableName.split(':')[0]
        if columns is None:
            # get all the columns
            columns = table.validcolumns.items()
        else:
            # note: this will raise a KeyError if one or more columns is
            # not in the table's validcolumns
            columns = {col: table.validcolumns[col] \
                for col in columns}.items()
        # get the values
        if _default_types_status['ilwd_as_int']:
            input_array = [tuple(
                        getattr(row, col) if dt != 'ilwd:char' \
                        else int(getattr(row, col)) \
                    for col,dt in columns) \
                for row in table]
        else:
            input_array = [tuple(getattr(row, col) for col,_ in columns) \
                for row in table]
        if dtype is None:
            dtype = columns
        # return the values as an instance of cls
        return cls.from_records(input_array, dtype=dtype,
            name=name)

    @property
    def columns(self):
        """
        Returns a tuple listing the field names in self. Equivalent to
        ``array.dtype.names``, where ``array`` is self.
        """
        return self.dtype.names

    @property
    def aliases(self):
        """
        Returns a dictionary of the aliases, or "titles", of the field names
        in self. An alias can be specified by passing a tuple in the name
        part of the dtype. For example, if an array is created with
        ``dtype=[(('foo', 'bar'), float)]``, the array will have a field
        called ``bar`` that has alias ``foo`` that can be accessed using
        either ``arr['foo']`` or ``arr['bar']``. Note that the first string
        in the dtype is the alias, the second the name. This function returns
        a dictionary in which the aliases are the keys and the names are the
        values. Only fields that have aliases are returned.
        """
        return dict(c[0] for c in self.dtype.descr if isinstance(c[0], tuple))


    def add_fields(self, arrays, names=None, assubarray=False):
        """
        Adds the given arrays as new fields to self.  Returns a new instance
        with the new fields added. Note: this array does not change; the
        returned array is a new copy. The internal lookup table of the new
        array will also be empty.

        Parameters
        ----------
        arrays: (list of) numpy array(s)
            The arrays to add. If adding multiple arrays, must be a list;
            if adding a single array, can just be that array.
        names: (list of) strings
            Optional, the name(s) of the new fields in the output array. If
            adding multiple fields, must be a list of strings with the same
            length as the list of arrays. If None provided, names used will
            be the same as the name of the datatype in the given arrays.
            If the datatype has no name, the new field will be ``'fi'`` where
            i is the index of the array in arrays.
        assubarray: bool
            Add the list of arrays as a single subarray field. If True, and
            names provided, names should be a string or a length-1 sequence.
            Default is False, in which case each array will be added as a
            separate field.

        Returns
        -------
        new_array: new instance of this array
            A copy of this array with the desired fields added.
        """
        newself = add_fields(self, arrays, names=names, assubarray=assubarray)
        self.__copy_attributes__(newself)
        return newself


    def with_fields(self, names, copy=False):
        """
        Get a view/copy of this array with only the specified fields.

        Parameters
        ----------
        names: {strings|list of strings}
            List of column names for the returned array; may be either a
            single name or a list of names.
        copy: bool, optional
            If True, will return a copy of the array rather than a view.
            Default is False.

        Returns
        -------
        lscarray: new instance of this array
            The view or copy of the array with only the specified fields.
        """
        newself = get_fields(self, names, copy=copy)  
        # copy relevant attributes
        self.__copy_attributes__(newself)
        return newself


    def without_fields(self, names, copy=False):
        """
        Get a view/copy of this array without the specified fields.

        Parameters
        ----------
        names: {strings|list of strings}
            List of column names for the returned array; may be either a
            single name or a list of names.
        copy: bool, optional
            If True, will return a copy of the array rather than a view.
            Default is False.

        Returns
        -------
        lscarray: new instance of this array
            The view or copy of the array without the specified fields.
        """
        # check if only given a single name
        if isinstance(names, str) or isinstance(names, unicode):
            names = [names]
        # cycle over self's fields, excluding any names in names
        keep_names = [name for name in self.dtype.names if name not in names]
        newself = get_fields(self, keep_names, copy=copy)  
        # copy relevant attributes
        self.__copy_attributes__(newself)
        return newself


    def lookup(self, column, value, default=KeyError):
        """
        Returns the elements in self for which the given column(s) matches the
        given value(s). If this is the first time that the column(s) has been
        requested, an internal dictionary is built first, then the elements
        returned. If the value(s) cannot be found, a KeyError is raised, or
        a default is returned if provided. To lookup multiple columns, pass
        the column names and the corresponding values as tuples.
        
        .. note::
            Every time a lookup on a new column is done, an array of indices
            is created that maps every unique value in the column to the
            indices in self that match that value. Since this mapping will
            change if the array is sorted, or if a slice of self is created,
            this look up table is not passed to new views of self. To reduce
            memory overhead, run clear_lookup on a column if you will no
            longer need to look up self using that column. See clear_look up
            for details.

        .. warning::
            If you change the value of an item that is used as a look
            up item, the internal lookup dictionary will not give correct
            values. Always run clear_lookup if you change the value of
            a column that you intend to use as a lookup key.

        Parameters
        ----------
        column: (tuple of) string(s)
            The name(s) of the column to look up.
        value: (tuple of) value(s)
            The value(s) in the columns to get.
        default: {KeyError | value}
            Optionally specify a value to return is the value is not found
            in self's column. Otherwise, a KeyError is raised.

        Returns
        -------
        matching: type(self)
            The rows in self that match the requested value.
        """
        try:
            return self[self.__lookuptable__[column][value]]
        except KeyError:
            # build the look up for this column
            if column not in self.__lookuptable__:
                # if column is a joint column, convert to list
                if isinstance(column, tuple):
                    colvals = self[list(column)]
                else:
                    colvals = self[column]
                self.__lookuptable__[column] = build_lookup_table(colvals)
            # try again
            try:
                return self[self.__lookuptable__[column][value]]
            except KeyError as err:
                if inspect.isclass(default) and issubclass(default, Exception):
                    raise default(err.message)
                else:
                    return default


    def clear_lookup(self, column=None):
        """
        Clears the internal lookup table used to provide fast lookup of
        the given column. If no column is specified, the entire table will
        be deleted. Run this if you will no longer need to look up self
        using a particular column (or, if column is None, any look ups at
        all). Note: if you clear the lookup table of a column, then try to
        lookup that column again, a new lookup table will be created.
        """
        if column is None:
            self.__lookuptable__.clear()
        else:
            self.__lookuptable__[column].clear()
            del self.__lookuptable__[column]


    def sort(self, *args, **kwargs):
        """
        Clears self's lookup table before sorting. See numpy.ndarray.sort for
        help.
        """
        self.clear_lookup()
        super(LSCArray, self).sort(*args, **kwargs)


    def join(self, other, map_field, expand_field_name=None,
            other_map_field=None, get_fields=None, map_indices=None):
        """
        Join another array to this array such that:
        ``self[map_field]`` == ``other[other_map_field]``. The fields from
        ``other`` are added as a sub-array to self with field name
        ``expand_field_name``.

        Parameters
        ----------
        other: LSCArray or similar
            The array that information is retrieved from. Must have
            ``lookup`` and ``with_fields`` methods.
        map_field: string
            The name of the field in self to use for mapping.
        expand_field_name: string
            The name of the field that will be added to self. The information
            from ``other`` will be contained as a subarray under this field.
        other_map_field: {None|string}
            The name of the field in ``other`` to use for mapping. If None
            provided, ``map_field`` will be used.
        get_fields: {None | (list of) strings}
            Optionally specify what fields to retrieve from ``other_array``.
            If None provided, will get all the fields in ``other_array``.
        map_indices: {None | array of ints}
            If provided, will only map rows in this array that have indices
            in the given array of indices. Any rows that are skipped will
            have a zeroed element in the expand field of the returned array.
            If None (the default), all rows in this array are mapped.

        Returns
        -------
        new_array: type(self)
            A copy of this array with the mapped information added to
            ``expand_field_name``.
        """
        return join_arrays(self, other, map_field,
            expand_field_name=expand_field_name,
            other_map_field=other_map_field, get_fields=get_fields,
            map_indices=map_indices)


def aliases_from_fields(fields):
    """
    Given a dictionary of fields, will return a dictionary mapping the aliases
    to the names.
    """
    return dict(c for c in fields if isinstance(c, tuple))


def fields_from_names(fields, names=None):
    """
    Given a dictionary of fields and a list of names, will return a dictionary
    consisting of the fields specified by names. Names can be either the names
    of fields, or their aliases.
    """
    if names is None:
        return fields
    aliases_to_names = aliases_from_fields(fields)
    names_to_aliases = dict(zip(aliases_to_names.values(),
        aliases_to_names.keys()))
    outfields = {}
    for name in names:
        try:
            outfields[name] = fields[name]
        except KeyError:
            if name in aliases_to_names:
                key = (name, aliases_to_names[name])
            elif name in names_to_aliases:
                key = (names_to_aliases[name], name)
            else:
                raise KeyError('default fields has no field %s' % name)
            outfields[key] = fields[key]
    return outfields

class _LSCArrayWithDefaults(LSCArray):
    """
    Subclasses LSCArray, adding class method ``default_fields`` and class
    attribute ``default_name``. If no name is provided when initialized, the
    ``default_name`` will be used. Likewise, if no dtype is provided when
    initalized, the default fields will be used. If names are provided, they
    must be names or aliases that are in default fields, else a KeyError is
    raised. Non-default fields can be created by specifying dtype directly. 

    The default ``default_name`` is None and ``default_fields`` returns an
    empty dictionary. This class is mostly meant to be subclassed by other
    classes, so they can add their own defaults.
    """
    default_name = None

    @classmethod
    def default_fields(cls, **kwargs):
        """
        The default fields. This function should be overridden by subclasses
        to return a dictionary of the desired default fields. Allows for
        key word arguments to be passed to it, for classes that need to be
        able to alter properties of some of the default fields.
        """
        return {}

    def __new__(cls, shape, name=None, field_args={}, **kwargs):
        """
        Makes use of cls.default_fields and cls.default_name.
        """
        if 'names' in kwargs and 'dtype' in kwargs:
            raise ValueError("Please provide names or dtype, not both")
        fields = cls.default_fields(**field_args)
        if 'names' in kwargs:
            names = kwargs.pop('names')
            if isinstance(names, str) or isinstance(names, unicode):
                names = [names]
            kwargs['dtype'] = fields_from_names(fields, names).items()
        if 'dtype' not in kwargs:
            kwargs['dtype'] = fields.items()
        return super(_LSCArrayWithDefaults, cls).__new__(cls, shape,
            name=None, **kwargs)



#
# =============================================================================
#
#                           Default LSCArrays
#
# =============================================================================
#

class Waveform(_LSCArrayWithDefaults):
    """
    Subclasses LSCArrayWithDefaults, with default name ``waveform``. Has
    class attributes instrinsic_params, extrinsic_params, and waveform_params.
    These are concatenated together to make the default columns.

    Also adds various common functions decorated as properties, such as
    mtotal = mass1+mass2.
    """
    default_name = 'waveform'

    # _static_fields are fields that are automatically inherited by
    # subclasses of this class
    _static_fields = {
        # intrinsic parameters
        ('m1', 'mass1'): float,
        ('m2', 'mass2'): float,
        ('s1x', 'spin1x'): float,
        ('s1y', 'spin1y'): float,
        ('s1z', 'spin1z'): float,
        ('s2x', 'spin2x'): float,
        ('s2y', 'spin2y'): float,
        ('s2z', 'spin2z'): float,
        'lambda1': float,
        'lambda2': float,
        'quadparam1': float,
        'quadparam2': float,
        'eccentricity': float,
        'argument_periapsis': float,
        # extrinsic parameters
        'phi_ref': float,
        ('inc', 'inclination'): float,
        # waveform parameters
        'sample_rate': int,
        'segment_length': int,
        'f_min': float,
        'f_ref': float,
        'f_max': float,
        'duration': float,
        'amp_order': int,
        'phase_order': int,
        'spin_order': int,
        'tidal_order': int,
        ('apprx', 'approximant'): 'lstring',
        'taper': 'lstring',
        'frame_axis': 'lstring',
        'modes_choice': 'lstring',
        }

    @classmethod
    def default_fields(cls):
        return cls._static_fields 

    # some other derived parameters
    @property
    def mtotal(self):
        return self.mass1 + self.mass2

    @property
    def mtotal_s(self):
        return lal.MTSUN_SI*self.mtotal

    @property
    def q(self):
        return self.mass1 / self.mass2

    @property
    def eta(self):
        return self.mass1*self.mass2 / self.mtotal**2.

    @property
    def mchirp(self):
        return self.eta**(3./5)*self.mtotal

    def tau0(self, f0=None):
        """
        Returns tau0. If f0 is not specified, uses self.f_min.
        """
        if f0 is None:
            f0 = self.f_min
        return (5./(256 * numpy.pi * f0 * self.eta)) * \
            (numpy.pi * self.mtotal_s * f0)**(-5./3.)
   
    def v0(self, f0=None):
        """
        Returns the velocity at f0, as a fraction of c. If f0 is not
        specified, uses self.f_min.
        """
        if f0 is None:
            f0 = self.f_min
        return (2*numpy.pi* f0 * self.mtotal_s)**(1./3)

    @property
    def s1(self):
        return numpy.array([self.spin1x, self.spin1y, self.spin1z]).T

    @property
    def s2(self):
        return numpy.array([self.spin2x, self.spin2y, self.spin2z]).T

    @property
    def s1mag(self):
        return numpy.sqrt((self.s1**2).sum(axis=1))

    @property
    def s2mag(self):
        return numpy.sqrt((self.s2**2).sum(axis=1))


class TmpltInspiral(Waveform):
    """
    Subclasses Waveform, with default name ``tmplt_inspiral``. Adds attributes
    ids and ifo_params; the columns defined in those are added to Waveform's
    default columns.

    Notes
    -----
    The attribute ``ifo_params`` defines the ``ifo`` column, which is a
    subarray. The default length is 1; but this can be changed using the
    ``default_nifos`` class attribute. If a larger number, a single row
    of a TmpltInspiral can store information about one or more ifos.

    Examples
    --------
    * Create a TmpltInspiral array from an hdf bank file:
``
>>> bankhdf = h5py.File('H1L1-BANK2HDF-1117400416-928800.hdf', 'r')

>>> templates = TmpltInspiral.from_arrays(bankhdf.values(), names=bankhdf.keys())

>>> templates = templates.add_fields(numpy.arange(len(templates)), names='template_id')

>>> templates.columns
    ('mass1', 'mass2', 'spin1z', 'spin2z', 'template_hash', 'template_id')

>>> templates.mass1
array([ 1.71731389,  1.10231435,  2.99999857, ...,  1.67488706,
        1.00531888,  2.11106491], dtype=float32)

>>> templates[['mass1', 'mass2']]
array([(1.7173138856887817, 1.2124452590942383),
       (1.1023143529891968, 1.0074082612991333),
       (2.9999985694885254, 1.0578444004058838), ...,
       (1.6748870611190796, 1.1758257150650024),
       (1.0053188800811768, 1.0020891427993774),
       (2.111064910888672, 1.0143394470214844)], 
      dtype=[('mass1', '<f4'), ('mass2', '<f4')])
``
    """
    default_name = 'tmplt_inspiral'

    @classmethod
    def default_fields(cls, nifos=1):
        """
        Admits argument nifos, which is used to set the size of the ifo
        sub-array.
        """
        fields = {
            'template_id': int,
            'process_id': int,
            # Note: ifo is a subarray with length set by nifos
            'ifo': ('S2', nifos),
            }
        # Note: cls._static_fields is Waveform's static fields
        return dict(cls._static_fields.items() + fields.items())

    def __new__(cls, shape, name=None, nifos=1, **kwargs):
        """
        Adds nifos to initialization.
        """
        return super(TmpltInspiral, cls).__new__(cls, shape, name=name, 
            field_args={'nifos': nifos}, **kwargs)


class SnglEvent(_LSCArrayWithDefaults):
    """
    Subclasses _LSCArrayWithDefaults, adding ranking_stat, snr, chisq, and
    end time as default columns. Also has a 'template' column, which is a
    sub-array of with fields ifo and template_id. If given a TmpltInspiral
    array, the template field can be expanded to have all the parameters of
    each single event; see expand_templates for details.

    Examples
    --------
    * Create a SngEvent array from an hdfcoinc merged file:
``
>>> hdf = h5py.File('H1-HDF_TRIGGER_MERGE_BNS1INJ-1117400416-928800.hdf', 'r')['H1']

>>> hsngls = lscarrays.SnglEvent(len(hdf['end_time']))

>>> hsngls['event_id'] = numpy.arange(hsngls.size)

>>> hsngls['snr'] = hdf['snr']

>>> hsngls['chisq'] = hdf['chisq']

>>> hsngls['chisq_dof'] = 2.*hdf['chisq_dof'].value - 2

>>> hsngls['end_time'] = hdf['end_time']

>>> hsngls['template']['template_id'] = hdf['template_id']

>>> hsngls['ifo'] = 'H1'

>>> hsngls['sigma'] = numpy.sqrt(hdf['sigmasq'])

>>> hsngls['ranking_stat'] = hsngls.snr

>>> reweight_idx = numpy.where(hsngls['chisq/chisq_dof > 1'])

>>> hsngls.ranking_stat[reweight_idx] = hsngls['snr'][reweight_idx] / ((1. + hsngls['chisq/chisq_dof'][reweight_idx]**3.)/2.)**(1./6)
``

    * Get the ranking stat by its alias:
``
>>> hsngls.ranking_stat
array([ 9.59688939,  5.25923089,  5.67155045, ...,  5.08446111,
        7.24139452,  7.55083953])

>>> hsngls.ranking_stat_alias
    'new_snr'

>>> hsngls.new_snr
array([ 9.59688939,  5.25923089,  5.67155045, ...,  5.08446111,
        7.24139452,  7.55083953])


    * Expand the template field (see TmpltInspiral help for how to create
    ``templates`` from an hdf bank file):
``
>>> hsngls.template.columns
    ('template_id',)

>>> templates.columns
    ('mass1', 'mass2', 'spin1z', 'spin2z', 'template_hash', 'template_id')

>>> hsngls = hsngls.expand_templates(templates)

>>> hsngls.template.columns
    ('mass1', 'mass2', 'spin1z', 'spin2z', 'template_hash', 'template_id')

>>> hsngls.ranking_stat, hsngls.template.template_id, hsngls.template.mass1
(array([ 9.59688939,  5.25923089,  5.67155045, ...,  5.08446111,
         7.24139452,  7.55083953]),
 array([   0,    0,    0, ..., 4030, 4030, 4030]),
 array([ 1.71731389,  1.71731389,  1.71731389, ...,  2.11106491,
         2.11106491,  2.11106491], dtype=float32))
``

    * Sort by the ranking stat (note that all other fields are sorted too):
``
>>> hsngls.sort(order='ranking_stat')

>>> hsngls.ranking_stat, hsngls.template.template_id, hsngls.template.mass1
(array([   5.00000016,    5.00000954,    5.00000962, ...,  116.15787474,
         124.83552448,  155.72376072]),
 array([3785,  590, 2314, ..., 2473, 3159,   86]),
 array([ 2.99986959,  1.85319662,  1.41802227, ...,  2.99999189,
         1.87226772,  2.88923597], dtype=float32))
``
    """
    default_name = 'sngl_event'
    ranking_stat_alias = 'new_snr'

    # we define the following as static parameters because they are inherited
    # by CoincEvent
    _static_fields = {
        # ids
        'process_id': int,
        'event_id': int,
        'template_id': int,
        # end times
        'end_time_s': "int_4s",
        'end_time_ns': "int_4s",
        }

    @classmethod
    def default_fields(cls, ranking_stat_alias='new_snr'):
        """
        The ranking stat alias can be set; default is ``'new_snr'``.
        """
        fields = {
            ('ifo', 'detector'): 'S2',
            'sigma': float,
            (ranking_stat_alias, 'ranking_stat'): float,
            'snr': float,
            'chisq': float,
            'chisq_dof': float,
            'bank_chisq': float,
            'bank_chisq_dof': float,
            'cont_chisq': float,
            'cont_chisq_dof': float,
            }
        return dict(cls._static_fields.items() + fields.items())

    def __new__(cls, shape, name=None, ranking_stat_alias='new_snr', **kwargs):
        """
        Adds ranking_stat_alias to initialization.
        """
        field_args = {'ranking_stat_alias': ranking_stat_alias}
        return super(SnglEvent, cls).__new__(cls, shape, name=name,
            field_args=field_args, **kwargs)

    @property
    def end_time(self):
        return self.end_time_s + 1e-9*self.end_time_ns

    def expand_templates(self, tmplt_inspiral_array, get_fields=None,
            selfs_map_field='template_id', tmplts_map_field='template_id'):
        """
        Given an array of templates, replaces the template column with a
        sub-array of the template data. This is done by getting all rows in
        the ``tmplt_inspiral_array`` such that:
        ``self[selfs_map_field] == tmplt_inspiral_array[tmplts_map_field]``.

        Parameters
        ----------
        tmplt_inspiral_array: (any subclass of) LSCArray
            The array of templates with additional fields to add.
        get_fields: {None | (list of) strings}
            The names of the fields to get from the tmplt_inspiral_array.
            If ``None``, all fields will be retrieved.
        selfs_map_field: {'template.template_id' | string}
            The name of the field in self's current template sub-array to use
            to match to templates in the tmplt_inspiral_array. Default is
            ``template.template_id``.
        tmplts_map_field: {'template_id' | string}
            The name of the field in the tmplt_inspiral_array to match.
            Default is ``template_id``.

        Returns
        -------
        new array: new instances of this array
            A copy of this array with the template sub-array containing all
            of the fields specified by ``get_fields``.
        """
        return self.join(tmplt_inspiral_array, selfs_map_field,
            other_map_field=tmplts_map_field, get_fields=get_fields)


class CoincEvent(SnglEvent):
    """
    Subclasses SnglEvent, but with different default stat_params. Also has a
    'sngl_events', which is a subarray with fields named by instruments. The
    default is to store ``event_id`` and ``end_time`` for each instrument,
    but this can be expanded to have all of the full parameters of the single
    events; see expand_sngls for details.

    Examples
    --------
    * Create an array from a statmap hdfcoinc file:
``
>>> coincs = CoincEvent(len(fg['fap']), names=['fap', 'far', 'ranking_stat', 'template', 'event_id', 'sngl_events'])

>>> coincs['fap'] = fg['fap']

>>> coincs['far'] = 1./fg['ifar'].value

>>> coincs['ranking_stat'] = fg['stat']

>>> coincs['template']['template_id'] = fg['template_id']

>>> coincs['event_id'] = numpy.arange(len(coincs))

>>> coincs['sngl_events']['ifos'] = [statmap.attrs['detector_2'], statmap.attrs['detector_1']]

>>> coincs['sngl_events']['event_ids'] = numpy.vstack((fg['trigger_id2'], fg['trigger_id1'])).T
``

    * Add information about the single events (see SnglEvent help for how to
      create ``hsngls`` and ``lsngls`` in this example):
``
>>> coincs.columns
    ('fap', 'event_id', 'ranking_stat', 'template', 'sngl_events', 'far')

>>> coincs.sngl_events.ifos
chararray([['H1', 'L1'],
       ['H1', 'L1'],
       ..., 
       ['H1', 'L1']], 
      dtype='|S2')

>>> hsngls.ifo
chararray(['H1', 'H1', 'H1', ..., 'H1', 'H1', 'H1'], 
      dtype='|S2')

>>> coincs = coincs.add_sngls_data(hsngls)

>>> coincs.columns
    ('fap', 'event_id', 'ranking_stat', 'template', 'sngl_events', 'far', 'H1')

>>> coincs['H1'].columns
('event_id',
 'ranking_stat',
 'chisq',
 'chisq_dof',
 'process_id',
 'snr',
 'end_time',
 'template',
 'ifo',
 'sigma')
``
>>> lsngls.ifo
chararray(['L1', 'L1', 'L1', ..., 'L1', 'L1', 'L1'], 
      dtype='|S2')

>>> lsngls.ifo
chararray(['L1', 'L1', 'L1', ..., 'L1', 'L1', 'L1'], 
      dtype='|S2')

>>> coincs = coincs.add_sngls_data(lsngls)

>>> coincs.columns
('fap',
 'event_id',
 'ranking_stat',
 'template',
 'sngl_events',
 'far',
 'H1',
 'L1')

>>> coincs.ranking_stat, coincs['H1']['ranking_stat'], coincs.L1.ranking_stat
(array([ 76.14434814, 8.63374901, ..., 139.97875977, 18.55389023]),
 array([ 44.2959137,  6.21183914, ..., 124.83552448, 14.78488204]),
 array([ 61.93411348, 5.9962226 , ...,  63.32571184, 11.20955446]))
``

    * Expand the template field (see TmpltInspiral help for how to create
      ``templates`` from an hdf bank file):
``
>>> coincs.template.columns
    ('template_id',)

>>> templates.columns
    ('mass1', 'mass2', 'spin1z', 'spin2z', 'template_hash', 'template_id')

>>> coincs = coincs.expand_templates(templates)

>>> coincs.template.columns
    ('mass1', 'mass2', 'spin1z', 'spin2z', 'template_hash', 'template_id')
``
    """
    default_name = 'coinc_event'
    # add detectors as a persistent attribute
    __persistent_attributes__ = ['detectors'] + \
        super(CoincEvent, CoincEvent).__persistent_attributes__

    @classmethod
    def default_fields(cls, ranking_stat_alias='new_snr',
            detectors=['detector1', 'detector2']):
        """
        Both the ranking stat alias and the maximum number of single-detector
        fields can be set.
        """
        fields = {
            ('false_alarm_rate', 'far'): float,
            ('false_alarm_probability', 'fap'): float,
            ('false_alarm_rate_exc', 'far_exc'): float,
            ('false_alarm_probability_exc', 'fap_exc'): float,
            (ranking_stat_alias, 'ranking_stat'): float,
            'snr': float,
            }
        sngls = {
            det: {
                'event_id': int,
                'end_time': float
                }.items()
            for det in detectors
            }
        # we'll inherit SnglEvent's _static_fields
        return dict(cls._static_fields.items() + fields.items() + \
            sngls.items())

    def __new__(cls, shape, name=None, detectors=['detector1', 'detector2'],
            ranking_stat_alias='new_snr', **kwargs):
        """
        Adds nsngls and ranking_stat_alias to initialization.
        """
        field_args = {'ranking_stat_alias': ranking_stat_alias,
            'detectors': detectors}
        # add the detectors to the requested names if not already
        if 'names' in kwargs:
            names = kwargs.pop('names')
            if isinstance(names, str) or isinstance(names, unicode):
                names = [names]
            else:
                names = list(names)
            names += [det for det in detectors if det not in names]
            kwargs['names'] = names
        # Note: we need to call _LSCArrayWithDefaults directly, as using
        # super will lead to SnglEvent's __new__, which sets its own field args
        obj = _LSCArrayWithDefaults.__new__(cls, shape, name=name,
            field_args=field_args, **kwargs)
        # set the detectors attribute
        obj.addattr('detectors', tuple(sorted(detectors)))
        return obj

    @property
    def detected_in(self):
        """
        Returns the names of the detectors that contributed to each coinc
        event. This is found by returning all of the detectors for which
        self[detector].event_id != lscarrays.ID_NOT_SET.
        """
        detectors = numpy.array(self.detectors)
        mask = numpy.vstack([
            self[det]['event_id'] != ID_NOT_SET \
            for det in detectors]).T
        return numpy.array([','.join(detectors[numpy.where(mask[ii,:])]) \
            for ii in range(self.size)])

    def expand_sngls(self, sngl_event_array, get_fields=None,
            sngls_map_field='event_id'):
        """
        Given an array of singles, adds sub-arrays of the single-detector
        trigger data named by ifo to self. For each ifo in that is in
        ``self['sngl_events']['ifos']``, the data is retrieved such that:
        ``self['sngl_events'][[(ifo, 'event_id')]] == sngl_event_array[['ifo', sngls_map_field]]``.
        If the ``sngl_event_array`` only has data of a subset of the ifos
        in ``self.sngl_events.ifos``, fields will only be added for those
        ifos. For example, if ``self.sngl_events.ifos == ['H1', 'L1']`` but
        ``sngl_event_array`` only has H1 data, then only
        ``self['H1']`` will be added.

        Parameters
        ----------
        sngl_event_array: (any subclass of) LSCArray
            The array of singles with additional fields to add.
        get_fields: {None | (list of) strings}
            The names of the fields to get from the sngl_event_array.
            If ``None``, all fields will be retrieved.
        sngls_map_field: {'event_id' | string}
            The name of the field in the ``sngl_event_array`` to match.
            Default is ``'event_id'``.

        Returns
        -------
        newarray: new instance of this array
            A copy of this array with the sngl_events sub-array containing all
            of the fields specified by ``get_fields``.

        Notes
        -----
        * The ``sngl_event_array`` must have a ``detector`` field.
        * The ``sngl_event_array`` must have one and only one row in it for a
          given ``detector, event_id`` pair in self. If ``sngl_event_array`` has an
          ifo that is in self, but does not have an ``event_id`` that self has
          for that ifo, a KeyError is raised. For example, you will get a
          KeyError if self has ``ifo, event_id`` pair ``'H1', 11`` and
          ``sngl_event_array`` has ifo ``'H1'``, but not ``'H1', 11``. If
          ``sngl_event_array`` has more than one entry for a given ``ifo,
          event_id`` pair, a ValueError is raised.
        * If some rows in self do not have a particular ifo, a zeroed entry
          will be added to that row for that ifo's field. For example, if self
          has ``sngl_events.ifos = [('H1', 'L1'), ('L1', V1')]``, the first
          row of ``newarray['V1']`` will be zeroed and the last row of 
          ``newarray['H1']`` will be zeroed.
        """
        #
        # Note: since this join has to do deal with possibly missing ifos, it
        # is a bit more complicated then the join that's carried out in
        # join_arrays; thus we do the join manually here, instead of calling
        # join_arrays.
        #
        # cycle over the ifos in self.sngl_ifos.ifos that are in the
        # sngl_event_array
        others_dets = numpy.unique(sngl_event_array['detector'])
        if not any(others_dets):
            raise ValueError("sngl_event_array's detector column is not " +
                "populated!")
        # if getting all fields, exclude the detector field, since that will
        # be the sub-array's name
        if get_fields is None:
            get_fields = [name for name in sngl_event_array.columns \
                if name != 'detector']
        new_self = self
        for det in self.detectors:
            if det not in others_dets:
                # just skip
                continue
            if others_dets.size == 1:
                # we can just look at the whole array
                other_array = sngl_event_array
            else:
                # pull out the rows with detector == det
                other_array = sngl_event_array[numpy.where(
                    sngl_event_array['detector'] == det)]
            # figure out what rows in self have an entry for this detector
            mask = self[det]['event_id'] != ID_NOT_SET
            if mask.all():
                # every row has an entry, just map all
                map_indices = None
            else:
                map_indices = numpy.where(mask)
            # join
            new_self = new_self.join(other_array,
                '%s.event_id' % det, det,
                sngls_map_field, get_fields=get_fields,
                map_indices=map_indices)
        return new_self


# we'll vectorize TimeDelayFromEarthCenter for faster processing of end times
def _time_delay_from_center(geocent_end_time, detector, ra, dec):
    return geocent_end_time + lal.TimeDelayFromEarthCenter(detector.location,
        ra, dec, geocent_end_time)
time_delay_from_center = numpy.vectorize(_time_delay_from_center)

class SimInspiral(Waveform):
    """
    Subclasses Waveform, adding attributes ``location_params``,
    ``recovered_params``, and ``distribution_params``, which define fields
    fields for a source's location, distance, and end time (both geocentric
    and in each detector) , whether the injection recovered, any recovered
    statistics, and the distribution used to create the injections.  Also has
    columns ``process_id`` and ``simulation_id``.  The ``recovered`` field is
    a subarray with field ``event_id``.  If given a CoincEvent or SnglEvent
    array, this can be expanded to have all of the recovered parameters and
    statistics; see expand_recovered for details.

    Notes
    -----
    * When a new array is initialized with a ``recovered`` field, all of the
      injections are marked as not recovered (i.e.,
      ``arr['recovered']['isfound'] = False``).

    Examples
    --------
    * Create a SimInspiral array from an hdfinjfind file:
``
>>> hdfinjfind = h5py.File('H1L1-HDFINJFIND_BNS1INJ_INJ_INJ-1117400416-928800.hdf', 'r')

>>> hdfinj = hdfinjfind['injections']
>>> sims = SimInspiral(len(hdfinj['end_time']), names=['simulation_id', 'geocent_end_time', 'distance', 'mass1', 'mass2', 'ra', 'dec', 'site_params', 'isrecovered', 'recovered'])

>>> sims['simulation_id'] = numpy.arange(sims.size)

>>> sims['geocent_end_time'] = hdfinj['end_time']

>>> sims['distance'] = hdfinj['distance']

>>> sims['mass1'] = hdfinj['mass1']

>>> sims['mass2'] = hdfinj['mass2']

>>> sims['ra'] = hdfinj['longitude']

>>> sims['dec'] = hdfinj['latitude']

>>> sims['isrecovered'][hdfinjfind['found']['injection_index']] = True

>>> sims['recovered']['event_id'][numpy.where(sims['isrecovered'])] = numpy.arange(len(hdfinjfind['found']['fap']))

>>> sims['site_params']['ifo'] = hdfinjfind.attrs['detector_2'], hdfinjfind.attrs['detector_1']

>>> sims['site_params']['ifo']
SimInspiral([['H1', 'L1'],
       ['H1', 'L1'],
       ['H1', 'L1'],
       ..., 
       ['H1', 'L1'],
       ['H1', 'L1'],
       ['H1', 'L1']], 
      dtype='|S2')

>>> sims['site_params']['sigma'][:,0] = hdfinj['eff_dist_h']*sims.distance

>>> sims['site_params']['sigma'][:,1] = hdfinj['eff_dist_l']*sims.distance
``

    * Add recovered information (see TmpltInspiral help for how to create
      ``templates`` from an hdf bank file):
``
>>> hdffound = hdfinjfind['found']

>>> coincs = CoincEvent(len(hdffound['fap']), names=['event_id', 'fap', 'far', 'ranking_stat', 'template', 'sngl_events'])

>>> coincs['fap'] = hdffound['fap']

>>> coincs['far'] = 1./hdffound['ifar'].value

>>> coincs['ranking_stat'] = hdffound['stat']

>>> coincs['template']['template_id'] = hdffound['template_id']

>>> coincs['event_id'] = numpy.arange(coincs.size)

>>> coincs['sngl_events']['ifos'] = [hdfinjfind.attrs['detector_2'], hdfinjfind.attrs['detector_1']]

>>> coincs['sngl_events']['event_ids'] = numpy.vstack((hdffound['trigger_id2'], hdffound['trigger_id1'])).T

>>> coincs = coincs.expand_templates(templates)

>>> sims.recovered.columns
    ('event_id',)

>>> sims['recovered'].columns
    ('event_id',)

>>> sims = sims.expand_recovered(coincs)

>>> sims['recovered'].columns
    ('event_id', 'ranking_stat', 'far', 'fap', 'L1', 'H1', 'template')

>>> sims['recovered']['template'].columns
    ('mass1', 'mass2', 'spin1z', 'spin2z', 'template_hash', 'template_id')

>>> recsims = sims[numpy.where(sims.isrecovered)]

>>> recsims.geocent_end_time, recsims.recovered.H1.end_time, recsims.recovered.L1.end_time, recsims.mchirp, recsims.recovered.template.mchirp
(array([ 1.11743314e+09, 1.11743322e+09, ..., 1.11808778e+09, 1.11808794e+09]),
 array([ 1.11743314e+09, 1.11743322e+09, ..., 1.11808778e+09, 1.11808794e+09]),
 array([ 1.11743314e+09, 1.11743322e+09, ..., 1.11808778e+09, 1.11808794e+09]),
 array([ 1.81795762,     1.74482645,     ..., 1.1793251,      1.41826222]),
 array([ 1.81705785,     1.74960184,     ..., 1.17973983,     1.41618061], dtype=float32))
``
    """
    default_name = 'sim_inspiral'
    # add detectors as a persistent attribute
    __persistent_attributes__ = ['detectors'] + \
        super(SimInspiral, SimInspiral).__persistent_attributes__

    @classmethod
    def default_fields(cls, detectors=['detector1', 'detector2'],
            nrecovered=1):
        """
        The number of ifos stored in site params and the maximum number of
        events that can be stored in the recovered fields can be set.
        """
        fields = {
            # ids
            'simulation_id': int,
            'process_id': int,
            # location params
            'geocent_end_time_s': 'int_4s',
            'geocent_end_time_ns': 'int_4s',
            'distance': float,
            ('right_ascension', 'ra'): float,
            ('declination', 'dec'): float,
            'polarization': float,
            # recovered params
            'isrecovered': bool,
            'recovered': ({
                'event_id': int,
                }.items(), nrecovered),
            # distribution params
            'distribution': {
                'min_vol': float,
                'volume_weight': float,
                # XXX: How to store information about mass and spin
                # distributions?
                }.items()
            }
            # site params
        site_params = {
            det: {
                'eff_dist': float,
                'sigma': float,
                }.items()
            for det in detectors
            }
        # we'll inherit static fields from Waveform
        return dict(cls._static_fields.items() + fields.items() + \
            site_params.items())

    def __new__(cls, shape, name=None, detectors=['detector1', 'detector2'],
            nrecovered=None, **kwargs):
        """
        Adds detectors and nrecovered to initialization.
        """
        field_args = {'nrecovered': nrecovered,
            'detectors': detectors}
        # add the detectors to the requested names if not already
        if 'names' in kwargs:
            names = kwargs.pop('names')
            if isinstance(names, str) or isinstance(names, unicode):
                names = [names]
            else:
                names = list(names)
            names += [det for det in detectors if det not in names]
            kwargs['names'] = names
        obj = super(SimInspiral, cls).__new__(cls, shape, name=name,
            field_args=field_args, **kwargs)
        # set the detectors attribute
        obj.addattr('detectors', tuple(sorted(detectors)))
        return obj

    @property
    def geocent_end_time(self):
        return self.geocent_end_time_s + 1e-9*self.geocent_end_time_ns

    def end_time(self, detector=None):
        """
        Returns the end time in the given detector. If detector is None,
        returns the geocentric end time.
        """
        geocent_end_time = self.geocent_end_time
        if detector is None:
            return geocent_end_time
        else:
            detector = lalsim.DetectorPrefixToLALDetector(detector)
            return time_delay_from_center(geocent_end_time, detector, self.ra,
                self.dec)

    def expand_recovered(self, event_array, get_fields=None,
            selfs_map_field='recovered.event_id',
            events_map_field='event_id'):
        """
        Given an array of (coinc) events, replaces the recovered column with a
        sub-array of the recovered data. This is done by getting
        all rows in the ``event_array`` such that:
        ``self[numpy.where(self['isrecovered'])][selfs_map_field] == event_array[events_map_field]``.

        Parameters
        ----------
        event_array: (any subclass of) LSCArray
            The array of events with additional fields to add.
        get_fields: {None | (list of) strings}
            The names of the fields to get from the ``event_array``.
            If ``None``, all fields will be retrieved.
        selfs_map_field: {'recovered.event_id' | string}
            The name of the field in self to use to match to events in
            ``event_array``. Default is ``recovered.event_id``.
        events_map_field: {'event_id' | string}
            The name of the field in the ``event_array`` to match.
            Default is ``event_id``.

        Returns
        -------
        new array: new instances of this array
            A copy of this array with the ``recovered`` sub-array containing
            all of the fields specified by ``get_fields``.
        
        Notes
        -----
        * Only elements for which ``self['isrecovered'] == True`` will be
          expanded. All other elements will have a zeroed row in ``recovered``
          field of the output array.

        * If the injection is mapped to multiple events, all of the events will
          be expanded.

        * To get the recovered parameters, expand the event array's template
          field first, then expand the recovered here. See
          (Coinc|Sngl)Event.expand_templates for details.

        * The coincs themselves contain subarrays of the single-detector
          triggers. To get access to the single-detector recovered information,
          add the single-detector information to the coinc event array first
          (``see CoincEvent.add_sngls_data``), then expand the coincs here.

        * If the injections are mapped to single events rather than coinc
          events, you can map to the single events by passing a SnglEvent array
          as the ``event_array``.
        """
        return self.join(event_array, selfs_map_field, 'recovered',
            other_map_field=events_map_field, get_fields=get_fields,
            map_indices=numpy.where(self['isrecovered']))

    @property
    def optimal_snr(self):
        """
        Gives the maximum SNR that the injections can have.
        """
        return numpy.sqrt(numpy.array([self[det]['sigma']**2
            for det in self.detectors]).sum())/self['distance']

    @property
    def detected_in(self):
        """
        Returns the names of the detectors that the injections was detected in.
        This is done by returning all of the detectors in self's detectors for
        which ``self['recovered'][detector].event_id != lscarrays.ID_NOT_SET``.

        Note: ``expand_recovered`` must have been run first on the list of
        events.
        """
        detectors = numpy.array([det for det in self.detectors \
            if det in self['recovered'].columns])
        if detectors.size == 0:
            raise ValueError("No detectors found in this array's recovered " +
                "field. Did you run expand_recovered?")
        mask = numpy.vstack([
            self['recovered'][det]['event_id'] != ID_NOT_SET \
            for det in detectors]).T
        return numpy.array([','.join(detectors[numpy.where(mask[ii,:])]) \
            for ii in range(self.size)])
