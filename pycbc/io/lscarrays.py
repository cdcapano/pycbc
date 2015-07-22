#! /usr/bin/env python

import os, sys
import math
import numpy
import operator
import numpy.lib.recfunctions as recfuncs
import numpy.core.records as numpyrecs

import lal
from glue.ligolw import types as ligolw_types

# data type mappings
ilwd_as_int = True
typeDict = {
    'lstring': str,
    'ilwd:char': int if ilwd_as_int else str
}
typeDictStrAsObj = {
    # we'll map strings to python objects to allow variable size strings
    str: object,
    'lstring': object,
    'ilwd:char': int if ilwd_as_int else object
}
# for xml tables, we'll use the mappings defined in glue
typeDict.update(ligolw_types.ToNumPyType)
typeDictStrAsObj.update(ligolw_types.ToNumPyType)

def typeMap(gettype, str_as_obj=False):
    """
    Checks typeDict for the given type. If it isn't found, just returns the
    input type.
    """
    d = typeDict if not str_as_obj else typeDictStrAsObj
    try:
        return d[gettype]
    except KeyError:
        return gettype
    except TypeError:
        return gettype


def parse_columns_into_dtype(input_array, columns=None, str_as_obj=False):
    """
    Given an input array and columns, parses into a numpy dtype.
    """
    # get an initial array
    obj = numpy.asarray(input_array)
    # check the dimensions
    if obj.ndim > 2:
        raise ValueError("input array must be 1 or 2 dimensional")
    # set the datatypes
    outdt = []
    if columns is not None:
        if obj.ndim == 2:
            # get the number of columns from the length of the second
            # dimension
            ncols = obj.shape[1]
            empty_input = ncols == 0
            if empty_input:
                # was passed an empty array, just expand
                ncols = len(columns)
        else:
            empty_input = obj.shape[0] == 0
            if empty_input:
                # empty array
                ncols = len(columns)
            else:
                ncols = 1
        # check that the number of columns match
        if len(columns) != ncols:
            raise ValueError("given number of columns does not match " +
                "the size of the input array")
        # get the names and dtype of each column (if specified)
        for ii in range(len(columns)):
            if isinstance(columns[ii], tuple):
                try:
                    colname, coltype = columns[ii]
                except ValueError:
                    colname = columns[ii][0]
                    coltype = None
            else:
                colname = columns[ii]
                coltype = None
            # if the column type is None, try to auto-detect it from the
            # value in the first row in the ith column of the input. We
            # use the values from the original input array, since numpy
            # may have already cast obj's dtype to something else.
            # Note: we'll only do this if we're not working with an empty
            # input array
            if coltype is None and not empty_input:
                if len(input_array.dtype) == 0:
                    coltype = input_array.dtype.type
                else:
                    try:
                        coltype = input_array.dtype[ii].type
                    except AttributeError:
                        if obj.ndim == 1:
                            coltype = type(input_array[ii])
                        else:
                            coltype = type(input_array[0][ii])
            # if the coltype is a string, we will have to look through
            # the input array to find the longest string in order to set
            # the appropriate bit length
            if not str_as_obj:
                dt = numpy.dtype(typeMap(coltype, str_as_obj=False))
                if dt == 'S' or dt == 'U':
                    # get the longest string
                    slen = len(max(obj[:,ii], key=len))+2
                    coltype = '%s%i' % (dt.char, slen)
            outdt.append((colname, typeMap(coltype, str_as_obj)))
    else:
        # just use whatever the current dtype is
        outdt = [(colname, typeMap(coltype, str_as_obj)) \
            for colname,coltype in obj.dtype.descr]
    return numpy.dtype(outdt)


def append_fields(input_array, names, data, dtypes=None):
    """
    Wrapper around numpy.lib.recfunctions.append_fields. Adds one or more
    columns to an existing array. Differences from recfunctions.append_fields
    are: usemask is forced to be False; an error is raised if the given data
    is not the same length as the base; the type of the returned array is the
    same as the input array. For more details, see the documentation for
    recfunctions.

    Parameters
    ----------
    input_array: array
        The array to add a field to.
    fields: list of (strings | tuples)
        List of columns to add; may be either a list of names
        of columns and/or a list of tuples in which the first element is the
        name of the column and the second element the data type to cast it
        to; e.g. ``['col1', ('col2', float), 'col3', ...]``. If not data type
        is used, the data type of the input data will be used. If only adding
        a single field, can just specify the name, e.g. ``'col1'``.
    data: array or sequence of arrays
        Array or sequence of arrays to fill the new fields with.
        Number of arrays must match the number of fields, and the length of
        each array must be the same as the input_array.

    Returns
    -------
    new_array: array
        A copy of the input array with the new fields added.
    """
    try:
        updated_arr = recfuncs.append_fields(input_array, names, data,
            dtypes=dtypes, asrecarray=False,
            usemask=False).view(type=type(input_array))
    # versions of numpy < 1.9.2 treat arrays with object fields too
    # conservatively. If any field in the array is an object, it will not
    # allow a view to be taken, even if the view does not cast the object
    # field into something else. This causes append_fields to fail with a
    # TypeError if input_array has any object fields. If this happens, we'll
    # add the field by hand
    except TypeError:
        if isinstance(names, str) or isinstance(names, unicode):
            # only single field provided, make into list
            names = [names]
            data = [data]
            if dtypes is not None:
                dtypes = [dtypes]
        # construct the new data type
        new_dtypes = []
        for ii,name in enumerate(names):
            if dtypes is not None:
                dt = dtypes[ii]
            else:
                this_data = data[ii]
                if isinstance(this_data, numpy.ndarray):
                    dt = this_data.dtype
                    # if length of dt > 1, is a sub array, just name
                    if len(dt) > 1:
                        dt = (name, dt)
                    else:
                        dt = (name, dt.descr[1])
                elif len(this_data) != 0:
                    dt = (name, type(this_data[0]))
                else:
                    dt = (name, None)
                new_dtypes.append(dt)
        new_dtype = input_array.dtype.descr + new_dtypes
        # create an empty array to store it
        updated_arr = numpy.empty(input_array.shape, dtype=new_dtype)
        # copy the data over
        for col in input_array.dtype.names:
            updated_arr[col] = input_array[col]
        for ii,col in enumerate(names):
            updated_arr[col] = data[ii]
        updated_arr = type(input_array)(updated_arr)
    # check the shape
    if updated_arr.shape != input_array.shape:
        raise ValueError("one or more of the provided arrays is longer " +
            "than the base array")
    return updated_arr


def get_fields(input_array, names, copy=False, outtype=None):
    """
    Given an array with named fields, creates a new LSCArray with the given
    columns. Only fields in input_array that are in the list of names will be
    extracted. All the fields listed in names must be present in the
    input_array.

    Parameters
    ----------
    input_array: array-like
        Anything that can be cast as a numpy array. The resulting array
        must have a dtype with at least one field in columns.
    names: {strings|list of strings}
        List of column names for the returned array; may be either a single
        name or a list of names.
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
    new_dtype = numpy.dtype({
        'names': names,
        'formats': [input_array.dtype.fields[name][0].str for name in names],
        'offsets': [input_array.dtype.fields[name][1] for name in names]})
    if copy:
        try:
            return input_array[names].view(type=outtype)
        # versions of numpy < 1.9.2 treat arrays with object fields too
        # conservatively. If any field in the array is an object, it will not
        # allow a view to be taken, even if the view does not cast the object
        # field into something else. This causes append_fields to fail with a
        # TypeError if input_array has any object fields. If this happens,
        # we'll make the copy by hand
        except TypeError:
            new_arr = numpy.empty(input_array.shape, dtype=new_dtype)
            for col in names:
                new_arr[col] = input_array[col]
            return outtype(new_arr)
    else:
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
        return dict(zip(unique_vals, unique_idx))
    # else, we need to be set each key value pointing to the
    # appropriate indices
    else:
        return dict([[unique_vals[ii], numpy.where(map_idx == ii)] \
            for ii in range(len(unique_vals))
            ])


class LSCArray(numpy.recarray):
    """
    Subclass of numpy.recarray that adds additional functionality.

    To initialize, pass any iterable that can be converted into a numpy
    array. In addition, you can pass a list of columns that specify the names
    and (optionally) the data types of each column in the desired output.

    Parameters
    ----------
    input_array: array_like
        Any object that can be converted to a numpy array. See numpy.array
        for details. Must be 1 or 2 dimensional.
    columns: list, optional
        Either a list of column names for the output array, or a list of
        tuples giving the name of each column and its data type. If provided,
        the number of elements in columns must be the same as the number
        of columns in the input array if the array is not empty. If
        data types are not provided, the datatype will be detected from the
        first row of the corresponding column. If an empty list is provided,
        any data types not specified will use the numpy default. See notes
        about string datatypes, below, for information on how strings are
        handled.
    name: {None|str}
        Name of the array. This is copied to any views or copies made of the
        array.

    Attributes
    ----------
    name: str
        Instance attribute. The name of the array.
    str_as_obj: {bool|False}
        Class attribute. Whether to cast python str and unicode types to
        objects or to fixed-length strings. See Notes, below for details.

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
    to version 1.9.2, you cannot create a view of an array that has any
    fields that are object data types, even if the view does not touch the
    object fields. (This has since been relaxed.) This means that if you are
    using an older version of numpy, any slices on an array will be a copy
    instead of a view, which increases memory overhead. 

    The second option, using strings of a fixed length, solves the issues
    with object fields. However, in order to determine the length of the
    strings to use, all of the rows in the column containing a string must
    be checked, which makes array initialization slower. Also, if you try
    to change one of the strings after the array is created, the string will
    be truncated at whatever string length is used.

    This class offers the option to set strings as objects, or as fixed-length
    strings. To toggle what it does, set LSCArray.str_as_obj. If True, any
    column that is set to a python str or unicode is set to object in the
    resulting array. (Numpy strings, e.g., 'S10', will remain as strings.)
    If set to False, the any column set to str or unicode will be searched
    for the longest string in the column upon creation; the resulting data
    type will then be 'S|Ux' where x is the length of the longest string.

    Examples
    --------
    Create an zeroed array with two columns, a, b:
    ``>>> x = LSCArray(numpy.zeros((4,2)), columns=['a', 'b'])``

    Set/retrieve a column using index or attribute syntax:
``
>>> x['a'] = [1.,2.,3.,4.]
    
>>> x.b = [5.,6.,7.,8.]

>>> x
LSCArray([(1.0, 5.0), (2.0, 6.0), (3.0, 7.0), (4.0, 8.0)], 
      dtype=[('a', '<f8'), ('b', '<f8')])

>>> x.a
    array([ 1.,  2.,  3.,  4.])

>>> x['b']
    array([ 5.,  6.,  7.,  8.])
``

    Retrieve a function of the columns as if it were a column:
``
>>> x['sin(a/b)']
array([ 0.19866933,  0.3271947 ,  0.41557185,  0.47942554])
``

    Convert a LIGOLw xml table:
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

    Only view a few of the columns:
``
>>> sim_array.with_fields(['simulation_id', 'mass1', 'mass2'])
LSCArray([(0, 5.943458080291748, 3.614427089691162),
       (1, 3.342266082763672, 3.338679075241089),
       (2, 3.7302539348602295, 6.2023820877075195), ...,
       (9236, 4.75996208190918, 4.678255081176758),
       (9237, 4.237561225891113, 2.1022119522094727),
       (9238, 4.627500057220459, 3.251383066177368)], 
      dtype=[('simulation_id', '<i8'), ('mass1', '<f4'), ('mass2', '<f4')])
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

    Add a field to the array which is itself an LSCArray:
``
>>> some_masses = LSCArray(numpy.random.uniform(1., 50., size=2*len(sim_array)).reshape((len(sim_array),2)), columns=['mass1', 'mass2'])

>>> sim_array = sim_array.add_fields('recovered', some_masses)

>>> sim_array.recovered
LSCArray([(27.54155066591844, 38.07147794479682),
       (21.722038073027466, 35.599184888132626),
       (31.480358805697204, 31.022242070659644), ...,
       (4.733996033280733, 41.00630522340374),
       (6.505521269745874, 27.111935847449796),
       (47.232391322216486, 37.00435499010594)], 
      dtype=[('mass1', '<f8'), ('mass2', '<f8')])

>>> sim_array.recovered.mass1
array([ 27.54155067,  21.72203807,  31.48035881, ...,   4.73399603,
         6.50552127,  47.23239132])
``
    """
    str_as_obj = False
    __persistent_attributes__ = ['name']

    def __new__(cls, input_array, columns=None, name=None):

        # if no name specified, but input has a name, use that
        if name is None and hasattr(input_array, 'name'):
            name = input_array.name

        # get the dtype
        outdt = parse_columns_into_dtype(input_array, columns=columns,
            str_as_obj=cls.str_as_obj)

        # recast the input data as an array with the proper dtype and class
        if isinstance(input_array, numpy.ndarray):
            try:
                obj = numpyrecs.fromarrays(numpy.asarray(
                    input_array.transpose()), dtype=outdt)
            # if input_array is already a structured array, we'll get a
            # ValueError from fromarrays, as it expects a list of flat arrays
            # in that case, we'll just try to create a view of the input array
            except ValueError:
                obj = input_array.view(dtype=outdt)
        else:
            obj = numpy.asarray(input_array, dtype=outdt)
        obj = obj.view(type=cls)
        obj.name = name

        return obj


    def __array_finalize__(self, obj):
        """
        Default values are set here.
        """
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.__lookuptable__ = {}


    def __copy_attributes__(self, other):
        """
        Copies the values of all of the attributes listed in
        self.__persistent_attributes__ to other.
        """
        [setattr(other, attr, getattr(self, attr)) \
            for attr in self.__persistent_attributes__]


    def __getitem__(self, item):
        """
        Wraps self's __getitem__ so that math functions on columns can be
        retrieved. Any function in numpy's library may be used.
        """
        # first try to return
        try:
            return super(LSCArray, self).__getitem__(item)
        except ValueError:
            return super(LSCArray, self).__getitem__(item)
            # arg isn't a simple argument of row, so we'll have to eval it
            item_dict = dict([ [col, super(LSCArray, self).__getitem__(col)]\
                for col in self.dtype.names])
            safe_dict = {}
            safe_dict.update(item_dict)
            safe_dict.update(numpy.__dict__)
            return eval(item, {"__builtins__": None}, safe_dict)

    @classmethod
    def from_ligolw_table(cls, table, columns=None):
        """
        Converts the given ligolw table into an LSCArray. The tableName
        attribute is copied to the array's name.

        Parameters
        ----------
        table: LIGOLw table instance
            The table to convert.
        columns: {None|list}
            Optionally specify a list of columns to retrieve. If None
            provided, all the columns in the table will be converted.

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
        if ilwd_as_int:
            input_array = [tuple(
                        getattr(row, col) if dt != 'ilwd:char' \
                        else int(getattr(row, col)) \
                    for col,dt in columns) \
                for row in table]
        else:
            input_array = [tuple(getattr(row, col) for col,_ in columns) \
                for row in table]
        # return the values as an instance of cls
        return cls(input_array, columns=columns, name=name)


    def add_fields(self, names, data, dtypes=None):
        """
        Returns a new instance with the given fields added. Note: this array
        does not change; the returned array is a new copy. The 
        internal lookup table of the new array will also be empty.

        Parameters
        ----------
        names: str | list of strings
            The name(s) of the fields to add. If only adding a single field,
            can just be the name of the field; otherwise, must be a list
            of strings.
        data: array or sequence of arrays
            Array or sequence of arrays to fill the new fields with.  Number
            of arrays must match the number of fields, and the length of each
            array must be the same as the input_array.
        dtypes: {None | (list of) argument(s) readable by numpy.dtype}
            If you would like to cast the data to a different data type, the
            data type to cast it to. Default is None, in which case the data
            type of the data is used. Number of dtypes provided must be the
            same as the number of arrays in data. Each dtype can be either
            a numpy.dtype, or an argument that is readable by numpy.dtype;
            see numpy.dtype for details.

        Returns
        -------
        new_array: new instance of this array
            A copy of this array with the desired fields added.
        """
        newself = append_fields(self, names, data, dtypes=dtypes)
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


    def lookup(self, column, value):
        """

        Returns the elements in self for which the given column matches the
        given value. If this is the first time that the given column has been
        requested, an internal dictionary is built first, then the elements
        returned. If the value cannot be found, a KeyError is raised.
        
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
        column: str
            The name of the column to look up.
        value: self.dtype(column)
            The value in self.column to get.

        Returns
        -------
        matching: type(self)
            The rows in self that match the request value.
        """
        try:
            return self[self.__lookuptable__[column][value]]
        except KeyError:
            # build the look up for this column
            if column not in self.__lookuptable__:
                colvals = self[column]
                self.__lookuptable__[column] = build_lookup_table(colvals)
            # try again
            return self[self.__lookuptable__[column][value]]


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
