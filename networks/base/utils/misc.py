from collections import abc
import functools
from inspect import getfullargspec
import itertools
import warnings


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Parameters
    ----------
    seq : Sequence
        The sequence to be checked.
    expected_type : type
        Expected type of sequence items.
    seq_type : type
        Expected sequence type.

    Returns
    -------
    out : bool
        Whether the sequence is valid.
    """
    if seq_type is None:
        expect_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        expect_seq_type = seq_type
    if not isinstance(seq, expect_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def concat_list(in_list):
    """Concatenate a list of list into a single list.

    Parameters
    ----------
    in_list : list
        The list of list to be merged.

    Returns
    -------
    list
        The concatenated flat list.
    """
    return list(itertools.chain(*in_list))


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length.

    Parameters
    ----------
    in_list : list
        The list to be sliced.
    lens : int | list
        The expected length of each out list.

    Returns
    -------
    list
        A list of sliced list.
    """
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError('"indices" must be an integer or a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError('sum of lens and list length does not '
                         f'match: {sum(lens)} != {len(in_list)}')
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def update_prefix_of_dict(_dict, old_prefix, new_prefix):
    if not isinstance(_dict, dict):
        return
    tmp = _dict
    for k, v in tmp.items():
        if isinstance(v, str) and v.startswith(old_prefix):
            _dict[k] = v.replace(old_prefix, new_prefix)
        else:
            if isinstance(v, dict):
                update_prefix_of_dict(_dict[k], old_prefix, new_prefix)
            elif isinstance(v, list):
                for _item in v:
                    update_prefix_of_dict(_item, old_prefix, new_prefix)


def update_value_of_dict(_dict, old_value, new_value):
    if not isinstance(_dict, dict):
        return
    tmp = _dict
    for k, v in tmp.items():
        if isinstance(v, str) and v == old_value:
            _dict[k] = new_value
        else:
            if isinstance(v, dict):
                update_value_of_dict(_dict[k], old_value, new_value)
            elif isinstance(v, list):
                for _item in v:
                    update_value_of_dict(_item, old_value, new_value)


def repalce_kwargs_in_dict(_dict):
    if not isinstance(_dict, dict):
        return
    _items = _dict.copy().items()
    for k, v in _items:
        if 'kwargs' == k:
            _kwargs = _dict.pop('kwargs')
            _dict.update(_kwargs)
        else:
            if isinstance(v, dict):
                repalce_kwargs_in_dict(_dict[k])
            elif isinstance(v, list):
                for _item in v:
                    repalce_kwargs_in_dict(_item)


def deprecated_api_warning(name_dict, cls_name=None):
    """A decorator to check if some argments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Parameters
    ----------
    name_dict : dict
        key (str): Deprecate argument names.
        val (str): Expected argument names.

    Returns
    -------
    func
        New function.
    """

    def api_warning_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead')
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead')
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper


class NiceRepr(object):
    """Inherit from this class and define ``__nice__`` to "nicely" print your
    objects.

    Defines ``__str__`` and ``__repr__`` in terms of ``__nice__`` function
    Classes that inherit from :class:`NiceRepr` should redefine ``__nice__``.
    If the inheriting class has a ``__len__``, method then the default
    ``__nice__`` method will return its length.

    Examples
    --------
    >>> class Foo(NiceRepr):
    ...    def __nice__(self):
    ...        return 'info'
    >>> foo = Foo()
    >>> assert str(foo) == '<Foo(info)>'
    >>> assert repr(foo).startswith('<Foo(info) at ')

    Examples
    --------
    >>> class Bar(NiceRepr):
    ...    pass
    >>> bar = Bar()
    >>> import pytest
    >>> with pytest.warns(None) as record:
    >>>     assert 'object at' in str(bar)
    >>>     assert 'object at' in repr(bar)

    Examples
    --------
    >>> class Baz(NiceRepr):
    ...    def __len__(self):
    ...        return 5
    >>> baz = Baz()
    >>> assert str(baz) == '<Baz(5)>'
    """

    def __nice__(self):
        """str: a "nice" summary string describing this module"""
        if hasattr(self, '__len__'):
            return str(len(self))
        else:
            raise NotImplementedError(
                f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        """str: the string of the module"""
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        """str: the string of the module"""
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)
