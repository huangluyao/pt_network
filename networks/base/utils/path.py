import os
import os.path as osp
from pathlib import Path


def is_filepath(x):
    return isinstance(x, str) or isinstance(x, Path)


def fopen(filepath, *args, **kwargs):
    if isinstance(filepath, str):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError('`filepath` should be a string or a Path')


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Parameters
    ----------
    dir_path : {str, obj:`Path`}
        Path of the directory.
    suffix : {str, tuple(str)}, optional
        File suffix that we are interested in, default is None.
    recursive : bool, optional
        If set to True, recursively scan the directory, default is False.

    Returns
    -------
    path_gen : generator
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def find_project_root(path, markers=('.git', )):
    """Finds the root directory (including itself) of specified markers.

    Parameters
    ----------
    path : str
        Path of directory or file.
    markers : list[str], optional
        List of file or directory names, default is ('.git', ).

    Returns
    -------
    root_dir : {str, None}
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None
