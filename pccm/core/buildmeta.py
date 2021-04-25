import copy
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

from ccimport import compat
from ccimport.buildtools.writer import group_dict_by_split


def _unique_list_keep_order(seq: list):
    if compat.Python3_7AndLater:
        # https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
        # only python 3.7 language std ensure the preserve-order dict
        return list(dict.fromkeys(seq))
    else:
        # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]


def _merge_compiler_to_flags(this: Dict[str, List[str]],
                             other: Dict[str, List[str]]):
    all_cflag_keys = [*this.keys(), *other.keys()]
    res_cflags = OrderedDict()  # type: Dict[str, List[str]]
    for k in all_cflag_keys:
        k_in_this = k in this
        k_in_other = k in other
        if k_in_this and not k_in_other:
            res_cflags[k] = this[k]
        elif not k_in_this and k_in_other:
            res_cflags[k] = other[k]
        else:
            res_cflags[k] = _unique_list_keep_order(this[k] + other[k])
    return res_cflags


class BuildMeta(object):
    def __init__(self,
                 includes: Optional[List[Union[str, Path]]] = None,
                 libpaths: Optional[List[Union[str, Path]]] = None,
                 libraries: Optional[List[str]] = None,
                 compiler_to_cflags: Optional[Dict[str, List[str]]] = None,
                 compiler_to_ldflags: Optional[Dict[str, List[str]]] = None):
        if includes is None:
            includes = []
        if libpaths is None:
            libpaths = []
        if libraries is None:
            libraries = []
        if compiler_to_cflags is None:
            compiler_to_cflags = OrderedDict()
        if compiler_to_ldflags is None:
            compiler_to_ldflags = OrderedDict()

        self.includes = includes
        self.libpaths = libpaths
        self.libraries = libraries
        self.compiler_to_cflags = group_dict_by_split(compiler_to_cflags)
        self.compiler_to_ldflags = group_dict_by_split(compiler_to_ldflags)

    def __add__(self, other: "BuildMeta"):
        merged_cflags = _merge_compiler_to_flags(self.compiler_to_cflags,
                                                 other.compiler_to_cflags)
        merged_ldflags = _merge_compiler_to_flags(self.compiler_to_ldflags,
                                                  other.compiler_to_ldflags)

        res = BuildMeta(
            self.includes + other.includes, self.libpaths + other.libpaths,
            _unique_list_keep_order(self.libraries + other.libraries),
            merged_cflags, merged_ldflags)
        return res

    def __radd__(self, other: "BuildMeta"):
        return other.__add__(self)

    def __iadd__(self, other: "BuildMeta"):
        merged_cflags = _merge_compiler_to_flags(self.compiler_to_cflags,
                                                 other.compiler_to_cflags)
        merged_ldflags = _merge_compiler_to_flags(self.compiler_to_ldflags,
                                                  other.compiler_to_ldflags)
        self.includes += other.includes
        self.libpaths += other.libpaths
        self.libraries += other.libraries
        self.compiler_to_cflags.update(merged_cflags)
        self.compiler_to_ldflags.update(merged_ldflags)
        return self
