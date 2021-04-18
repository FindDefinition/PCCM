from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

from ccimport.buildtools.writer import group_dict_by_split


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
            compiler_to_cflags = {}
        if compiler_to_ldflags is None:
            compiler_to_ldflags = {}

        self.includes = includes 
        self.libpaths = libpaths 
        self.libraries = libraries 
        self.compiler_to_cflags = compiler_to_cflags 
        self.compiler_to_ldflags = compiler_to_ldflags 

    def __add__(self, other: "BuildMeta"):
        pass
