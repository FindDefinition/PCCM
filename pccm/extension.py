import abc
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

from ccimport import compat
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from ccimport.buildtools.writer import DEFAULT_MSVC_DEP_PREFIX

import pccm
from pccm.core import Class
from pccm.core import Class, CodeFormatter

class ExtCallback(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ext: Extension, extdir: Path, target_path: Path):
        pass


class PCCMExtension(Extension):
    def __init__(self,
                 cus: List[Class],
                 out_path: Union[str, Path],
                 namespace_root: Optional[Union[str, Path]] = None,
                 pybind_file_suffix: str = ".cc",
                 msvc_deps_prefix=DEFAULT_MSVC_DEP_PREFIX,
                 out_root: Optional[Union[str, Path]] = None,
                 disable_pch: bool = False,
                 disable_anno: bool = False,
                 std: Optional[str] = "c++14",
                 objects_folder: Optional[Union[str, Path]] = None,
                 extcallback: Optional[ExtCallback] = None,
                 debug_file_gen: bool = False,
                 verbose: bool = False):
        # don't invoke the original build_ext for this special extension
        out_path_p = Path(out_path)
        super().__init__(out_path_p.stem, sources=[])
        self.cus = cus
        self.out_path = out_path_p
        self.namespace_root = namespace_root
        self.extcallback = extcallback
        self.objects_folder = objects_folder
        self.pybind_file_suffix = pybind_file_suffix
        self.msvc_deps_prefix = msvc_deps_prefix
        self.out_root = out_root
        self.disable_pch = disable_pch
        self.disable_anno = disable_anno
        self._pccm_std = std
        self.debug_file_gen = debug_file_gen
        self._pccm_verbose = verbose


class PCCMBuild(build_ext):
    def run(self):
        # override build_ext.run to avoid copy
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if not isinstance(ext, PCCMExtension):
            return super().build_extension(ext)
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        extdir = Path(extdir)
        out_path = extdir / ext.out_path
        if out_path.exists():
            shutil.rmtree(out_path)
        if not os.path.exists(self.build_temp):
            Path(self.build_temp).mkdir(exist_ok=True,
                                        parents=True,
                                        mode=0o755)
        build_out_path = Path.cwd() / Path(self.build_temp) / ext.out_path
        build_out_path.parent.mkdir(exist_ok=True, parents=True, mode=0o755)
        out_path.parent.mkdir(exist_ok=True, parents=True, mode=0o755)
        libpaths = []
        lib_path = pccm.builder.build_pybind(ext.cus,
                                             build_out_path,
                                             namespace_root=ext.namespace_root,
                                             pybind_file_suffix=ext.pybind_file_suffix,
                                             msvc_deps_prefix=ext.msvc_deps_prefix,
                                             out_root=ext.out_root,
                                             disable_pch=ext.disable_pch,
                                             disable_anno=ext.disable_anno,
                                             verbose=ext._pccm_verbose,
                                             load_library=False,
                                             std=ext._pccm_std,
                                             objects_folder=ext.objects_folder,
                                             debug_file_gen=ext.debug_file_gen)
        pyi_path = build_out_path
        lib_path = Path(lib_path)
        out_lib_path = out_path.parent / lib_path.name
        shutil.copy(str(lib_path), str(out_lib_path))
        shutil.copytree(str(pyi_path), str(out_path))
        if compat.InWindows:
            lib_path = Path(lib_path)
            win_lib_path = lib_path.parent / (lib_path.stem + ".lib")
            if win_lib_path.exists():
                shutil.copy(str(win_lib_path),
                            str(out_lib_path.parent / win_lib_path.name))
        if ext.extcallback is not None:
            ext.extcallback(ext, extdir, out_lib_path)
