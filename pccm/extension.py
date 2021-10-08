from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

import abc
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from pccm.core import Class
import pccm 
from ccimport import compat

class ExtCallback(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ext: Extension,
                 extdir: Path, target_path: Path):
        pass


class PCCMExtension(Extension):
    def __init__(self, cus: List[Class], out_path: Union[str, Path], namespace_root: Optional[Union[str, Path]] = None,
                     extcallback: Optional[ExtCallback] = None):
        # don't invoke the original build_ext for this special extension
        out_path_p = Path(out_path)
        super().__init__(out_path_p.stem, sources=[])
        self.cus = cus
        self.out_path = out_path_p
        self.namespace_root = namespace_root
        self.extcallback = extcallback


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
            Path(self.build_temp).mkdir(exist_ok=True, parents=True, mode=0o755)
        build_out_path = Path.cwd() / Path(
            self.build_temp) / ext.out_path
        build_out_path.parent.mkdir(exist_ok=True, parents=True, mode=0o755)
        out_path.parent.mkdir(exist_ok=True, parents=True, mode=0o755)
        libpaths = []
        lib_path = pccm.builder.build_pybind(ext.cus,
                                        build_out_path,
                                        namespace_root=ext.namespace_root,
                                        verbose=False,
                                        load_library=False)
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
