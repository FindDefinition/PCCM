from pathlib import Path
from typing import Dict, List, Optional, Union

import ccimport

from pccm.core import Class, CodeGenerator
from pccm.middlewares import pybind
from pccm.core.buildmeta import BuildMeta

def build_pybind(cus: List[Class],
                 out_path: Union[str, Path],
                 includes: Optional[List[Union[str, Path]]] = None,
                 libpaths: Optional[List[Union[str, Path]]] = None,
                 libraries: Optional[List[str]] = None,
                 compile_options: Optional[List[str]] = None,
                 link_options: Optional[List[str]] = None,
                 std="c++14",
                 disable_hash=True,
                 load_library=True,
                 pybind_file_suffix: str = ".cc",
                 additional_cflags: Optional[Dict[str, List[str]]] = None,
                 additional_lflags: Optional[Dict[str, List[str]]] = None,
                 msvc_deps_prefix="Note: including file:",
                 build_dir: Optional[Union[str, Path]] = None,
                 namespace_root: Optional[Union[str, Path]] = None,
                 verbose=False):
    mod_name = Path(out_path).stem
    if build_dir is None:
        build_dir = Path(out_path).parent / "build"
    if includes is None:
        includes = []
    if libpaths is None:
        libpaths = []
    if libraries is None:
        libraries = []
    if additional_cflags is None:
        additional_cflags = {}
    if additional_lflags is None:
        additional_lflags = {}

    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True)
    pb = pybind.Pybind11(mod_name, mod_name, pybind_file_suffix)
    cg = CodeGenerator([pb])
    cg.build_graph(cus, namespace_root)
    header_dict, impl_dict = cg.code_generation(cg.get_code_units())
    HEADER_ROOT = build_dir / "include"
    SRC_ROOT = build_dir / "src"
    includes.append(HEADER_ROOT)
    extern_build_meta = BuildMeta(includes, libpaths, libraries, additional_cflags, additional_lflags)
    for cu in cg.get_code_units():
        extern_build_meta += cu.build_meta
    cg.code_written(HEADER_ROOT, header_dict)
    paths = cg.code_written(SRC_ROOT, impl_dict)
    header_dict, impl_dict = cg.code_generation(pb.get_code_units())
    cg.code_written(HEADER_ROOT, header_dict)
    paths += cg.code_written(SRC_ROOT, impl_dict)
    return ccimport.ccimport(paths,
                             out_path,
                             extern_build_meta.includes,
                             extern_build_meta.libpaths,
                             extern_build_meta.libraries,
                             compile_options,
                             link_options,
                             std=std,
                             source_paths_for_hash=None,
                             disable_hash=disable_hash,
                             load_library=load_library,
                             additional_cflags=extern_build_meta.compiler_to_cflags,
                             additional_lflags=extern_build_meta.compiler_to_ldflags,
                             msvc_deps_prefix=msvc_deps_prefix,
                             verbose=verbose)
