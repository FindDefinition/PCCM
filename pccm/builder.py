from pathlib import Path
from typing import Dict, List, Optional, Union

import ccimport
from ccimport.buildtools.writer import DEFAULT_MSVC_DEP_PREFIX

from pccm.core import Class, CodeFormatter, CodeGenerator
from pccm.core.buildmeta import BuildMeta
from pccm.middlewares import pybind


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
                 msvc_deps_prefix=DEFAULT_MSVC_DEP_PREFIX,
                 build_dir: Optional[Union[str, Path]] = None,
                 namespace_root: Optional[Union[str, Path]] = None,
                 code_fmt: Optional[CodeFormatter] = None,
                 out_root: Optional[Union[str, Path]] = None,
                 suffix_to_compiler: Optional[Dict[str, List[str]]] = None,
                 disable_pch: bool = False,
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
    if out_root is None:
        out_root = build_dir
    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
    pb = pybind.Pybind11(mod_name, mod_name, pybind_file_suffix)
    cg = CodeGenerator([pb], verbose=verbose)
    cg.build_graph(cus, namespace_root)
    header_dict, impl_dict, header_to_impls = cg.code_generation(cg.get_code_units())
    HEADER_ROOT = build_dir / "include"
    SRC_ROOT = build_dir / "src"
    pch_to_sources = {} # type: Dict[Path, List[Path]]
    if not disable_pch:
        for header, impls in header_to_impls.items():
            pch_to_sources[HEADER_ROOT / header] = [SRC_ROOT / p for p in impls]
    includes.append(HEADER_ROOT)
    extern_build_meta = BuildMeta(includes, libpaths, libraries,
                                  additional_cflags, additional_lflags)
    for cu in cg.get_code_units():
        extern_build_meta += cu.build_meta
    cg.code_written(HEADER_ROOT, header_dict, code_fmt)
    paths = cg.code_written(SRC_ROOT, impl_dict, code_fmt)
    header_dict, impl_dict, header_to_impls = cg.code_generation(pb.get_code_units())
    cg.code_written(HEADER_ROOT, header_dict, code_fmt)
    paths += cg.code_written(SRC_ROOT, impl_dict, code_fmt)
    pyi = pb.generate_python_interface()
    for k, v in pyi.items():
        k_path = k.replace(".", "/") + ".pyi"
        k_path_parts = k.split(".")[:-1]
        pyi_path = Path(out_path) / k_path
        pyi_path.parent.mkdir(exist_ok=True, parents=True, mode=0o755)
        mk_init = Path(out_path)
        init_path = (mk_init / "__init__.pyi")
        if not init_path.exists():
            with init_path.open("w") as f:
                f.write("")
        for part in k_path_parts:
            init_path = (mk_init / part / "__init__.pyi")
            if not init_path.exists():
                with init_path.open("w") as f:
                    f.write("")
            mk_init = mk_init / part
        with pyi_path.open("w") as f:
            f.write(v)
    return ccimport.ccimport(
        paths,
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
        build_dir=build_dir,
        out_root=out_root,
        pch_to_sources=pch_to_sources,
        suffix_to_compiler=suffix_to_compiler,
        verbose=verbose)
