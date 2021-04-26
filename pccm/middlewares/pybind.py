from typing import Dict, List, Union, Optional

from pccm.core import (Class, ConstructorMeta, ExternalFunctionMeta,
                       FunctionDecl, ManualClass, ManualClassGenerator, Member,
                       MemberFunctionMeta, MiddlewareMeta,
                       StaticMemberFunctionMeta)
from pccm.core.codegen import Block, generate_code, generate_code_list
from pccm.core.markers import middleware_decorator
from pccm.core.buildmeta import _unique_list_keep_order
from enum import Enum


class Pybind11Meta(MiddlewareMeta):
    """may add some attributes in future.
    """
    pass


class ReturnPolicy(Enum):
    TakeOwnerShip = "pybind11::return_value_policy::take_ownership"
    Copy = "pybind11::return_value_policy::copy"
    Move = "pybind11::return_value_policy::move"
    Ref = "pybind11::return_value_policy::reference"
    RefInternal = "pybind11::return_value_policy::reference_internal"
    Auto = "pybind11::return_value_policy::automatic"
    AutoRef = "pybind11::return_value_policy::automatic_reference"


class Pybind11MethodMeta(Pybind11Meta):
    """may add some attributes in future.
    """
    def __init__(self,
                 ret_policy: ReturnPolicy = ReturnPolicy.Auto,
                 call_guard: Optional[str] = None):
        super().__init__(Pybind11)
        self.ret_policy = ret_policy
        self.call_guard = call_guard


class Pybind11PropMeta(Pybind11Meta):
    """may add some attributes in future.
    """
    def __init__(self,
                 readwrite: bool = True):
        super().__init__(Pybind11)
        self.readwrite = readwrite


class PybindMethodDecl(object):
    def __init__(self, decl: FunctionDecl, namespace: str, class_name: str,
                 mw_meta: Pybind11MethodMeta):
        self.decl = decl
        self.mw_meta = mw_meta
        self.func_name = decl.meta.name
        if decl.code.is_template():
            raise ValueError("pybind can't bind template function")
        if isinstance(decl.meta, ConstructorMeta):
            arg_types = [a.type_str for a in decl.code.arguments]
            self.addr = "pybind11::init<{}>()".format(", ".join(arg_types))
        elif isinstance(decl.meta,
                        (MemberFunctionMeta, StaticMemberFunctionMeta)):
            self.addr = "&{}::{}::{}".format(namespace.replace(".", "::"),
                                             class_name, self.func_name)
        else:
            raise NotImplementedError
        self.args = []  # type: List[str]
        for argu in decl.code.arguments:
            if argu.default:
                self.args.append("pybind11::arg(\"{}\") = {}".format(
                    argu.name, argu.default))
            else:
                self.args.append("pybind11::arg(\"{}\")".format(argu.name))

    def to_string(self) -> str:
        if isinstance(self.decl.meta, ConstructorMeta):
            return ".def({}, {})".format(self.addr, ", ".join(self.args))
        def_stmt = "def"
        if isinstance(self.decl.meta, StaticMemberFunctionMeta):
            def_stmt = "def_static"
        attrs = self.args.copy()
        attrs.append(self.mw_meta.ret_policy.value)
        if self.mw_meta.call_guard is not None:
            attrs.append("pybind11::call_guard<{}>()".format(
                self.mw_meta.call_guard))
        # if self.decl.meta.mw_metas
        if attrs:
            return ".{}(\"{}\", {}, {})".format(def_stmt, self.func_name,
                                                self.addr, ", ".join(attrs))
        else:
            return ".{}(\"{}\", {})".format(def_stmt, self.func_name,
                                            self.addr)

class PybindPropDecl(object):
    def __init__(self, decl: Member, namespace: str, class_name: str, mw_meta: Pybind11PropMeta):
        self.decl = decl
        self.mw_meta = mw_meta
        self.addr = "&{}::{}::{}".format(namespace.replace(".", "::"),
                                        class_name, self.decl.name)


    def to_string(self) -> str:
        def_stmt = "def_readwrite"
        if not self.mw_meta.readwrite:
            def_stmt = "def_read_only"
        return ".{}(\"{}\", {})".format(def_stmt, self.decl.name,
                                        self.addr)


class Pybind11Class(ManualClass):
    # TODO split pybind defs to multiple file for faster compilation.
    # TODO handle inherit
    def __init__(self, module_name: str, file_suffix: str = ".cc"):
        super().__init__()
        self.ns_to_cls_to_func_prop_decls = {
        }  # type: Dict[str, Dict[str, List[Union[PybindMethodDecl, PybindPropDecl]]]]
        self.ns_to_cls_to_properties = {
        }  # type: Dict[str, Dict[str, List[PybindPropDecl]]]

        self.module_name = module_name
        self.add_include("pybind11/stl.h")
        self.add_include("pybind11/pybind11.h")
        self.add_include("pybind11/numpy.h")
        self.file_suffix = file_suffix
        self.built = False

    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl,
                             mw_meta: Pybind11MethodMeta):
        if cu.namespace not in self.ns_to_cls_to_func_prop_decls:
            self.ns_to_cls_to_func_prop_decls[cu.namespace] = {}
        cls_to_func_decls = self.ns_to_cls_to_func_prop_decls[cu.namespace]
        if cu.class_name not in cls_to_func_decls:
            cls_to_func_decls[cu.class_name] = []
        cls_to_func_decls[cu.class_name].append(
            PybindMethodDecl(func_decl, cu.namespace, cu.class_name, mw_meta))

    def handle_member(self, cu: Class, member_decl: Member,
                      mw_meta: Pybind11PropMeta):
        if cu.namespace not in self.ns_to_cls_to_func_prop_decls:
            self.ns_to_cls_to_func_prop_decls[cu.namespace] = {}
        cls_to_prop_decls = self.ns_to_cls_to_func_prop_decls[cu.namespace]
        if cu.class_name not in cls_to_prop_decls:
            cls_to_prop_decls[cu.class_name] = []
        cls_to_prop_decls[cu.class_name].append(
            PybindPropDecl(member_decl, cu.namespace, cu.class_name, mw_meta))

    def postprocess(self):
        if self.built:
            return
        submodules = {}  # type: Dict[str, str]
        sub_defs = []  # type: List[str]
        for ns in self.ns_to_cls_to_func_prop_decls.keys():
            ns_parts = ns.split(".")
            for i in range(1, len(ns_parts) + 1):
                sub_name = "_".join(ns_parts[:i])
                sub_ns = ".".join(ns_parts[:i])
                if i == 1:
                    parent_name = "m"
                else:
                    parent_name = "m_{}".format("_".join(ns_parts[:i - 1]))
                if sub_ns not in submodules:
                    stmt = "pybind11::module_ m_{} = {}.def_submodule(\"{}\");".format(
                        sub_name, parent_name, ns_parts[i - 1])
                    sub_defs.append(stmt)
                    submodules[sub_ns] = "m_{}".format(sub_name)
        class_defs = []  # type: List[Block]
        for ns, cls_to_decl in self.ns_to_cls_to_func_prop_decls.items():
            for cls_name, decls in cls_to_decl.items():
                has_constructor = False
                for d in decls:
                    if isinstance(d, PybindMethodDecl) and isinstance(d.decl.meta, ConstructorMeta):
                        has_constructor = True
                        break
                cls_qual_name = "{}::{}".format(ns.replace(".", "::"),
                                                cls_name)
                submod = submodules[ns]
                cls_def = "pybind11::class_<{cls_qual_name}>({submod}, \"{cls_name}\")".format(
                    cls_qual_name=cls_qual_name,
                    submod=submod,
                    cls_name=cls_name)
                cls_method_defs = []  # type: List[str]
                if not has_constructor:
                    cls_method_defs.append(".def(pybind11::init<>())")
                for decl in decls:
                    cls_method_defs.append(decl.to_string())

                cls_method_defs[-1] += ";"
                cls_def_block = Block(cls_def, cls_method_defs, "")
                class_defs.append(cls_def_block)
        code_block = Block("PYBIND11_MODULE({}, m){{".format(self.module_name),
                           sub_defs + class_defs, "}")
        code = generate_code(code_block, 0, 2)
        self.add_impl_main("{}_pybind_main".format(self.module_name),
                           "\n".join(code), self.file_suffix)
        self.built = True

    def _get_import_and_anno_from_raw(self, anno_raw: str):
        mod_parts = anno_raw.split(".")
        if len(mod_parts) == 1:
            from_import = None 
        else:
            from_import = "from {} import {}".format(
                    ".".join(mod_parts[:-1]), mod_parts[-1])
        return mod_parts[-1], from_import

    def generate_python_interface(self):
        """
        dep_imports
        class xxx:
            prop decls
            methods (overloaded methods)
        TODO handle c++ operators
        TODO better code
        """
        init_import = "from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union"
        ns_to_interface = {}  # type: Dict[str, str]
        for ns, cls_to_decl in self.ns_to_cls_to_func_prop_decls.items():
            imports = [init_import]  # type: List[str]
            ns_cls_blocks = []  # type: List[Block]
            for cls_name, decls in cls_to_decl.items():
                # split decl to method decl and prop decl 
                method_decls = [] # type: List[PybindMethodDecl]
                prop_decls = [] # type: List[PybindPropDecl]
                for decl in decls:
                    if isinstance(decl, PybindMethodDecl):
                        method_decls.append(decl)
                    else:
                        prop_decls.append(decl)
                name_to_overloaded = {
                }  # type: Dict[str, List[PybindMethodDecl]]
                decl_codes = []  # type: List[Union[Block, str]]
                for prop_decl in prop_decls:
                    prop_anno = "Any"
                    if prop_decl.decl.pyanno is not None:
                        prop_anno, from_import = self._get_import_and_anno_from_raw(prop_decl.decl.pyanno)
                        if from_import is not None:
                            imports.append(from_import)
                    decl_codes.append("{}: {}".format(prop_decl.decl.name, prop_anno))
                for decl in method_decls:
                    if decl.func_name not in name_to_overloaded:
                        name_to_overloaded[decl.func_name] = []
                    name_to_overloaded[decl.func_name].append(decl)
                for func_name, over_decls in name_to_overloaded.items():
                    if len(over_decls) == 1:
                        fmt = "{}def {}({}){}: ..."
                    else:
                        fmt = "@overload\n{}def {}({}){}: ..."
                    for pydecl in over_decls:
                        res_anno = ""
                        if pydecl.decl.code.ret_pyanno is not None:
                            res_anno_raw = pydecl.decl.code.ret_pyanno
                            res_anno, from_import = self._get_import_and_anno_from_raw(res_anno_raw)
                            if from_import is not None:
                                imports.append(from_import)
                            res_anno = " -> {}".format(res_anno)

                        decl_func_name = func_name
                        if isinstance(pydecl.decl.meta, ConstructorMeta):
                            decl_func_name = "__init__"
                        arg_names = []  # type: List[str]
                        for arg in pydecl.decl.code.arguments:
                            anno = arg.pyanno
                            if anno is not None:
                                if anno == cls_name:
                                    anno_str = "\"{}\"".format(anno)
                                else:
                                    mod_parts = anno.split(".")
                                    anno_str = mod_parts[-1]
                                    anno_str, from_import = self._get_import_and_anno_from_raw(anno)
                                    if from_import is not None:
                                        imports.append(from_import)
                                arg_names.append("{}: {}".format(
                                    arg.name, anno_str))
                            else:
                                arg_names.append(arg.name)
                        if not isinstance(
                                pydecl.decl.meta,
                            (ExternalFunctionMeta, StaticMemberFunctionMeta)):
                            arg_names.insert(0, "self")
                        py_sig = ", ".join(arg_names)
                        decorator = ""
                        if isinstance(pydecl.decl.meta,
                                      StaticMemberFunctionMeta):
                            decorator = "@staticmethod\n"
                        decl_codes.append(
                            fmt.format(decorator, decl_func_name, py_sig,
                                       res_anno))
                class_block = Block("class {}:".format(cls_name), decl_codes)
                ns_cls_blocks.append(class_block)
            imports = _unique_list_keep_order(imports)

            ns_to_interface[ns] = "\n".join(
                generate_code_list(imports + ns_cls_blocks, 0, 4))
        return ns_to_interface


class Pybind11(ManualClassGenerator):
    def __init__(self,
                 module_name: str,
                 subnamespace: str,
                 file_suffix: str = ".cc"):
        super().__init__(subnamespace)
        self.module_name = module_name
        self.singleton = Pybind11Class(module_name, file_suffix)
        self.singleton.graph_inited = True

    def create_manual_class(self, cu: Class) -> ManualClass:
        self.singleton._unified_deps.append(cu)
        return self.singleton

    def get_code_units(self) -> List[Class]:
        self.singleton.postprocess()
        return [self.singleton]

    def generate_python_interface(self):
        return self.singleton.generate_python_interface()


def pybind_mark(func=None,
                ret_policy: ReturnPolicy = ReturnPolicy.Auto,
                nogil: bool = False):
    call_guard = None # type: Optional[str]
    if nogil:
        call_guard = "pybind11::gil_scoped_release"
    pybind_meta = Pybind11MethodMeta(ret_policy, call_guard)
    return middleware_decorator(func, pybind_meta)
