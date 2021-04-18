from typing import Dict, List

from pccm.core import (Class, ConstructorMeta, FunctionDecl, ManualClass,
                       ManualClassGenerator, Member, MemberFunctionMeta,
                       MiddlewareMeta)
from pccm.core.codegen import Block, generate_code
from pccm.core.markers import middleware_decorator


class Pybind11Meta(MiddlewareMeta):
    """may add some attributes in future.
    """
    pass


class PybindMethodDecl(object):
    def __init__(self, decl: FunctionDecl, namespace: str, class_name: str):
        self.decl = decl
        self.func_name = decl.meta.name
        if isinstance(decl.meta, ConstructorMeta):
            arg_types = [a.type_str for a in decl.code.arguments]
            self.addr = "pybind11::init<{}>()".format(", ".join(arg_types))
        elif isinstance(decl.meta, MemberFunctionMeta):
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
        if self.args:
            return ".def(\"{}\", {}, {})".format(self.func_name, self.addr,
                                                 ", ".join(self.args))
        else:
            return ".def(\"{}\", {})".format(self.func_name, self.addr)


class Pybind11Class(ManualClass):
    # TODO split pybind defs to multiple file for faster compilation.
    def __init__(self, module_name: str, file_suffix: str = ".cc"):
        super().__init__()
        self.ns_to_cls_to_func_decls = {
        }  # type: Dict[str, Dict[str, List[PybindMethodDecl]]]
        self.module_name = module_name
        self.add_include("pybind11/stl.h")
        self.add_include("pybind11/pybind11.h")
        self.add_include("pybind11/numpy.h")
        self.file_suffix = file_suffix
        self.built = False

    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl):
        if cu.namespace not in self.ns_to_cls_to_func_decls:
            self.ns_to_cls_to_func_decls[cu.namespace] = {}
        cls_to_func_decls = self.ns_to_cls_to_func_decls[cu.namespace]
        if cu.class_name not in cls_to_func_decls:
            cls_to_func_decls[cu.class_name] = []
        cls_to_func_decls[cu.class_name].append(
            PybindMethodDecl(func_decl, cu.namespace, cu.class_name))

    def handle_member(self, cu: Class, member_decl: Member):
        # TODO add suport for pybind member.
        pass

    def postprocess(self):
        if self.built:
            return
        submodules = {}
        sub_defs = []  # type: List[str]
        for ns in self.ns_to_cls_to_func_decls.keys():
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
        for ns, cls_to_decl in self.ns_to_cls_to_func_decls.items():
            for cls_name, decls in cls_to_decl.items():
                has_constructor = False
                for d in decls:
                    if isinstance(d.decl.meta, ConstructorMeta):
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


def pybind_mark(func=None):
    pybind_meta = Pybind11Meta(Pybind11)
    return middleware_decorator(func, pybind_meta)
