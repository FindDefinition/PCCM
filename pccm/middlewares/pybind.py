import ast
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Union, Set

from ccimport import compat

from pccm.core import (Class, ConstructorMeta, ExternalFunctionMeta,
                       FunctionDecl, ManualClass, ManualClassGenerator, Member,
                       MemberFunctionMeta, MiddlewareMeta,
                       StaticMemberFunctionMeta, CodeSectionClassDef)
from pccm.core.buildmeta import _unique_list_keep_order
from pccm.core.codegen import Block, generate_code, generate_code_list
from pccm.core.markers import middleware_decorator

_AUTO_ANNO_TYPES = {
    "int": "int",
    "int8_t": "int",
    "int16_t": "int",
    "int32_t": "int",
    "int64_t": "int",
    "uint8_t": "int",
    "uint16_t": "int",
    "uint32_t": "int",
    "uint64_t": "int",
    "unsigned": "int",
    "long": "int",
    "short": "int",
    "float": "float",
    "double": "float",
    "unsigned long": "int",
    "unsigned int": "int",
    "bool": "bool",
    "std::string": "str",
    "void": "None",
}  # type: Dict[str, str]


def _get_attribute_name(node, parts):
    if isinstance(node, ast.Attribute):
        parts.append(node.attr)
        return _get_attribute_name(node.value, parts)
    elif isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        raise NotImplementedError


def get_attribute_name_parts(node):
    parts = []
    _get_attribute_name(node, parts)
    return parts[::-1]


def _anno_parser(node: ast.AST, imports: List[str]) -> str:
    if isinstance(node, ast.Module):
        assert len(node.body) == 1
        return _anno_parser(node.body[0], imports)
    if isinstance(node, ast.Expr):
        return _anno_parser(node.value, imports)
    elif isinstance(node, ast.Subscript):
        assert isinstance(node.value, (ast.Name, ast.Attribute))
        if compat.Python3_9AndLater:
            # >= 3.9 ast.Index is deprecated
            assert isinstance(node.slice, ast.Tuple)
            return "{}[{}]".format(_anno_parser(node.value, imports),
                                   _anno_parser(node.slice, imports))
        else:
            assert isinstance(node.slice, ast.Index)
            assert isinstance(
                node.slice.value,
                (ast.Tuple, ast.Attribute, ast.Name, ast.Subscript))
            return "{}[{}]".format(_anno_parser(node.value, imports),
                                   _anno_parser(node.slice.value, imports))
    elif isinstance(node, ast.Tuple):
        return ", ".join(_anno_parser(e, imports) for e in node.elts)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        value_parts = get_attribute_name_parts(node)  # type: List[str]
        from_import = "from {} import {}".format(".".join(value_parts[:-1]),
                                                 value_parts[-1])
        imports.append(from_import)
        return value_parts[-1]
    else:
        msg = "pyanno only support format like name.attr[nested_name.attr[name1, name2], name3]\n"
        msg += "but we get ast node {}".format(ast.dump(node))
        raise ValueError(msg)


def python_anno_parser(anno_str: str):
    tree = ast.parse(anno_str)
    imports = []  # type: List[str]
    refined_name = _anno_parser(tree, imports)
    return refined_name, imports


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


class MethodType(Enum):
    PropSetter = "PropSetter"
    PropGetter = "PropGetter"
    Normal = "Normal"


class Pybind11MethodMeta(Pybind11Meta):
    """may add some attributes in future.
    """
    def __init__(self,
                 bind_name: str = "",
                 method_type: MethodType = MethodType.Normal,
                 prop_name: str = "",
                 ret_policy: ReturnPolicy = ReturnPolicy.Auto,
                 call_guard: Optional[str] = None,
                 virtual: bool = False):
        super().__init__(Pybind11)
        self.bind_name = bind_name
        self.method_type = method_type
        self.prop_name = prop_name
        self.ret_policy = ret_policy
        self.call_guard = call_guard
        self.virtual = virtual


class Pybind11PropMeta(Pybind11Meta):
    """may add some attributes in future.
    """
    def __init__(self, readwrite: bool = True):
        super().__init__(Pybind11)
        self.readwrite = readwrite


class PybindMethodDecl(object):
    def __init__(self, decl: FunctionDecl, namespace: str, class_name: str,
                 mw_meta: Pybind11MethodMeta):
        self.decl = decl
        self.mw_meta = mw_meta
        if mw_meta.bind_name:
            self.func_name = mw_meta.bind_name
        else:
            self.func_name = decl.meta.name
        self.method_type = mw_meta.method_type
        member_meta = decl.meta
        if mw_meta.virtual:
            assert member_meta.virtual is True, "you must mark func as virtual member first."

        if mw_meta.method_type == MethodType.PropGetter:
            assert isinstance(decl.meta, MemberFunctionMeta)
            assert len(
                decl.code.arguments) == 0, "prop getter can't have argument"
        if mw_meta.method_type == MethodType.PropSetter:
            assert isinstance(decl.meta, MemberFunctionMeta)
            assert len(
                decl.code.arguments) == 1, "prop setter must have one argument"
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

        self.setter_pybind_decl = None  # type: Optional[PybindMethodDecl]
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
        func_name = self.func_name

        if isinstance(self.decl.meta, StaticMemberFunctionMeta):
            def_stmt = "def_static"
        if self.method_type == MethodType.PropGetter:
            def_stmt = "def_property_readonly"
            func_name = self.mw_meta.prop_name
        attrs = self.args.copy()
        attrs.append(self.mw_meta.ret_policy.value)
        if self.mw_meta.call_guard is not None:
            attrs.append("pybind11::call_guard<{}>()".format(
                self.mw_meta.call_guard))
        if self.setter_pybind_decl is not None:
            attrs.insert(0, self.setter_pybind_decl.addr)
            def_stmt = "def_property"
        # if self.decl.meta.mw_metas
        if attrs:
            return ".{}(\"{}\", {}, {})".format(def_stmt, func_name, self.addr,
                                                ", ".join(attrs))
        else:
            return ".{}(\"{}\", {})".format(def_stmt, func_name, self.addr)

    def get_virtual_string(self, parent_cls_name: str):
        fmt = "{} {} {{PYBIND11_OVERRIDE({}, {}, {}, {});}}"
        if self.decl.meta.pure_virtual:
            fmt = "{} {} {{PYBIND11_OVERRIDE_PURE({}, {}, {}, {});}}"
        post_meta_attrs = self.decl.meta.get_post_attrs()
        override = ""
        if "override" not in post_meta_attrs:
            override = "override"
        sig_str = self.decl.code.get_sig(self.decl.meta.name,
                                         self.decl.meta,
                                         withpost=True,
                                         with_semicolon=False,
                                         with_pure=False)
        arg_names = ", ".join(a.name for a in self.decl.code.arguments)
        return fmt.format(sig_str, override, self.decl.code.return_type,
                          parent_cls_name, self.func_name, arg_names)


class PybindPropDecl(object):
    def __init__(self, decl: Member, namespace: str, class_name: str,
                 mw_meta: Pybind11PropMeta):
        self.decl = decl
        self.mw_meta = mw_meta
        self.addr = "&{}::{}::{}".format(namespace.replace(".", "::"),
                                         class_name, self.decl.name)

    def to_string(self) -> str:
        def_stmt = "def_readwrite"
        if not self.mw_meta.readwrite:
            def_stmt = "def_readonly"
        return ".{}(\"{}\", {})".format(def_stmt, self.decl.name, self.addr)


class PybindClassMixin:
    def add_pybind_member(self: "Class",
                          name: str,
                          type: str,
                          default: Optional[str] = None,
                          array: Optional[str] = None,
                          pyanno: Optional[str] = None,
                          readwrite: bool = True,
                          mw_metas: Optional[List[MiddlewareMeta]] = None):
        if mw_metas is None:
            mw_metas = []
        mw_metas.append(Pybind11PropMeta(readwrite))
        return self.add_member(name=name,
                               type=type,
                               default=default,
                               array=array,
                               pyanno=pyanno,
                               mw_metas=mw_metas)


class Pybind11ClassHandler(ManualClass):
    # TODO split pybind defs to multiple file for faster compilation.
    # TODO handle inherit
    def __init__(self, module_name: str, file_suffix: str = ".cc"):
        super().__init__()
        self.ns_to_cls_to_func_prop_decls = OrderedDict(
        )  # type: Dict[str, Dict[str, List[Union[PybindMethodDecl, PybindPropDecl]]]]
        self.module_name = module_name
        self.add_include("pybind11/stl.h")
        self.add_include("pybind11/pybind11.h")
        self.add_include("pybind11/numpy.h")
        self.file_suffix = file_suffix
        self.built = False

    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl,
                             mw_meta: Pybind11MethodMeta):
        if cu.namespace not in self.ns_to_cls_to_func_prop_decls:
            self.ns_to_cls_to_func_prop_decls[cu.namespace] = OrderedDict()
        cls_to_func_decls = self.ns_to_cls_to_func_prop_decls[cu.namespace]
        if cu.class_name not in cls_to_func_decls:
            cls_to_func_decls[cu.class_name] = []
        cls_to_func_decls[cu.class_name].append(
            PybindMethodDecl(func_decl, cu.namespace, cu.class_name, mw_meta))

    def handle_member(self, cu: Class, member_decl: Member,
                      mw_meta: Pybind11PropMeta):
        if cu.namespace not in self.ns_to_cls_to_func_prop_decls:
            self.ns_to_cls_to_func_prop_decls[cu.namespace] = OrderedDict()
        cls_to_prop_decls = self.ns_to_cls_to_func_prop_decls[cu.namespace]
        if cu.class_name not in cls_to_prop_decls:
            cls_to_prop_decls[cu.class_name] = []
        cls_to_prop_decls[cu.class_name].append(
            PybindPropDecl(member_decl, cu.namespace, cu.class_name, mw_meta))

    def postprocess(self):
        # TODO handle overload
        # TODO handle inherit
        if self.built:
            return
        submodules = OrderedDict()  # type: Dict[str, str]
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
        virtual_class_defs = []  # type: List[Block]
        for ns, cls_to_decl in self.ns_to_cls_to_func_prop_decls.items():
            for cls_name, decls in cls_to_decl.items():
                # find virtual pybind functions.
                has_virtual = False
                virtual_decls = []  # type: List[PybindMethodDecl]
                for decl in decls:
                    if isinstance(decl,
                                  PybindMethodDecl) and decl.mw_meta.virtual:
                        has_virtual = True
                        virtual_decls.append(decl)
                virtual_cls_name = "Py" + cls_name
                if has_virtual:
                    virtual_cls_def = CodeSectionClassDef(
                        virtual_cls_name,
                        dep_alias=[],
                        code_before=[],
                        code_after=[],
                        external_funcs=[],
                        typedefs=["using {}::{};".format(cls_name, cls_name)],
                        static_consts=[],
                        functions=[
                            d.get_virtual_string(cls_name)
                            for d in virtual_decls
                        ],
                        members=[],
                        parent_class=cls_name)
                    ns_before, ns_after = virtual_cls_def.generate_namespace(
                        ns)
                    block = Block("\n".join(ns_before),
                                  [virtual_cls_def.to_block()],
                                  "\n".join(ns_after),
                                  indent=0)
                    virtual_class_defs.append(block)
                # for every getters, match a setter if possible.
                getter_prop_name_to_decl = {
                }  # type: Dict[str, PybindMethodDecl]
                for decl in decls:
                    if isinstance(
                            decl, PybindMethodDecl
                    ) and decl.method_type == MethodType.PropGetter:
                        prop_name = decl.mw_meta.prop_name
                        assert prop_name not in getter_prop_name_to_decl, "duplicate getter {}".format(
                            prop_name)
                        getter_prop_name_to_decl[decl.mw_meta.prop_name] = decl
                setter_prop_name = set()  # type: Set[str]

                for decl in decls:
                    if isinstance(
                            decl, PybindMethodDecl
                    ) and decl.method_type == MethodType.PropSetter:
                        prop_name = decl.mw_meta.prop_name
                        assert prop_name in getter_prop_name_to_decl
                        assert prop_name not in setter_prop_name, "duplicate setter {}".format(
                            prop_name)
                        getter_decl = getter_prop_name_to_decl[prop_name]
                        setter_prop_name.add(prop_name)
                        getter_decl.setter_pybind_decl = decl

                has_constructor = False
                for d in decls:
                    if isinstance(d, PybindMethodDecl) and isinstance(
                            d.decl.meta, ConstructorMeta):
                        has_constructor = True
                        break
                cls_qual_name = "{}::{}".format(ns.replace(".", "::"),
                                                cls_name)
                cls_def_arguments = [cls_qual_name]
                if has_virtual:
                    cls_def_arguments.append("{}::{}".format(
                        ns.replace(".", "::"), virtual_cls_name))
                cls_def_argu_str = ", ".join(cls_def_arguments)
                submod = submodules[ns]
                cls_def = "pybind11::class_<{cls_def_argu_str}>({submod}, \"{cls_name}\")".format(
                    cls_def_argu_str=cls_def_argu_str,
                    submod=submod,
                    cls_name=cls_name)
                cls_method_defs = []  # type: List[str]
                if not has_constructor:
                    cls_method_defs.append(".def(pybind11::init<>())")
                for decl in decls:
                    if isinstance(
                            decl, PybindMethodDecl
                    ) and decl.method_type == MethodType.PropSetter:
                        continue
                    cls_method_defs.append(decl.to_string())

                cls_method_defs[-1] += ";"
                cls_def_block = Block(cls_def, cls_method_defs, "")
                class_defs.append(cls_def_block)
        code_block = Block("PYBIND11_MODULE({}, m){{".format(self.module_name),
                           sub_defs + class_defs, "}")
        code = generate_code_list([*virtual_class_defs, code_block], 0, 2)
        self.add_impl_main("{}_pybind_main".format(self.module_name),
                           "\n".join(code), self.file_suffix)
        self.built = True

    def _extract_anno_with_cache(self, user_anno: Optional[str], type_str: str,
                                 type_to_anno_cache: Dict[str, str]):
        from_imports = []  # type: List[str]
        if user_anno is None:
            if type_str in _AUTO_ANNO_TYPES:
                user_anno = _AUTO_ANNO_TYPES[type_str]
                return user_anno, from_imports
        if user_anno is None:
            # find anno in cache
            if type_str in type_to_anno_cache:
                user_anno = type_to_anno_cache[type_str]
        anno = None
        if user_anno is not None:
            anno, from_imports = python_anno_parser(user_anno)
            type_to_anno_cache[type_str] = user_anno
        return anno, from_imports

    def generate_python_interface(self):
        """
        dep_imports
        class xxx:
            prop decls
            methods (overloaded methods)
        TODO handle c++ operators
        TODO better code
        TODO auto generate STL annotations
        TODO insert docstring if exists
        """
        init_import = "from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union"
        ns_to_interface = OrderedDict()  # type: Dict[str, str]
        type_to_anno_cache = {}  # type: Dict[str, str]
        for ns, cls_to_decl in self.ns_to_cls_to_func_prop_decls.items():
            imports = [init_import]  # type: List[str]
            ns_cls_blocks = []  # type: List[Block]
            for cls_name, decls in cls_to_decl.items():
                # split decl to method decl and prop decl
                method_decls = []  # type: List[PybindMethodDecl]
                prop_decls = []  # type: List[PybindPropDecl]
                for decl in decls:
                    if isinstance(decl, PybindMethodDecl):
                        method_decls.append(decl)
                    else:
                        prop_decls.append(decl)
                name_to_overloaded = OrderedDict(
                )  # type: Dict[str, List[PybindMethodDecl]]
                decl_codes = []  # type: List[Union[Block, str]]
                for prop_decl in prop_decls:
                    prop_anno = "Any"
                    prop_type = prop_decl.decl.type_str
                    user_anno = prop_decl.decl.pyanno
                    anno, from_imports = self._extract_anno_with_cache(
                        user_anno, prop_type, type_to_anno_cache)
                    if anno is not None:
                        prop_anno = anno
                    imports.extend(from_imports)
                    if prop_anno == cls_name:
                        prop_anno = "\"{}\"".format(prop_anno)

                    decl_codes.append("{}: {}".format(prop_decl.decl.name,
                                                      prop_anno))
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
                        user_anno = pydecl.decl.code.ret_pyanno
                        ret_type = pydecl.decl.code.return_type
                        anno, from_imports = self._extract_anno_with_cache(
                            user_anno, ret_type, type_to_anno_cache)
                        if anno is not None:
                            if anno == cls_name:
                                anno = "\"{}\"".format(anno)
                            res_anno = " -> {}".format(anno)
                            imports.extend(from_imports)
                        decl_func_name = func_name
                        if isinstance(pydecl.decl.meta, ConstructorMeta):
                            decl_func_name = "__init__"

                        arg_names = []  # type: List[str]
                        for arg in pydecl.decl.code.arguments:
                            user_anno = arg.pyanno
                            user_anno, from_imports = self._extract_anno_with_cache(
                                user_anno, arg.type_str, type_to_anno_cache)
                            if user_anno is not None:
                                if user_anno == cls_name:
                                    user_anno = "\"{}\"".format(user_anno)
                                imports.extend(from_imports)
                                arg_names.append("{}: {}".format(
                                    arg.name, user_anno))
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
                        if pydecl.method_type == MethodType.PropGetter:
                            decorator = "@property\n"
                            decl_func_name = pydecl.mw_meta.prop_name
                        elif pydecl.method_type == MethodType.PropSetter:
                            decorator = "@{}.setter\n".format(
                                pydecl.mw_meta.prop_name)
                            decl_func_name = pydecl.mw_meta.prop_name

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
        self.singleton = Pybind11ClassHandler(module_name, file_suffix)
        self.singleton.graph_inited = True

    def create_manual_class(self, cu: Class) -> ManualClass:
        self.singleton._unified_deps.append(cu)
        return self.singleton

    def get_code_units(self) -> List[Class]:
        self.singleton.postprocess()
        return [self.singleton]

    def generate_python_interface(self):
        return self.singleton.generate_python_interface()


def mark(func=None,
         bind_name: str = "",
         prop_name: str = "",
         method_type: MethodType = MethodType.Normal,
         ret_policy: ReturnPolicy = ReturnPolicy.Auto,
         virtual: bool = False,
         nogil: bool = False):
    if virtual:
        assert not nogil, "you can't release gil for python virtual function."
    call_guard = None  # type: Optional[str]
    if nogil:
        call_guard = "pybind11::gil_scoped_release"
    pybind_meta = Pybind11MethodMeta(bind_name, method_type, prop_name,
                                     ret_policy, call_guard, virtual)
    return middleware_decorator(func, pybind_meta)


def mark_prop_getter(func=None,
                     prop_name: str = "",
                     ret_policy: ReturnPolicy = ReturnPolicy.Auto,
                     nogil: bool = False):
    return mark(func,
                "",
                prop_name,
                MethodType.PropGetter,
                ret_policy,
                nogil=nogil)


def mark_prop_setter(func=None,
                     prop_name: str = "",
                     ret_policy: ReturnPolicy = ReturnPolicy.Auto,
                     nogil: bool = False):
    return mark(func,
                "",
                prop_name,
                MethodType.PropSetter,
                ret_policy,
                nogil=nogil)


pybind_mark = mark

if __name__ == "__main__":
    # print(ast.parse)
    print(python_anno_parser("Tuple[Tuple[spconv.Tensor, int], float]"))
    # python_anno_parser("Tuple[Tuple[spconv.Tensor, int], float]")
