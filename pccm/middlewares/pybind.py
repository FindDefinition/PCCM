import ast
from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Union

from ccimport import compat

from pccm.core import (Class, CodeSectionClassDef, ConstructorMeta, EnumClass,
                       ExternalFunctionMeta, FunctionCode, FunctionDecl,
                       ManualClass, ManualClassGenerator, Member,
                       MemberFunctionMeta, MiddlewareMeta, ParameterizedClass,
                       StaticMemberFunctionMeta)
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

_IDENTITY_DEFAULT_HANDLER = lambda x: x


def _bool_default_handler(cpp_value: str):
    if cpp_value == "true":
        return "True"
    elif cpp_value == "false":
        return "False"
    else:
        return cpp_value


_AUTO_ANNO_TYPES_DEFAULT_HANDLER = {
    "int": _IDENTITY_DEFAULT_HANDLER,
    "int8_t": _IDENTITY_DEFAULT_HANDLER,
    "int16_t": _IDENTITY_DEFAULT_HANDLER,
    "int32_t": _IDENTITY_DEFAULT_HANDLER,
    "int64_t": _IDENTITY_DEFAULT_HANDLER,
    "uint8_t": _IDENTITY_DEFAULT_HANDLER,
    "uint16_t": _IDENTITY_DEFAULT_HANDLER,
    "uint32_t": _IDENTITY_DEFAULT_HANDLER,
    "uint64_t": _IDENTITY_DEFAULT_HANDLER,
    "unsigned": _IDENTITY_DEFAULT_HANDLER,
    "long": _IDENTITY_DEFAULT_HANDLER,
    "short": _IDENTITY_DEFAULT_HANDLER,
    "float": _IDENTITY_DEFAULT_HANDLER,
    "double": _IDENTITY_DEFAULT_HANDLER,
    "unsigned long": _IDENTITY_DEFAULT_HANDLER,
    "unsigned int": _IDENTITY_DEFAULT_HANDLER,
    "bool": _bool_default_handler,
    "std::string": _IDENTITY_DEFAULT_HANDLER,
    "void": _IDENTITY_DEFAULT_HANDLER,
}  # type: Dict[str, Callable[[str], str]]


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


class TemplateTypeStmt(object):
    def __init__(self, name: str, args: List[Union[str, "TemplateTypeStmt"]]):
        self.name = name
        self.args = args


def _simple_template_type_parser(stmt: str, start: int, end: int):
    # find '<'
    assert stmt[end - 1] == ">"
    left = -1
    for i in range(start, end):
        pass
    pass


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
        super().__init__(Pybind11SplitImpl)
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
        super().__init__(Pybind11SplitImpl)
        self.readwrite = readwrite


class PybindMethodDecl(object):
    def __init__(self, decl: FunctionDecl, namespace: str, class_name: str,
                 mw_meta: Pybind11MethodMeta):
        self.decl = decl
        self.mw_meta = mw_meta
        if mw_meta.bind_name:
            self.func_name = mw_meta.bind_name
        else:
            self.func_name = decl.get_function_name()
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

    def get_overload_addr(self):
        addr = self.addr
        arg_types = ", ".join([a.type_str for a in self.decl.code.arguments])
        meta = self.decl.meta
        addr_fmt = "pybind11::overload_cast<{}>({})"
        if isinstance(meta, MemberFunctionMeta):
            if meta.const:
                addr_fmt = "pybind11::overload_cast<{}>({}, pybind11::const_)"
        addr = addr_fmt.format(arg_types, self.addr)
        return addr

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
        addr = self.addr
        if self.decl.is_overload:
            addr = self.get_overload_addr()
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
            return ".{}(\"{}\", {}, {})".format(def_stmt, func_name, addr,
                                                ", ".join(attrs))
        else:
            return ".{}(\"{}\", {})".format(def_stmt, func_name, addr)

    def get_virtual_string(self, parent_cls_name: str):
        fmt = "{} {} {{PYBIND11_OVERRIDE({}, {}, {}, {});}}"
        if self.decl.meta.pure_virtual:
            fmt = "{} {} {{PYBIND11_OVERRIDE_PURE({}, {}, {}, {});}}"
        post_meta_attrs = self.decl.meta.get_post_attrs()
        override = ""
        if "override" not in post_meta_attrs:
            override = "override"
        sig_str = self.decl.code.get_sig(self.decl.get_function_name(),
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


def _postprocess_class(cls_name: str, cls_namespace: str, submod: str,
                       decls: List[Union[PybindMethodDecl, PybindPropDecl]],
                       enum_classes: List[EnumClass]):
    has_virtual = False
    vblock = None  # Optional[Block]
    virtual_decls = []  # type: List[PybindMethodDecl]
    method_decls = []  # type: List[PybindMethodDecl]
    prop_decls = []  # type: List[PybindPropDecl]
    for decl in decls:
        if isinstance(decl, PybindMethodDecl):
            method_decls.append(decl)
        else:
            prop_decls.append(decl)
    for decl in method_decls:
        if decl.mw_meta.virtual:
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
            enum_classes=[],
            typedefs=["using {}::{};".format(cls_name, cls_name)],
            static_consts=[],
            functions=[d.get_virtual_string(cls_name) for d in virtual_decls],
            members=[],
            parent_class=cls_name)
        ns_before, ns_after = virtual_cls_def.generate_namespace(cls_namespace)
        vblock = Block("\n".join(ns_before), [virtual_cls_def.to_block()],
                       "\n".join(ns_after),
                       indent=0)
    # for every getters, match a setter if possible.
    getter_prop_name_to_decl = {}  # type: Dict[str, PybindMethodDecl]
    for decl in method_decls:
        if decl.method_type == MethodType.PropGetter:
            prop_name = decl.mw_meta.prop_name
            assert prop_name not in getter_prop_name_to_decl, "duplicate getter {}".format(
                prop_name)
            getter_prop_name_to_decl[decl.mw_meta.prop_name] = decl
    setter_prop_name = set()  # type: Set[str]

    for decl in method_decls:
        if decl.method_type == MethodType.PropSetter:
            prop_name = decl.mw_meta.prop_name
            assert prop_name in getter_prop_name_to_decl
            assert prop_name not in setter_prop_name, "duplicate setter {}".format(
                prop_name)
            getter_decl = getter_prop_name_to_decl[prop_name]
            setter_prop_name.add(prop_name)
            getter_decl.setter_pybind_decl = decl

    has_constructor = False
    for d in method_decls:
        if isinstance(d.decl.meta, ConstructorMeta):
            has_constructor = True
            break
    cls_qual_name = "{}::{}".format(cls_namespace.replace(".", "::"), cls_name)
    cls_def_arguments = [cls_qual_name]
    if has_virtual:
        cls_def_arguments.append("{}::{}".format(
            cls_namespace.replace(".", "::"), virtual_cls_name))
    cls_def_argu_str = ", ".join(cls_def_arguments)
    cls_def_name = "{}_{}".format(submod, cls_name)
    cls_def = "pybind11::class_<{cls_def_argu_str}> {def_name}({submod}, \"{cls_name}\");".format(
        cls_def_argu_str=cls_def_argu_str,
        def_name=cls_def_name,
        submod=submod,
        cls_name=cls_name)
    cls_def_stmts = [cls_def]  # type: List[Union[Block, str]]
    if not has_constructor:
        cls_def_stmts.append(
            "{}.def(pybind11::init<>());".format(cls_def_name))
    for decl in decls:
        if isinstance(decl, PybindMethodDecl
                      ) and decl.method_type == MethodType.PropSetter:
            continue
        cls_def_stmts.append("{}".format(cls_def_name) + decl.to_string() +
                             ";")
    for ec in enum_classes:
        is_scoped = ec.scoped
        if is_scoped:
            ec_prefix = "pybind11::enum_<{}::{}>({}, \"{}\")".format(
                cls_name, ec.name, cls_def_name, ec.name)
        else:
            ec_prefix = "pybind11::enum_<{}::{}>({}, \"{}\", pybind11::arithmetic())".format(
                cls_name, ec.name, cls_def_name, ec.name)
        ec_values = []  # type: List[Union[Block, str]]
        cnt = 0
        for key, value in ec.items:
            stmt = ".value(\"{key}\", {class_name}::{enum_name}::{key})".format(
                key=key, class_name=cls_name, enum_name=ec.name)
            if is_scoped and cnt == len(ec.items) - 1:
                stmt += ";"
            ec_values.append(stmt)
            cnt += 1
        if not is_scoped:
            ec_values.append(".export_values();")
        cls_def_stmts.append(Block(ec_prefix, ec_values, ""))
    cls_def_block = Block("", cls_def_stmts, "")
    return cls_def_block, vblock


def _extract_anno_default(user_anno: Optional[str],
                          type_str: str,
                          cpp_default: Optional[str] = None):
    from_imports = []  # type: List[str]
    default = None
    if user_anno is None:
        if type_str in _AUTO_ANNO_TYPES:
            user_anno = _AUTO_ANNO_TYPES[type_str]
            if default is None:
                if cpp_default is not None:
                    if type_str in _AUTO_ANNO_TYPES_DEFAULT_HANDLER:
                        handler = _AUTO_ANNO_TYPES_DEFAULT_HANDLER[type_str]
                        default = handler(cpp_default)
            return user_anno, from_imports, default
    anno = None
    if user_anno is not None:
        user_anno_type_default = user_anno.split("=")
        user_anno_type = user_anno_type_default[0]
        if len(user_anno_type_default) == 2:
            default = user_anno_type_default[1]

        anno, from_imports = python_anno_parser(user_anno_type)
    if default is None:
        if cpp_default is not None:
            if type_str in _AUTO_ANNO_TYPES_DEFAULT_HANDLER:
                handler = _AUTO_ANNO_TYPES_DEFAULT_HANDLER[type_str]
                default = handler(cpp_default)

    return anno, from_imports, default


def _generate_python_interface_class(cls_name: str,
                                     decls: List[Union[PybindMethodDecl,
                                                       PybindPropDecl]],
                                     enum_classes: List[EnumClass]):
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
    imports = []  # type: List[str]
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
        anno, from_imports, default = _extract_anno_default(
            user_anno, prop_type)
        if anno is not None:
            prop_anno = anno
        imports.extend(from_imports)
        if prop_anno == cls_name:
            prop_anno = "\"{}\"".format(prop_anno)
        default_str = ""
        if default is not None:
            default_str = " = {}".format(default)
        decl_codes.append("{}: {}{}".format(prop_decl.decl.name, prop_anno,
                                            default_str))
    for decl in method_decls:
        if decl.func_name not in name_to_overloaded:
            name_to_overloaded[decl.func_name] = []
        name_to_overloaded[decl.func_name].append(decl)
    for func_name, over_decls in name_to_overloaded.items():
        if len(over_decls) == 1:
            fmt = "{}def {}({}){}: {}..."
        else:
            fmt = "@overload\n{}def {}({}){}: {}..."
        for pydecl in over_decls:
            doc = pydecl.decl.code.generate_python_doc()
            if doc:
                doc_lines = doc.split("\n")
                doc_lines = [" " * 4 + l for l in doc_lines]
                doc_lines.insert(0, "\n    \"\"\"")
                doc_lines.append("    \"\"\"\n    ")
                doc = "\n".join(doc_lines)
            res_anno = ""
            user_anno = pydecl.decl.code.ret_pyanno
            ret_type = pydecl.decl.code.return_type
            anno, from_imports, default = _extract_anno_default(
                user_anno, ret_type)
            if anno is not None:
                if anno == cls_name:
                    anno = "\"{}\"".format(anno)
                res_anno = " -> {}".format(anno)
                imports.extend(from_imports)
            decl_func_name = func_name
            if isinstance(pydecl.decl.meta, ConstructorMeta):
                decl_func_name = "__init__"

            arg_names = []  # type: List[str]
            have_default = False  # type: bool
            for arg in pydecl.decl.code.arguments:
                user_anno = arg.pyanno
                user_anno, from_imports, default = _extract_anno_default(
                    user_anno, arg.type_str, arg.default)
                default_str = ""
                if default is not None:
                    default_str = " = {}".format(default)
                    have_default = True
                else:
                    if have_default:
                        msg = ("you must provide a python default anno value "
                               "for {} of {}. format: PythonType = Default")
                        raise ValueError(msg.format(arg.name, decl_func_name))
                if user_anno is not None:
                    if user_anno == cls_name:
                        user_anno = "\"{}\"".format(user_anno)
                    imports.extend(from_imports)
                    arg_names.append("{}: {}{}".format(arg.name, user_anno,
                                                       default_str))
                else:
                    arg_names.append(arg.name)
            if not isinstance(
                    pydecl.decl.meta,
                (ExternalFunctionMeta, StaticMemberFunctionMeta)):
                arg_names.insert(0, "self")
            py_sig = ", ".join(arg_names)
            decorator = ""
            if isinstance(pydecl.decl.meta, StaticMemberFunctionMeta):
                decorator = "@staticmethod\n"
            if pydecl.method_type == MethodType.PropGetter:
                decorator = "@property\n"
                decl_func_name = pydecl.mw_meta.prop_name
            elif pydecl.method_type == MethodType.PropSetter:
                decorator = "@{}.setter\n".format(pydecl.mw_meta.prop_name)
                decl_func_name = pydecl.mw_meta.prop_name

            decl_codes.append(
                fmt.format(decorator, decl_func_name, py_sig, res_anno, doc))
    # Class EnumName:
    for ec in enum_classes:
        ec_items = []  # type: List[Union[Block, str]]
        enum_type = "EnumValue"
        if ec.scoped:
            enum_type = "EnumClassValue"

        prefix = "class {}:".format(ec.name)
        for key, value in ec.items:
            ec_items.append("{k} = {ectype}({v}) # type: {ectype}".format(
                k=key, v=value, ectype=enum_type))
        def_items = ec_items.copy()
        def_items.append("@staticmethod")
        def_items.append(
            "def __members__() -> Dict[str, {}]: ...".format(enum_type))

        decl_codes.append(Block(prefix, def_items))
        if not ec.scoped:
            decl_codes.extend(ec_items)

    class_block = Block("class {}:".format(cls_name), decl_codes)
    return class_block, imports


class Pybind11SingleClassHandler(ManualClass):
    # TODO split pybind defs to multiple file for faster compilation.
    # TODO handle inherit
    def __init__(self, cu: Class, file_suffix: str = ".cc"):
        super().__init__()
        self.add_include("pybind11/stl.h")
        self.add_include("pybind11/pybind11.h")
        self.add_include("pybind11/numpy.h")
        self.file_suffix = file_suffix
        self.built = False

        self.func_decls = []  # type: List[PybindMethodDecl]
        self.prop_decls = []  # type: List[PybindPropDecl]
        self.cu = cu

        self.bind_func_name = "bind_{}".format(cu.class_name)

    def get_pybind_decls(
            self) -> List[Union[PybindMethodDecl, PybindPropDecl]]:
        res = []  # type: List[Union[PybindMethodDecl, PybindPropDecl]]
        res.extend(self.func_decls)
        res.extend(self.prop_decls)
        return res

    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl,
                             mw_meta: Pybind11MethodMeta):
        assert cu.namespace is not None
        self.func_decls.append(
            PybindMethodDecl(func_decl, cu.namespace, cu.class_name, mw_meta))

    def handle_member(self, cu: Class, member_decl: Member,
                      mw_meta: Pybind11PropMeta):
        assert cu.namespace is not None
        self.prop_decls.append(
            PybindPropDecl(member_decl, cu.namespace, cu.class_name, mw_meta))

    def postprocess(self):
        if self.built:
            return
        bind_code = FunctionCode()
        bind_code.arg("module", "const pybind11::module_&")
        func_meta = StaticMemberFunctionMeta(impl_file_suffix=self.file_suffix)
        cls_def_block, vblock = _postprocess_class(self.cu.class_name,
                                                   self.cu.namespace, "module",
                                                   self.get_pybind_decls(),
                                                   self.cu._enum_classes)
        func_meta.name = self.bind_func_name
        if vblock is not None:
            bind_code.code_after_include = "\n".join(
                generate_code(vblock, 0, 2))
        bind_code.raw("\n".join(generate_code(cls_def_block, 0, 2)))
        func_decl = FunctionDecl(func_meta, bind_code)
        self.add_func_decl(func_decl)
        # TODO better code
        if not isinstance(self.cu, ParameterizedClass):
            self.add_impl_only_dependency_by_name(self.bind_func_name,
                                                  type(self.cu))
        else:
            self.add_impl_only_param_class_by_name(self.bind_func_name, "bind",
                                                   self.cu)
        self.built = True


class Pybind11SplitMain(ParameterizedClass):
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

    def postprocess(self, bind_cus: List[Pybind11SingleClassHandler]):
        # TODO handle inherit
        if self.built:
            return
        submodules = OrderedDict()  # type: Dict[str, str]
        sub_defs = []  # type: List[str]
        for bind_cu in bind_cus:
            origin_cu = bind_cu.cu
            ns = origin_cu.namespace
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
        create_stmts = []  # type: List[str]
        for bind_cu in bind_cus:
            assert bind_cu.namespace is not None
            bind_ns = bind_cu.namespace
            submodule_name = submodules[bind_ns]
            bind_func_name = "{}::{}::{}".format(bind_ns.replace(".", "::"),
                                                 bind_cu.class_name,
                                                 bind_cu.bind_func_name)
            create_stmts.append("{}({});".format(bind_func_name,
                                                 submodule_name))

        code_block = Block("PYBIND11_MODULE({}, m){{".format(self.module_name),
                           sub_defs + create_stmts, "}")
        code = generate_code(code_block, 0, 2)
        self.add_impl_main("{}_pybind_main".format(self.module_name),
                           "\n".join(code), self.file_suffix)
        self.built = True

    def generate_python_interface(self,
                                  bind_cus: List[Pybind11SingleClassHandler]):
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
        init_import = "from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union"
        init_pccm_import = "from pccm.stubs import EnumValue, EnumClassValue"

        ns_to_interfaces = OrderedDict()  # type: Dict[str, List[Block]]
        ns_to_imports = OrderedDict()  # type: Dict[str, List[str]]
        ns_to_interface = OrderedDict()  # type: Dict[str, str]

        for bind_cu in bind_cus:
            origin_cu = bind_cu.cu
            ns = origin_cu.namespace
            assert ns is not None
            if ns not in ns_to_interfaces:
                ns_to_interfaces[ns] = []
                ns_to_imports[ns] = []
            class_block, cls_imports = _generate_python_interface_class(
                origin_cu.class_name, bind_cu.get_pybind_decls(),
                origin_cu._enum_classes)
            ns_to_imports[ns].extend(cls_imports)
            ns_to_interfaces[ns].append(class_block)

        for k, interfaces in ns_to_interfaces.items():
            imports = ns_to_imports[k]
            imports.insert(0, init_pccm_import)
            imports.insert(0, init_import)

            imports = _unique_list_keep_order(imports)
            ns_to_interface[k] = "\n".join(
                generate_code_list(imports + interfaces, 0, 4))
        return ns_to_interface


class Pybind11SplitImpl(ManualClassGenerator):
    def __init__(self,
                 module_name: str,
                 subnamespace: str,
                 file_suffix: str = ".cc"):
        super().__init__(subnamespace)
        self.file_suffix = file_suffix
        self.module_name = module_name
        self.main_cu = Pybind11SplitMain(module_name, file_suffix)
        self.main_cu.graph_inited = True
        self.main_cu.namespace = "{}_pybind_main".format(module_name)
        self.bind_cus = []  # type: List[Pybind11SingleClassHandler]

    def create_manual_class(self, cu: Class) -> ManualClass:
        bind_cu = Pybind11SingleClassHandler(cu, self.file_suffix)
        bind_cu.class_name = "PyBind" + cu.class_name
        bind_cu.namespace = cu.namespace
        self.bind_cus.append(bind_cu)
        self.main_cu._unified_deps.append(bind_cu)
        # self.main_cu.add_param_class("bind_{}".format(bind_cu.class_name), bind_cu)
        return bind_cu

    def get_code_units(self) -> List[Class]:
        for bind in self.bind_cus:
            bind.postprocess()
        self.main_cu.postprocess(self.bind_cus)
        res = []  # type: List[Class]
        res.extend(self.bind_cus)
        res.append(self.main_cu)
        return res

    def generate_python_interface(self):
        return self.main_cu.generate_python_interface(self.bind_cus)


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


if __name__ == "__main__":
    # print(ast.parse)
    print(python_anno_parser("Tuple[Tuple[spconv.Tensor, int], float]"))
    # python_anno_parser("Tuple[Tuple[spconv.Tensor, int], float]")
