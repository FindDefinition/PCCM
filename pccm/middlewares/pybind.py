import ast
from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from ccimport import compat

from pccm.core import (Class, CodeSectionClassDef, ConstructorMeta, EnumClass,
                       ExternalFunctionMeta, FunctionCode, FunctionDecl,
                       ManualClass, ManualClassGenerator, Member,
                       MemberFunctionMeta, MiddlewareMeta, ParameterizedClass,
                       StaticMemberFunctionMeta)
from pccm.core.buildmeta import _unique_list_keep_order
from pccm.core.codegen import Block, generate_code, generate_code_list
from pccm.core.markers import middleware_decorator

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
    "std::intptr_t": _IDENTITY_DEFAULT_HANDLER,
    "std::uintptr_t": _IDENTITY_DEFAULT_HANDLER,
    "size_t": _IDENTITY_DEFAULT_HANDLER,
    "std::size_t": _IDENTITY_DEFAULT_HANDLER,
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


def _anno_parser(node: ast.AST, imports: List[str],
                 disable_from_import: bool) -> str:
    if isinstance(node, ast.Module):
        assert len(node.body) == 1
        return _anno_parser(node.body[0], imports, disable_from_import)
    if isinstance(node, ast.Expr):
        return _anno_parser(node.value, imports, disable_from_import)
    elif isinstance(node, ast.Subscript):
        assert isinstance(node.value, (ast.Name, ast.Attribute))
        if compat.Python3_9AndLater:
            # >= 3.9 ast.Index is deprecated
            # assert isinstance(node.slice, (ast.Tuple, ast.Name))
            return "{}[{}]".format(
                _anno_parser(node.value, imports, disable_from_import),
                _anno_parser(node.slice, imports, disable_from_import))
        else:
            assert isinstance(node.slice, ast.Index)
            assert isinstance(
                node.slice.value,
                (ast.Tuple, ast.Attribute, ast.Name, ast.Subscript))
            return "{}[{}]".format(
                _anno_parser(node.value, imports, disable_from_import),
                _anno_parser(node.slice.value, imports, disable_from_import))
    elif isinstance(node, ast.Tuple):
        return ", ".join(
            _anno_parser(e, imports, disable_from_import) for e in node.elts)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        value_parts = get_attribute_name_parts(node)  # type: List[str]
        if disable_from_import:
            regular_import = "import {}".format(value_parts[0])
            imports.append(regular_import)
            return ".".join(value_parts)
        else:
            from_import = "from {} import {}".format(
                ".".join(value_parts[:-1]), value_parts[-1])
            imports.append(from_import)
            return value_parts[-1]
    else:
        msg = "pyanno only support format like name.attr[nested_name.attr[name1, name2], name3]\n"
        msg += "but we get ast node {}".format(ast.dump(node))
        raise ValueError(msg)


class TemplateTypeStmt(object):
    NameToHandler = {
        "int":
        lambda args: "int",
        "std::uintptr_t":
        lambda args: "int",
        "std::intptr_t":
        lambda args: "int",
        "std::size_t":
        lambda args: "int",
        "size_t":
        lambda args: "int",
        "int8_t":
        lambda args: "int",
        "int16_t":
        lambda args: "int",
        "int32_t":
        lambda args: "int",
        "int64_t":
        lambda args: "int",
        "uint8_t":
        lambda args: "int",
        "uint16_t":
        lambda args: "int",
        "uint32_t":
        lambda args: "int",
        "uint64_t":
        lambda args: "int",
        "unsigned":
        lambda args: "int",
        "long":
        lambda args: "int",
        "short":
        lambda args: "int",
        "float":
        lambda args: "float",
        "double":
        lambda args: "float",
        "unsigned long":
        lambda args: "int",
        "unsigned int":
        lambda args: "int",
        "unsigned long long":
        lambda args: "int",
        "bool":
        lambda args: "bool",
        "std::string":
        lambda args: "str",
        "void":
        lambda args: "None",
        "std::array":
        lambda args: "List[{}]".format(args[0].to_pyanno()),
        "tv::array":
        lambda args: "List[{}]".format(args[0].to_pyanno()),
        "std::tuple":
        lambda args: "Tuple[{}]".format(", ".join(a.to_pyanno()
                                                  for a in args)),
        "std::vector":
        lambda args: "List[{}]".format(args[0].to_pyanno()),
        "std::list":
        lambda args: "List[{}]".format(args[0].to_pyanno()),
        "std::map":
        lambda args: "Dict[{}]".format(", ".join(a.to_pyanno()
                                                 for a in args[:2])),
        "std::unordered_map":
        lambda args: "Dict[{}]".format(", ".join(a.to_pyanno()
                                                 for a in args[:2])),
        "std::set":
        lambda args: "Set[{}]".format(args[0].to_pyanno()),
        "std::unordered_set":
        lambda args: "Set[{}]".format(args[0].to_pyanno()),
    }  # type: Dict[str, Callable[[List["TemplateTypeStmt"]], str]]

    def __init__(self,
                 name: str,
                 args: List["TemplateTypeStmt"],
                 not_template: bool,
                 invalid: bool = False,
                 exist_anno: str = ""):
        self.name = name
        self.args = args
        self.not_template = not_template
        self.invalid = invalid
        self.exist_anno = exist_anno

    def to_pyanno(self) -> str:
        if self.invalid:
            return "Any"
        if self.exist_anno:
            return self.exist_anno
        if self.name in self.NameToHandler:
            pyanno_generic = self.NameToHandler[self.name](self.args)
        else:
            pyanno_generic = "Any"
        return pyanno_generic


def _simple_template_type_parser_recursive(
        stmt: str, begin: int, end: int, bracket_pair: Dict[int, int],
        exist_annos: Dict[str, str]) -> TemplateTypeStmt:
    if stmt[end - 1] != ">":
        # type with no template param
        name = stmt[begin:end].strip()
        if name in exist_annos:
            return TemplateTypeStmt("", [], True, exist_anno=exist_annos[name])
        return TemplateTypeStmt(name, [], True)
    left = stmt.find("<", begin, end)

    if left == -1:
        raise ValueError("invalid")
    name = stmt[begin:left].strip()
    if name in exist_annos:
        return TemplateTypeStmt("", [], True, exist_anno=exist_annos[name])
    template_arg_ranges = []  # type: List[Tuple[int, int]]
    pos = left + 1
    arg_range_start = left + 1
    while pos < end - 1:
        if pos in bracket_pair:
            pos = bracket_pair[pos] + 1
        if pos >= end - 1:
            break
        val = stmt[pos]
        if val == ",":
            template_arg_ranges.append((arg_range_start, pos))
            arg_range_start = pos + 1
        pos += 1
    template_arg_ranges.append((arg_range_start, end - 1))
    args = [
        _simple_template_type_parser_recursive(stmt, b, e, bracket_pair,
                                               exist_annos)
        for b, e in template_arg_ranges
    ]
    return TemplateTypeStmt(name, args, False)


def _simple_template_type_parser(
        stmt: str, exist_annos: Dict[str, str]) -> TemplateTypeStmt:
    # TODO parse const/ref
    invalid = TemplateTypeStmt("", [], False, True)
    bracket_stack = []  # type: List[Tuple[str, int]]
    N = len(stmt)
    pos = 0
    bracket_pair = {}  # type: Dict[int, int]
    while pos < N:
        val = stmt[pos]
        if val == "<":
            bracket_stack.append((val, pos))
        elif val == ">":
            if not bracket_stack:
                return invalid
            start_val, start = bracket_stack.pop()
            bracket_pair[start] = pos
        pos += 1
    if bracket_stack:
        return invalid
    if "\"" in stmt:
        res = TemplateTypeStmt("", [], False, True)
    try:
        res = _simple_template_type_parser_recursive(stmt, 0, len(stmt),
                                                     bracket_pair, exist_annos)
    except ValueError:
        res = TemplateTypeStmt("", [], False, True)
    return res


def python_anno_parser(anno_str: str):
    anno_str = anno_str.strip()
    disable_from_import = False
    if anno_str[0] == "~":
        disable_from_import = True
        anno_str = anno_str[1:]

    if anno_str == "None":
        return "None", []
    tree = ast.parse(anno_str)
    imports = []  # type: List[str]
    refined_name = _anno_parser(tree, imports, disable_from_import)
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
                 virtual: bool = False,
                 keep_alives: Optional[List[Tuple[int, int]]] = None,
                 is_raw_bind: bool = False,
                 raw_bind_anno: str = ""):
        super().__init__(Pybind11SplitImpl)
        self.bind_name = bind_name
        self.method_type = method_type
        self.prop_name = prop_name
        self.ret_policy = ret_policy
        self.call_guard = call_guard
        self.virtual = virtual
        self.keep_alives = keep_alives
        self.is_raw_bind = is_raw_bind
        self.raw_bind_anno = raw_bind_anno


class Pybind11PropMeta(Pybind11Meta):
    """may add some attributes in future.
    """
    def __init__(self, name: str, readwrite: bool = True):
        super().__init__(Pybind11SplitImpl)
        self.readwrite = readwrite
        self.name = name

class Pybind11BindMethodMeta(Pybind11Meta):
    """may add some attributes in future.
    """
    def __init__(self):
        super().__init__(Pybind11SplitImpl)

class PybindMethodDecl(object):
    def __init__(self, decl: FunctionDecl, namespace: str, class_name: str,
                 mw_meta: Pybind11MethodMeta):
        self.decl = decl
        self.mw_meta = mw_meta
        if mw_meta.is_raw_bind:
            assert isinstance(decl.meta, StaticMemberFunctionMeta), "bind method must be static"
        self.func_name = decl.get_function_name()
        self.bind_name = self.func_name
        if mw_meta.bind_name:
            self.bind_name = mw_meta.bind_name
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
            if self.args:
                return ".def({}, {})".format(self.addr, ", ".join(self.args))
            else:
                return ".def({})".format(self.addr)

        def_stmt = "def"
        bind_name = self.bind_name

        if isinstance(self.decl.meta, StaticMemberFunctionMeta):
            def_stmt = "def_static"
        if self.method_type == MethodType.PropGetter:
            def_stmt = "def_property_readonly"
            bind_name = self.mw_meta.prop_name
        addr = self.addr
        if self.decl.is_overload:
            addr = self.get_overload_addr()
        attrs = self.args.copy()
        if self.mw_meta.keep_alives is not None:
            attrs.extend("pybind11::keep_alive<{}, {}>()".format(x, y)
                         for x, y in self.mw_meta.keep_alives)
        attrs.append(self.mw_meta.ret_policy.value)
        if self.mw_meta.call_guard is not None:
            attrs.append("pybind11::call_guard<{}>()".format(
                self.mw_meta.call_guard))
        if self.setter_pybind_decl is not None:
            attrs.insert(0, self.setter_pybind_decl.addr)
            def_stmt = "def_property"
        # if self.decl.meta.mw_metas
        if attrs:
            return ".{}(\"{}\", {}, {})".format(def_stmt, bind_name, addr,
                                                ", ".join(attrs))
        else:
            return ".{}(\"{}\", {})".format(def_stmt, bind_name, addr)

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
                          parent_cls_name, self.bind_name, arg_names)


class PybindPropDecl(object):
    def __init__(self, decl: Member, namespace: str, class_name: str,
                 mw_meta: Pybind11PropMeta):
        self.decl = decl
        self.mw_meta = mw_meta
        self.addr = "&{}::{}::{}".format(namespace.replace(".", "::"),
                                         class_name, self.decl.name)

    def get_prop_name(self) -> str:
        if self.mw_meta.name:
            return self.mw_meta.name
        else:
            return self.decl.name

    def to_string(self) -> str:
        def_stmt = "def_readwrite"
        if not self.mw_meta.readwrite:
            def_stmt = "def_readonly"
        return ".{}(\"{}\", {})".format(def_stmt, self.mw_meta.name, self.addr)


class PybindClassMixin:
    def add_pybind_member(self: "Class",
                          name: str,
                          type: str,
                          default: Optional[str] = None,
                          array: Optional[str] = None,
                          pyanno: Optional[str] = None,
                          readwrite: bool = True,
                          prop_name: Optional[str] = None,
                          mw_metas: Optional[List[MiddlewareMeta]] = None):
        if mw_metas is None:
            mw_metas = []
        if prop_name is None:
            prop_name = name
        name_part = name.split(",")
        prop_name_part = prop_name.split(",")
        assert len(name_part) == len(prop_name_part)
        for name, prop_name in zip(name_part, prop_name_part):
            name = name.strip()
            prop_name = prop_name.strip()
            mw_metas_part = mw_metas.copy()
            mw_metas_part.append(Pybind11PropMeta(prop_name, readwrite))
            self.add_member(name=name,
                            type=type,
                            default=default,
                            array=array,
                            pyanno=pyanno,
                            mw_metas=mw_metas_part)
        return


def _postprocess_class(cls_name: str,
                       cls_namespace: str,
                       submod: str,
                       decls: List[Union[PybindMethodDecl, PybindPropDecl]],
                       enum_classes: List[EnumClass],
                       parent_ns: str = ""):
    has_virtual = False
    vblock = None  # Optional[Block]
    virtual_decls = []  # type: List[PybindMethodDecl]
    method_decls = []  # type: List[PybindMethodDecl]
    prop_decls = []  # type: List[PybindPropDecl]
    for decl in decls:
        if isinstance(decl, PybindMethodDecl):
            if not decl.mw_meta.is_raw_bind:
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
    parent = ""
    if parent_ns:
        parent = ", {}".format(parent_ns)
    cls_def = "pybind11::class_<{cls_def_argu_str}{parent}> {def_name}({submod}, \"{cls_name}\");".format(
        cls_def_argu_str=cls_def_argu_str,
        def_name=cls_def_name,
        submod=submod,
        cls_name=cls_name,
        parent=parent)
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


def _extract_anno_default(
    user_anno: Optional[str],
    type_str: str,
    exist_annos: Dict[str, str],
    cpp_default: Optional[str] = None,
):
    from_imports = []  # type: List[str]
    default = None
    if user_anno is None:
        try_extract_pyanno_res = _simple_template_type_parser(
            type_str, exist_annos)
        try_extract_pyanno = try_extract_pyanno_res.to_pyanno()
        if try_extract_pyanno != "Any":
            if default is None:
                if cpp_default is not None:
                    if type_str in _AUTO_ANNO_TYPES_DEFAULT_HANDLER:
                        handler = _AUTO_ANNO_TYPES_DEFAULT_HANDLER[type_str]
                        default = handler(cpp_default)
            try_extract_pyanno, from_imports = python_anno_parser(
                try_extract_pyanno)
            return try_extract_pyanno, from_imports, default
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


def _collect_exist_annos(decls: List[Union[PybindMethodDecl, PybindPropDecl]]):
    # TODO handle annos for pccm classes
    # we need to remove all annotations of pccm class

    # split decl to method decl and prop decl
    method_decls = []  # type: List[PybindMethodDecl]
    prop_decls = []  # type: List[PybindPropDecl]
    for decl in decls:
        if isinstance(decl, PybindMethodDecl):
            method_decls.append(decl)
        else:
            prop_decls.append(decl)
    exist_annos = {}  # type: Dict[str, str]
    for prop_decl in prop_decls:
        prop_type = prop_decl.decl.type_str
        user_anno = prop_decl.decl.pyanno
        if user_anno is not None:
            user_anno_pair = user_anno.split("=")
            user_anno_type = user_anno_pair[0].strip()
            exist_annos[prop_type] = user_anno_type

    for pydecl in method_decls:
        user_anno = pydecl.decl.code.ret_pyanno
        ret_type = pydecl.decl.code.return_type
        if user_anno is not None:
            user_anno_pair = user_anno.split("=")
            user_anno_type = user_anno_pair[0].strip()
            exist_annos[ret_type] = user_anno_type

        for arg in pydecl.decl.code.arguments:
            user_anno = arg.pyanno
            if user_anno is not None:
                user_anno_pair = user_anno.split("=")
                user_anno_type = user_anno_pair[0].strip()
                exist_annos[arg.type_str] = user_anno_type

    return exist_annos


def _generate_python_interface_class(cls_name: str,
                                     decls: List[Union[PybindMethodDecl,
                                                       PybindPropDecl]],
                                     enum_classes: List[EnumClass],
                                     exist_annos: Dict[str, str]):
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
            if not decl.mw_meta.is_raw_bind:
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
            user_anno, prop_type, exist_annos)
        if anno is not None:
            prop_anno = anno
        imports.extend(from_imports)
        if prop_anno == cls_name:
            prop_anno = "\"{}\"".format(prop_anno)
        default_str = ""
        if default is not None:
            default_str = " = {}".format(default)
        decl_codes.append("{}: {}{}".format(prop_decl.get_prop_name(),
                                            prop_anno, default_str))
    for decl in method_decls:
        if decl.bind_name not in name_to_overloaded:
            name_to_overloaded[decl.bind_name] = []
        name_to_overloaded[decl.bind_name].append(decl)
    for bind_name, over_decls in name_to_overloaded.items():
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
                user_anno, ret_type, exist_annos)
            if anno is not None:
                if anno == cls_name:
                    anno = "\"{}\"".format(anno)
                res_anno = " -> {}".format(anno)
                imports.extend(from_imports)
            decl_bind_name = bind_name
            if isinstance(pydecl.decl.meta, ConstructorMeta):
                decl_bind_name = "__init__"

            arg_names = []  # type: List[str]
            have_default = False  # type: bool
            for arg in pydecl.decl.code.arguments:
                user_anno = arg.pyanno
                user_anno, from_imports, default = _extract_anno_default(
                    user_anno, arg.type_str, exist_annos, arg.default)
                default_str = ""
                if default is not None:
                    default_str = " = {}".format(default)
                    have_default = True
                else:
                    if have_default:
                        msg = ("you must provide a python default anno value "
                               "for {} of {}. format: PythonType = Default")
                        raise ValueError(msg.format(arg.name, decl_bind_name))
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
                decl_bind_name = pydecl.mw_meta.prop_name
            elif pydecl.method_type == MethodType.PropSetter:
                decorator = "@{}.setter\n".format(pydecl.mw_meta.prop_name)
                decl_bind_name = pydecl.mw_meta.prop_name

            decl_codes.append(
                fmt.format(decorator, decl_bind_name, py_sig, res_anno, doc))
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
        self.raw_bind_annos: List[str] = []

    def get_pybind_decls(
            self) -> List[Union[PybindMethodDecl, PybindPropDecl]]:
        res = []  # type: List[Union[PybindMethodDecl, PybindPropDecl]]
        res.extend(self.func_decls)
        res.extend(self.prop_decls)
        return res

    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl,
                             mw_meta: Pybind11MethodMeta):
        assert cu.namespace is not None
        if mw_meta.raw_bind_anno:
            self.raw_bind_annos.append(mw_meta.raw_bind_anno)
        self.func_decls.append(
            PybindMethodDecl(func_decl, cu.namespace, cu.class_name, mw_meta))

    def handle_member(self, cu: Class, member_decl: Member,
                      mw_meta: Pybind11PropMeta):
        assert cu.namespace is not None

        self.prop_decls.append(
            PybindPropDecl(member_decl, cu.namespace, cu.class_name, mw_meta))

    def postprocess(self, parent_is_pybind: bool = False):
        if self.built:
            return
        bind_code = FunctionCode()
        bind_code.arg("module", "const pybind11::module_&")
        func_meta = StaticMemberFunctionMeta(impl_file_suffix=self.file_suffix)
        parent_name = "" if not parent_is_pybind else self.cu.get_parent_name()
        cls_def_block, vblock = _postprocess_class(self.cu.class_name,
                                                   self.cu.namespace, "module",
                                                   self.get_pybind_decls(),
                                                   self.cu._enum_classes,
                                                   parent_name)
        func_meta.name = self.bind_func_name
        if vblock is not None:
            bind_code.code_after_include = "\n".join(
                generate_code(vblock, 0, 2))
        bind_code.raw("\n".join(generate_code(cls_def_block, 0, 2)))
        for decl in self.func_decls:
            if decl.mw_meta.is_raw_bind:
                bind_code.raw("{}::{}(module);".format(self.cu.canonical_name,
                                                    decl.func_name))
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
            bind_cpp_ns = bind_ns.replace(".", "::")
            bind_func_name = "{}::{}::{}".format(bind_cpp_ns,
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
        TODO insert docstring if exists
        """
        init_import = "from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union"
        init_pccm_import = "from pccm.stubs import EnumValue, EnumClassValue"
        ns_to_interfaces = OrderedDict()  # type: Dict[str, List[Block]]
        ns_to_imports = OrderedDict()  # type: Dict[str, List[str]]
        ns_to_interface = OrderedDict()  # type: Dict[str, str]
        ns_to_raw_bind_annos = OrderedDict()  # type: Dict[str, List[str]]

        exist_annos = {}  # type: Dict[str, str]
        for bind_cu in bind_cus:
            exist_annos.update(_collect_exist_annos(
                bind_cu.get_pybind_decls()))
        for bind_cu in bind_cus:
            origin_cu = bind_cu.cu
            ns = origin_cu.namespace
            assert ns is not None
            if ns not in ns_to_interfaces:
                ns_to_interfaces[ns] = []
                ns_to_imports[ns] = []
                ns_to_raw_bind_annos[ns] = []
            class_block, cls_imports = _generate_python_interface_class(
                origin_cu.class_name, bind_cu.get_pybind_decls(),
                origin_cu._enum_classes, exist_annos)
            ns_to_imports[ns].extend(cls_imports)
            ns_to_interfaces[ns].append(class_block)
            ns_to_raw_bind_annos[ns].extend(bind_cu.raw_bind_annos)
        module_as_init = set()  # type: Set[str]
        for k, interfaces in ns_to_interfaces.items():
            k_prefix = ".".join(k.split(".")[:-1])
            if k_prefix and k_prefix in ns_to_interfaces:
                module_as_init.add(k_prefix)

        for k, interfaces in ns_to_interfaces.items():
            k_file = k
            # if this module have submodule, we need
            # to use a module directory instead a single file.
            if k in module_as_init:
                k_file += ".__init__"
            imports = ns_to_imports[k]
            imports.insert(0, init_pccm_import)
            imports.insert(0, init_import)

            imports = _unique_list_keep_order(imports)
            ns_to_interface[k_file] = "\n".join(
                generate_code_list(imports + interfaces, 0, 4))
            if k in ns_to_raw_bind_annos:
                ns_to_interface[k_file] += "\n" + "\n".join(ns_to_raw_bind_annos[k])
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
        bind_cu_ids: Set[str] = set()
        for bind in self.bind_cus:
            bind_cu_ids.add(bind.cu.canonical_name)
        for bind in self.bind_cus:
            parent_name = bind.cu.get_parent_name()
            bind.postprocess(parent_name in bind_cu_ids)
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
         nogil: bool = False,
         keep_alives: Optional[List[Tuple[int, int]]] = None,
         _is_raw_bind: bool = False,
         _raw_bind_anno: str = ""):
    if virtual:
        assert not nogil, "you can't release gil for python virtual function."
    call_guard = None  # type: Optional[str]
    if nogil:
        call_guard = "pybind11::gil_scoped_release"
    pybind_meta = Pybind11MethodMeta(bind_name, method_type, prop_name,
                                     ret_policy, call_guard, virtual,
                                     keep_alives, _is_raw_bind, _raw_bind_anno)
    return middleware_decorator(func, pybind_meta)


def mark_bind_raw(func=None, raw_bind_anno: str = ""):
    return mark(func, _is_raw_bind=True, _raw_bind_anno=raw_bind_anno)


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
    print(python_anno_parser("Dict[int, List[int]]"))
    # python_anno_parser("Tuple[Tuple[spconv.Tensor, int], float]")
    print(
        _simple_template_type_parser("std::vector<std::tuple<int, ArrayPtr>>",
                                     {
                                         "ArrayPtr": "ArrayPtr"
                                     }).to_pyanno())
