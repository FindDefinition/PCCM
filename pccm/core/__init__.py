import abc
import contextlib
import difflib
import inspect
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from ccimport import compat, loader

from pccm.constants import PCCM_FUNC_META_KEY, PCCM_MAGIC_STRING
from pccm.core.buildmeta import BuildMeta, _unique_list_keep_order
from pccm.core.codegen import Block, generate_code, generate_code_list

_HEADER_ONLY_PRE_ATTRS = set(["static", "virtual"])  # type: Set[str]
_HEADER_ONLY_POST_ATTRS = set(["final", "override", "noexcept"])  # type: Set[str]


class MiddlewareMeta(object):
    def __init__(self, mw_type: Type["_MW_TYPES"]):
        self.type = mw_type


class FunctionMeta(object):
    def __init__(self,
                 inline: bool = False,
                 constexpr: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        self.name = name
        if attrs is None:
            attrs = []
        self.attrs = attrs  # type: List[str]
        # the inline attr is important because
        # it will determine function impl code location.
        # in header, or in source file.
        self.inline = inline
        self.constexpr = constexpr
        # impl file location. if empty, use default location strategy
        self.impl_loc = impl_loc
        self.impl_file_suffix = impl_file_suffix
        self.mw_metas = []  # type: List[MiddlewareMeta]

    def get_pre_attrs(self) -> List[str]:
        res = self.attrs.copy()  # type: List[str]
        if self.inline:
            res.append("inline")
        if self.constexpr:
            res.append("constexpr")
        return res

    def get_post_attrs(self) -> List[str]:
        return []

    def is_header_only(self):
        return self.inline or self.constexpr


def get_func_meta_except(func) -> FunctionMeta:
    if not hasattr(func, PCCM_FUNC_META_KEY):
        raise ValueError(
            "you need to mark method before use middleware decorator.")
    return getattr(func, PCCM_FUNC_META_KEY)


class ConstructorMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 constexpr: bool = False,
                 explicit: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
                         constexpr=constexpr,
                         attrs=attrs,
                         name=name,
                         impl_loc=impl_loc,
                         impl_file_suffix=impl_file_suffix)
        self.explicit = explicit

    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        if self.explicit:
            res.append("explicit")
        return res


class DestructorMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 constexpr: bool = False,
                 virtual: bool = True,
                 override: bool = False,
                 final: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
                         constexpr=constexpr,
                         attrs=attrs,
                         name=name,
                         impl_loc=impl_loc,
                         impl_file_suffix=impl_file_suffix)
        self.virtual = virtual
        self.override = override
        self.final = final
        if override or final:
            assert override != final

    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        if self.virtual:
            res.append("virtual")
        return res

    def get_post_attrs(self) -> List[str]:
        res = super().get_post_attrs()  # type: List[str]
        if self.override:
            res.append("override")
        if self.final:
            res.append("final")
        return res


class MemberFunctionMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 constexpr: bool = False,
                 virtual: bool = True,
                 override: bool = False,
                 final: bool = False,
                 const: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
                         constexpr=constexpr,
                         attrs=attrs,
                         name=name,
                         impl_loc=impl_loc,
                         impl_file_suffix=impl_file_suffix)
        self.virtual = virtual
        self.override = override
        self.final = final
        if override or final:
            assert override != final
        self.const = const

    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        if self.virtual:
            res.append("virtual")
        return res

    def get_post_attrs(self) -> List[str]:
        res = super().get_post_attrs()  # type: List[str]
        if self.override:
            res.append("override")
        if self.final:
            res.append("final")
        if self.const:
            res.append("const")
        return res


class StaticMemberFunctionMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 constexpr: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
                         constexpr=constexpr,
                         attrs=attrs,
                         name=name,
                         impl_loc=impl_loc,
                         impl_file_suffix=impl_file_suffix)

    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        res.append("static")
        return res


class ExternalFunctionMeta(FunctionMeta):
    """external function will be put above
    """
    def __init__(self,
                 inline: bool = False,
                 constexpr: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
                         constexpr=constexpr,
                         attrs=attrs,
                         name=name,
                         impl_loc=impl_loc,
                         impl_file_suffix=impl_file_suffix)


class Argument(object):
    def __init__(self,
                 name: str,
                 type: str,
                 default: Optional[str] = None,
                 array: Optional[str] = None,
                 pyanno: Optional[str] = None):
        self.name = name.strip()
        self.type_str = str(type).strip()  # type: str
        self.default = default
        self.array = array
        self.pyanno = pyanno
        if pyanno is not None:
            self.pyanno = pyanno.strip()
            assert len(pyanno) != 0


class Typedef(object):
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content  # type: str

    def to_string(self) -> str:
        return "using {} = {};".format(self.name, self.content)


class StaticConst(object):
    def __init__(self, name: str, type: str, value: str):
        self.name = name
        self.type = type  # type: str
        self.value = value  # type: str

    def to_string(self) -> str:
        return "static constexpr {} {} = {};".format(self.type, self.name,
                                                     self.value)


class Member(Argument):
    def __init__(self,
                 cls_type,
                 name: str,
                 type: str,
                 default: Optional[str] = None,
                 array: Optional[str] = None,
                 pyanno: Optional[str] = None,
                 mw_metas: Optional[List[MiddlewareMeta]] = None):
        super().__init__(name, type, default, array, pyanno)
        if mw_metas is None:
            mw_metas = []
        self.mw_metas = mw_metas
        self.cls_type = cls_type  # type: Optional[Type[Class]]

    def to_string(self) -> str:
        if self.array is None:
            if self.default is None:
                return "{} {};".format(self.type_str, self.name)
            else:
                return "{} {} = {};".format(self.type_str, self.name,
                                            self.default)
        else:
            if self.default is None:
                return "{} {}[{}];".format(self.type_str, self.name,
                                           self.array)
            else:
                return "{} {}[{}] = {};".format(self.type_str, self.name,
                                                self.array, self.default)


class TemplateTypeArgument(object):
    def __init__(self,
                 name: str,
                 default: Optional[str] = None,
                 template: str = "",
                 packed: bool = False):
        self.name = name
        self.template = template
        self.packed = packed
        self.default = default

    def to_string(self) -> str:
        pack = ""
        if self.packed:
            if self.default:
                raise ValueError("packed arg can't have default value")
            pack = "..."
        if self.template:
            res = "{} typename{} {}".format(self.template, pack, self.name)
        else:
            res = "typename{} {}".format(pack, self.name)
        if self.default:
            res += " = {}".format(self.default)
        return res


class TemplateNonTypeArgument(object):
    def __init__(self,
                 name: str,
                 type: str,
                 default: Optional[str] = None,
                 packed: bool = False):
        self.name = name
        self.type = type
        self.packed = packed
        self.default = default

    def to_string(self) -> str:
        pack = ""
        if self.packed:
            if self.default:
                raise ValueError("packed arg can't have default value")
            pack = "..."
        res = "{}{} {}".format(self.type, pack, self.name)
        if self.default:
            res += " = {}".format(self.default)
        return res


class FunctionCode(object):
    def __init__(self,
                 code: str,
                 arguments: Union[List[Argument], Tuple[Argument]] = (),
                 return_type: str = "void",
                 ctor_inits: Optional[List[Tuple[str, str]]] = None):
        self.arguments = list(arguments)  # type: List[Argument]
        self.return_type = return_type
        if ctor_inits is None:
            ctor_inits = []
        self.ctor_inits = ctor_inits
        self._template_arguments = [
        ]  # type: List[Union[TemplateTypeArgument, TemplateNonTypeArgument]]
        self._blocks = [Block("", [], indent=0)]  # type: List[Block]
        self.raw(code)

        self.ret_pyanno = None  # type: Optional[str]

    def is_template(self) -> bool:
        return len(self._template_arguments) > 0

    def raw(self, code: str):
        # align code indent to zero if possible
        lines = code.split("\n")
        # filter empty lines
        lines = list(filter(lambda x: len(x.strip()) > 0, lines))
        if not lines:
            return self
        min_indent = max(len(l) for l in lines)
        for l in lines:
            for i in range(len(l)):
                if l[i] != " ":
                    min_indent = min(min_indent, i)
                    break
        self._blocks[-1].body.append("\n".join(l[min_indent:] for l in lines))
        return self

    @contextlib.contextmanager
    def block(self, prefix: str):
        self._blocks.append(Block(prefix + "{", [], "}"))
        yield
        last_block = self._blocks.pop()
        self._blocks[-1].body.append(last_block)

    @contextlib.contextmanager
    def macro_block(self, prefix: str):
        self._blocks.append(Block(prefix, []))
        yield
        last_block = self._blocks.pop()
        self._blocks[-1].body.append(last_block)

    @contextlib.contextmanager
    def for_(self, for_stmt: str, prefix: str = ""):
        if prefix:
            self.raw("{}\n".format(prefix))
        with self.block("for ({})".format(for_stmt)):
            yield

    @contextlib.contextmanager
    def range_(self, var: str, stop: str, prefix: str = ""):
        with self.for_(
                "int {i} = 0; {i} < {stop}; ++{i}".format(i=var, stop=stop),
                prefix):
            yield

    @contextlib.contextmanager
    def while_(self, for_stmt: str, prefix: str = ""):
        if prefix:
            self.raw("{}\n".format(prefix))
        with self.block("while ({})".format(for_stmt)):
            yield

    @contextlib.contextmanager
    def if_(self, if_test: str, attr: str = ""):
        with self.block("if {}({})".format(attr, if_test)):
            yield

    @contextlib.contextmanager
    def else_if_(self, if_test: str):
        with self.block("else if ({})".format(if_test)):
            yield

    @contextlib.contextmanager
    def else_(self):
        with self.block("else"):
            yield

    @contextlib.contextmanager
    def macro_if_(self, if_test: str, attr: str = ""):
        with self.macro_block("#if {}({})".format(attr, if_test)):
            yield

    @contextlib.contextmanager
    def macro_else_if_(self, if_test: str):
        with self.macro_block("#elif ({})".format(if_test)):
            yield

    @contextlib.contextmanager
    def macro_else_(self):
        with self.macro_block("#else"):
            yield

    def macro_endif_(self):
        self.raw("#endif")

    def unpack(self, args: list) -> str:
        return ", ".join(map(str, args))

    def _clean_pre_attrs_impl(self, attrs: List[str]):
        res_attrs = []  # type: List[str]
        for attr in attrs:
            if attr not in _HEADER_ONLY_PRE_ATTRS:
                res_attrs.append(attr)
        return res_attrs

    def _clean_post_attrs_impl(self, attrs: List[str]):
        res_attrs = []  # type: List[str]
        for attr in attrs:
            if attr not in _HEADER_ONLY_POST_ATTRS:
                res_attrs.append(attr)
        return res_attrs

    def get_sig(self, name: str, meta: FunctionMeta) -> str:
        """
        template <args>
        pre_attrs ret_type name(args) post_attrs;
        """
        header_only = meta.is_header_only() or self.is_template()
        pre_attrs = _unique_list_keep_order(meta.get_pre_attrs())
        post_attrs = _unique_list_keep_order(meta.get_post_attrs())
        return_type = self.return_type
        if isinstance(meta, (ConstructorMeta, DestructorMeta)):
            return_type = ""
        else:
            if not header_only:
                assert self.return_type != "auto" and self.return_type != "decltype(auto)"

        fmt = "{ret_type} {name}({args}) {post_attrs};"
        template_fmt = ""
        if self.is_template():
            temp_arg_str = ", ".join(t.to_string()
                                     for t in self._template_arguments)
            template_fmt = "template <{}>\n".format(temp_arg_str)
        pre_attrs_str = " ".join(pre_attrs)
        post_attrs_str = " ".join(post_attrs)
        arg_strs = []  # type: List[str]
        for arg in self.arguments:
            arg_fmt = "{type} {name}".format(type=arg.type_str, name=arg.name)
            if arg.default:
                arg_fmt += " = {}".format(arg.default)
            arg_strs.append(arg_fmt)
        arg_str = ", ".join(arg_strs)
        prefix_fmt = fmt.format(ret_type=return_type,
                                               name=name,
                                               args=arg_str,
                                               post_attrs=post_attrs_str)
        if pre_attrs_str:
            prefix_fmt = pre_attrs_str + " " + prefix_fmt
        return template_fmt + prefix_fmt

    def get_impl(self, name: str, meta: FunctionMeta, class_name: str = ""):
        """
        template <args>
        pre_attrs ret_type BoundClass::name(args) post_attrs {body};
        """
        header_only = meta.is_header_only() or self.is_template()
        pre_attrs = _unique_list_keep_order(meta.get_pre_attrs())
        post_attrs = _unique_list_keep_order(meta.get_post_attrs())
        if not header_only:
            pre_attrs = self._clean_pre_attrs_impl(pre_attrs)
            post_attrs = self._clean_post_attrs_impl(post_attrs)
        fmt = "{ret_type} {bound}{name}({args}) {ctor_inits} {post_attrs} {{"
        template_fmt = ""

        if self.is_template():
            temp_arg_str = ", ".join(t.to_string()
                                     for t in self._template_arguments)
            template_fmt = "template <{}>\n".format(temp_arg_str)
        pre_attrs_str = " ".join(pre_attrs)
        post_attrs_str = " ".join(post_attrs)
        return_type = self.return_type
        ctor_inits = ""
        if isinstance(meta, (ConstructorMeta, DestructorMeta)):
            return_type = ""
            if self.ctor_inits:
                string = ", ".join("{}({})".format(k, v)
                                   for k, v in self.ctor_inits)
                ctor_inits = ": {}".format(string)
        bound = ""
        if class_name and not header_only:
            bound = "{}::".format(class_name)
        arg_strs = []  # type: List[str]
        for arg in self.arguments:
            arg_fmt = "{type} {name}".format(type=arg.type_str, name=arg.name)
            if arg.default and header_only:
                arg_fmt += " = {}".format(arg.default)
            arg_strs.append(arg_fmt)
        arg_str = ", ".join(arg_strs)
        prefix_fmt = fmt.format(ret_type=return_type,
                                               bound=bound,
                                               name=name,
                                               args=arg_str,
                                               ctor_inits=ctor_inits,
                                               post_attrs=post_attrs_str)
        if pre_attrs_str:
            prefix_fmt = pre_attrs_str + " " + prefix_fmt
        block = Block(template_fmt + prefix_fmt, self._blocks, "}")
        return block

    def arg(self,
            name: str,
            type: str,
            default: Optional[str] = None,
            pyanno: Optional[str] = None):
        """add a argument.
        """
        name_part = name.split(",")
        for part in name_part:
            if not part.strip():
                raise ValueError("you provide a empty name in", name)
            self.arguments.append(
                Argument(part.strip(), type, default, pyanno=pyanno))
        return self

    def targ(self,
             name: str,
             default: Optional[str] = None,
             template: str = "",
             packed: bool = False):
        """add a template type argument.
        """
        name_part = name.split(",")
        for part in name_part:
            if not part.strip():
                raise ValueError("you provide a empty name in", name)
            self._template_arguments.append(
                TemplateTypeArgument(part.strip(), default, template, packed))
        return self

    def nontype_targ(self,
                     name: str,
                     type: str,
                     default: Optional[str] = None,
                     packed: bool = False):
        """add a non-type template argument.
        """
        name_part = name.split(",")
        for part in name_part:
            if not part.strip():
                raise ValueError("you provide a empty name in", name)
            self._template_arguments.append(
                TemplateNonTypeArgument(part.strip(), type, default, packed))
        return self

    def ret(self, return_type: str, pyanno: Optional[str] = None):
        """set function return type.
        """
        self.return_type = return_type.strip()
        self.ret_pyanno = pyanno
        return self

    def ctor_init(self, name: str, value: str):
        """append a constructor initializer list
        """
        self.ctor_inits.append((name.strip(), value.strip()))
        return self


class FunctionDecl(object):
    def __init__(self, meta: FunctionMeta, code: FunctionCode):
        self.meta = meta
        self.code = code


class Class(object):
    """
    TODO split user Class and graph node.
    TODO add better virtual function check support by using python mro.
    TODO find a way to implement param class inherit.
    TODO add alias for non-param Class
    TODO add param class resume if class provide hash method
    TODO support dynamic method

    TODO handle inherited codes
    """
    def __init__(self):
        self._this_cls_type = None  # type: Optional[Type[Class]]
        self._members = []  # type: List[Member]
        if compat.Python3_7AndLater:
            self._param_class = {}  # type: Dict[str, ParameterizedClass]
            self._param_class_alias = {}  # type: Dict[str, str]
        else:
            self._param_class = OrderedDict(
            )  # type: OrderedDict[str, ParameterizedClass]
            self._param_class_alias = OrderedDict(
            )  # type: OrderedDict[str, str]

        self._typedefs = []  # type: List[Typedef]
        self._static_consts = []  # type: List[StaticConst]
        self._code_before_class = []  # type: List[str]
        self._code_after_class = []  # type: List[str]
        self._includes = []  # type: List[str]
        # TODO we can't use set here because we need to keep order of deps
        self._deps = []  # type: List[Type[Class]]
        self._impl_mains = OrderedDict(
        )  # type: Dict[str, Tuple[str, List[str]]]
        self._global_codes = []  # type: List[str]
        # filled during graph building
        self._graph_inited = False  # type: bool
        self._unified_deps = []  # type: List[Class]
        self._function_decls = []  # type: List[FunctionDecl]
        self._namespace = None  # type: Optional[str]
        if compat.Python3_7AndLater:
            self._impl_only_cls_dep = {
            }  # type: Dict[Callable, List[Type[Class]]]
            self._impl_only_param_cls_dep = {
            }  # type: Dict[Callable, List[ParameterizedClass]]
        else:
            self._impl_only_cls_dep = OrderedDict(
            )  # type: OrderedDict[Callable, List[Type[Class]]]
            self._impl_only_param_cls_dep = OrderedDict(
            )  # type: OrderedDict[Callable, List[ParameterizedClass]]

        self._build_meta = BuildMeta()

    def set_this_class_type(self, this_cls_type: Type["Class"]):
        self._this_cls_type = this_cls_type

    @property
    def build_meta(self) -> BuildMeta:
        return self._build_meta

    @property
    def class_name(self) -> str:
        return type(self).__name__

    @property
    def namespace(self) -> Optional[str]:
        return self._namespace

    @namespace.setter
    def namespace(self, val: str):
        self._namespace = val

    @property
    def graph_inited(self) -> bool:
        return self._graph_inited

    @graph_inited.setter
    def graph_inited(self, val: bool):
        self._graph_inited = val

    @property
    def uid(self) -> Optional[str]:
        if self.namespace is None:
            return None
        return "{}-{}".format(self.namespace, self.class_name)

    def add_member(self,
                   name: str,
                   type: str,
                   default: Optional[str] = None,
                   array: Optional[str] = None,
                   pyanno: Optional[str] = None,
                   mw_metas: Optional[List[MiddlewareMeta]] = None):
        name_part = name.split(",")
        for part in name_part:
            if not part.strip():
                raise ValueError("you provide a empty name in", name)
            self._members.append(
                Member(self._this_cls_type, part.strip(), type, default, array,
                       pyanno, mw_metas))

    def add_dependency(self, *no_param_class_cls: Type["Class"]):
        # TODO enable name alias for Class
        # TODO name alias must be unique
        for npcls in no_param_class_cls:
            if issubclass(npcls, ParameterizedClass):
                raise ValueError(
                    "you can't use class inherit from param class as"
                    " a dependency. use add_param_class instead.")
            self._deps.append(npcls)

    def add_param_class(self,
                        subnamespace: str,
                        param_class: "ParameterizedClass",
                        name_alias: Optional[str] = None):
        # TODO check alias is unique
        if not isinstance(param_class, ParameterizedClass):
            msg = "can only add Param Class, but your {} is Class".format(
                param_class.class_name)
            raise ValueError(msg)
        assert subnamespace not in self._param_class
        self._param_class[subnamespace] = param_class
        if name_alias is not None:
            self._param_class_alias[subnamespace] = name_alias

    def add_impl_only_dependency(self, func,
                                 *no_param_class_cls: Type["Class"]):
        if inspect.ismethod(func):
            func = func.__func__
        func_meta = get_func_meta_except(func)
        if func_meta.inline:
            raise ValueError("inline function can't have impl-only dep")
        if func not in self._impl_only_cls_dep:
            self._impl_only_cls_dep[func] = []
        for npcls in no_param_class_cls:
            self._impl_only_cls_dep[func].append(npcls)
            if npcls not in self._deps:
                # only add once
                self.add_dependency(npcls)

    def add_impl_only_param_class(self,
                                  func_or_list_funcs: Union[Callable,
                                                            List[Callable]],
                                  subnamespace: str,
                                  param_class: "ParameterizedClass",
                                  name_alias: Optional[str] = None):
        if not isinstance(func_or_list_funcs, list):
            func_or_list_funcs = [func_or_list_funcs]
        for func in func_or_list_funcs:
            if inspect.ismethod(func):
                func = func.__func__
            func_meta = get_func_meta_except(func)
            if func_meta.inline:
                raise ValueError("inline function can't have impl-only dep")
            if func not in self._impl_only_param_cls_dep:
                self._impl_only_param_cls_dep[func] = []
            self._impl_only_param_cls_dep[func].append(param_class)
        return self.add_param_class(subnamespace, param_class, name_alias)

    def add_impl_main(self,
                      impl_name: str,
                      main_code: str,
                      impl_file_suffix: str = ".cc"):
        if impl_name not in self._impl_mains:
            self._impl_mains[impl_name] = (impl_file_suffix, [])
        assert impl_file_suffix == self._impl_mains[impl_name][0]
        self._impl_mains[impl_name][1].append(main_code)

    def add_global_code(self, content: str):
        self._global_codes.append(content)

    def add_typedef(self, name: str, content: str):
        self._typedefs.append(Typedef(name, content))

    def add_static_const(self, name: str, type: str, value: str):
        self._static_consts.append(StaticConst(name, type, value))

    def add_code_before_class(self, code: str):
        """this function should only be used for macro defs.
        """
        self._code_before_class.append(code)

    def add_code_after_class(self, code: str):
        """this function should only be used for macro undefs.
        """
        self._code_after_class.append(code)

    def add_include(self, *inc_path: str):
        """can be used for empty class for external dependency.
        """
        for p in inc_path:
            self._includes.append("#include <{}>".format(p))

    @property
    def include_file(self) -> Optional[str]:
        if self.namespace is None:
            return None
        return "{}/{}.h".format("/".join(self.namespace.split(".")),
                                self.class_name)

    def get_includes_with_dep(self) -> List[str]:
        res = self._includes.copy()
        res.extend("#include <{}>".format(d.include_file)
                   for d in self.get_common_deps())
        return res

    def get_parent_class(self):  # -> Optional[Type["Class"]]
        """TODO find a better way to check invalid param class inherit
        """
        pccm_base_types = []  # List[Type[Class]]
        for base in type(self).__bases__:
            if issubclass(base, Class):
                pccm_base_types.append(base)
        assert len(pccm_base_types) == 1, "you can only inherit one class."
        pccm_base = pccm_base_types[0]
        if pccm_base is not Class and base is not ParameterizedClass:
            # assert not issubclass(mro[1], ParameterizedClass), "you can't inherit a param class."
            if not issubclass(pccm_base, ParameterizedClass):
                # you inherit a class. you must set _this_cls_type by super().__init__(__class__)
                msg = (
                    "you must use self.set_this_class_type(__class__) to init this class type"
                    " when you inherit pccm.Class")
                assert self._this_cls_type is not None, msg
                return pccm_base
        return None

    def get_class_deps(self) -> List[Type["Class"]]:
        res = list(self._deps)
        # get all dep from "add_dependency" and inherited class
        p = self.get_parent_class()
        if p is not None:
            res.append(p)
        return res

    def get_dependency_alias(self, dep: "Class") -> Optional[str]:
        """we provide some name alias inside class def for dep and param class dep.
        for Class, they are unique, so we just export their class name.
        for Param Class, one PClass may be instantiated multiple times.
        so we can't export their alias directly. user must provide a alias manually.
        """
        if not isinstance(dep, ParameterizedClass):
            # class name alias
            name_with_ns = "{}::{}".format("::".join(dep.namespace.split(".")),
                                           dep.class_name)
            ns_stmt = "using {} = {};".format(dep.class_name, name_with_ns)
            return ns_stmt
        else:
            for k, v in self._param_class.items():
                if dep is v and k in self._param_class_alias:
                    name_with_ns = "::".join(dep.namespace.split("."))
                    ns_stmt = "using {} = {}::{};".format(
                        self._param_class_alias[k], name_with_ns, v.class_name)
                    return ns_stmt
        return None

    def get_dependency_aliases(self) -> List[str]:
        assert self.graph_inited, "you must build dependency graph before generate code"
        # generate namespace alias for class
        dep_alias = []  # type: List[str]
        for dep in self._unified_deps:
            alias_stmt = self.get_dependency_alias(dep)
            if alias_stmt:
                dep_alias.append(alias_stmt)
        return dep_alias

    def get_common_dependency_aliases(self) -> List[str]:
        """return all "no impl-only" dependency aliases.
        these aliases will be put both in header and in impl.
        """
        assert self.graph_inited, "you must build dependency graph before generate code"
        # generate namespace alias for class
        dep_alias = []  # type: List[str]
        for dep in self.get_common_deps():
            alias_stmt = self.get_dependency_alias(dep)
            if alias_stmt:
                dep_alias.append(alias_stmt)
        return dep_alias

    def get_code_class_def(
            self, cu_name: str, ext_decls: List[str],
            member_func_decls: List[str]) -> "CodeSectionClassDef":
        assert self.graph_inited, "you must build dependency graph before generate code"
        # generate namespace alias for class
        dep_alias = self.get_common_dependency_aliases()
        typedef_strs = [d.to_string() for d in self._typedefs]
        sc_strs = [d.to_string() for d in self._static_consts]

        member_def_strs = [
            d.to_string() for d in self._members
            if d.cls_type is self._this_cls_type
        ]
        parent_class_alias = None  # type: Optional[str]
        parent = self.get_parent_class()
        if parent is not None:
            # TODO better way to get alias name
            parent_class_alias = parent.__name__
        cdef = CodeSectionClassDef(cu_name, dep_alias, self._code_before_class,
                                   self._code_after_class, ext_decls,
                                   typedef_strs, sc_strs, member_func_decls,
                                   member_def_strs, parent_class_alias)
        return cdef

    def get_common_deps(self) -> List["Class"]:
        assert self.graph_inited, "you must build dependency graph before generate code"
        res = []  # type: List[Class]
        for dep in self._unified_deps:
            is_impl_only = False
            if isinstance(dep, ParameterizedClass):
                for pcls_deps in self._impl_only_param_cls_dep.values():
                    for dep_candidate in pcls_deps:
                        if dep_candidate is dep:
                            is_impl_only = True
                            break
                    if is_impl_only:
                        break
            else:
                dep_type = type(dep)
                for cls_deps in self._impl_only_cls_dep.values():
                    for dep_type_candidate in cls_deps:
                        if dep_type_candidate is dep_type:
                            is_impl_only = True
                            break
                    if is_impl_only:
                        break
            if not is_impl_only:
                res.append(dep)
        return res

    def get_members(self, no_parent: bool = True):
        """this function return member functions that keep def order.
        """
        this_cls = type(self)
        if not no_parent:
            res = inspect.getmembers(this_cls, inspect.isfunction)
            # inspect.getsourcelines need to read file, so .__code__.co_firstlineno
            # is greatly faster than it.
            # res.sort(key=lambda x: inspect.getsourcelines(x[1])[1])
            res.sort(key=lambda x: x[1].__code__.co_firstlineno)
            return res
        parents = inspect.getmro(this_cls)[1:]
        parents_methods = set()
        for parent in parents:
            members = inspect.getmembers(parent, predicate=inspect.isfunction)
            parents_methods.update(members)

        child_methods = set(
            inspect.getmembers(this_cls, predicate=inspect.isfunction))
        child_only_methods = child_methods - parents_methods
        res = list(child_only_methods)
        # res.sort(key=lambda x: inspect.getsourcelines(x[1])[1])
        res.sort(key=lambda x: x[1].__code__.co_firstlineno)
        return res


class CodeSection(abc.ABC):
    # @abc.abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    def generate_namespace(self, namespace: str):
        if namespace == "":
            return [], []
        namespace_parts = namespace.split(".")
        namespace_before = []  # type: List[str]
        namespace_after = []  # type: List[str]

        for p in namespace_parts:
            namespace_before.append("namespace {} {{".format(p))
        for p in namespace_parts[::-1]:
            namespace_after.append("}} // namespace {}".format(p))
        return namespace_before, namespace_after


class CodeSectionHeader(CodeSection):
    """
    include
    namespace {
        Classes
    }
    """
    def __init__(self, namespace: str, global_codes: List[str],
                 includes: List[str], class_defs: List["CodeSectionClassDef"]):
        self.namespace = namespace
        self.includes = includes
        self.class_defs = class_defs
        self.global_codes = global_codes

    def to_string(self) -> str:
        namespace_before, namespace_after = self.generate_namespace(
            self.namespace)
        ns_before = "\n".join(namespace_before)
        ns_after = "\n".join(namespace_after)
        class_strs = [c.to_block() for c in self.class_defs]
        # class_strs = list(filter(len, class_strs))
        block = Block("\n".join(["#pragma once"] + self.includes +
                                self.global_codes + [ns_before]),
                      class_strs,
                      ns_after,
                      indent=0)
        return "\n".join(generate_code(block, 0, 2))


class CodeSectionClassDef(CodeSection):
    """
    Dependency Namespace Alias
    CodeBeforeClass
    External Functions
    ClassBody : public ParentClass {
        typedefs
        static constants
        members
        functions
    }
    CodeAfterClass
    """
    def __init__(self,
                 class_name: str,
                 dep_alias: List[str],
                 code_before: List[str],
                 code_after: List[str],
                 external_funcs: List[str],
                 typedefs: List[str],
                 static_consts: List[str],
                 functions: List[str],
                 members: List[str],
                 parent_class: Optional[str] = None):
        self.class_name = class_name
        self.dep_alias = dep_alias
        self.code_before = code_before
        self.code_after = code_after
        self.external_funcs = external_funcs
        self.typedefs = typedefs
        self.static_consts = static_consts
        self.functions = functions
        self.members = members
        self.parent_class = parent_class

    def to_block(self) -> Block:
        code_before_cls = self.dep_alias + self.code_before + generate_code_list(
            self.external_funcs, 0, 2)
        class_contents = self.typedefs + self.members + self.static_consts + self.functions
        if self.parent_class is not None:
            prefix = code_before_cls + [
                "struct {class_name} : public {parent} {{".format(
                    class_name=self.class_name, parent=self.parent_class)
            ]
        else:
            prefix = code_before_cls + [
                "struct {class_name} {{".format(class_name=self.class_name)
            ]
        block = Block("\n".join(prefix), class_contents, "};")
        return block


class CodeSectionImpl(CodeSection):
    """
    include (def)
    namespace {
        class typedefs
        impl functions[]
        impl main codes
    }
    """
    def __init__(self, namespace: str, class_typedefs: List[str],
                 includes: List[str], func_impls: List[str]):
        self.namespace = namespace
        self.includes = includes
        self.class_typedefs = class_typedefs
        self.func_impls = func_impls

    def to_string(self) -> str:
        namespace_before, namespace_after = self.generate_namespace(
            self.namespace)
        include_str = "\n".join(self.includes)
        ns_before = "\n".join(namespace_before)
        ns_after = "\n".join(namespace_after)
        block = Block("", [
            include_str, ns_before, *self.class_typedefs, *self.func_impls,
            ns_after
        ],
                      indent=0)
        return "\n".join(generate_code(block, 0, 2))


def extract_module_id_of_class(
        cu_type: Type[Class],
        root: Optional[Union[str, Path]] = None) -> Optional[str]:
    path = Path(inspect.getfile(cu_type))
    if root is not None:
        relative_path = path.relative_to(Path(root))
        import_parts = list(relative_path.parts)
        import_parts[-1] = relative_path.stem
    else:
        if loader.locate_top_package(path) is None:
            return None
        import_parts = loader.try_capture_import_parts(path, None)
    return ".".join(import_parts)


class ParameterizedClass(Class):
    """special subclass of Class. this class isn't related to c++ template,
    so it's name isn't 'TemplateClass'
    """


class ManualClass(ParameterizedClass):
    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl,
                             mw_meta: MiddlewareMeta):
        pass

    def handle_member(self, cu: Class, member_decl: Member,
                      mw_meta: MiddlewareMeta):
        pass


class ManualClassGenerator(abc.ABC):
    """generate additional Class based on existed Class.
    for example, pybind11
    """
    def __init__(self, subnamespace: str):
        super().__init__()
        self.subnamespace = subnamespace

    @abc.abstractmethod
    def create_manual_class(self, cu: Class) -> ManualClass:
        pass


class AutoClassGenerator(ParameterizedClass):
    """generate additional Class based on existed Class.
    for example, pybind11
    """
    def handle(self, cu: Class):
        pass


class ManualClassTransformer(object):
    """modify existing Class.
    """
    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl,
                             mw_meta: MiddlewareMeta):
        pass

    def handle_member(self, cu: Class, member_decl: Member,
                      mw_meta: MiddlewareMeta):
        pass


class AutoClassTransformer(object):
    """modify existing Class.
    """
    def handle(self, cu: Class):
        pass


_MW_TYPES = Union[ManualClassGenerator, AutoClassGenerator,
                  ManualClassTransformer, AutoClassTransformer]


class CodeFormatter(object):
    def __call__(self, code: str):
        return code


class CodeGenerator(object):
    def __init__(self,
                 middlewares: Optional[List[_MW_TYPES]] = None,
                 verbose: bool = False):
        if middlewares is None:
            middlewares = []
        self.middlewares = middlewares

        self.code_units = []  # type: List[Class]
        self.built = False
        self.verbose = verbose

        self._get_members_cache = {}  # type: Dict[Type[Class], Any]
        self._get_decls_cache = {}  # type: Dict[Type[Class], Any]

    def _apply_middleware_to_cus(self, uid_to_cu: Dict[str, Class]):
        # manual middlewares
        new_uid_to_cu = OrderedDict()  # type: Dict[str, Class]
        for middleware in self.middlewares:
            mw_type = type(middleware)
            if isinstance(middleware, ManualClassGenerator):
                for k, cu in uid_to_cu.items():
                    decls_with_meta = [
                    ]  # type: List[Tuple[FunctionDecl, MiddlewareMeta]]
                    members_with_meta = [
                    ]  # type: List[Tuple[Member, MiddlewareMeta]]
                    # TODO only one meta is allowed
                    for decl in cu._function_decls:
                        for mw_meta in decl.meta.mw_metas:
                            if mw_meta.type is mw_type:
                                decls_with_meta.append((decl, mw_meta))
                    for member in cu._members:
                        for mw_meta in member.mw_metas:
                            if mw_meta.type is mw_type:
                                members_with_meta.append((member, mw_meta))
                    if not decls_with_meta and not members_with_meta:
                        continue
                    new_pcls = middleware.create_manual_class(cu)
                    if new_pcls.namespace is None:
                        new_pcls.namespace = cu.namespace + "." + middleware.subnamespace
                    for decl, mw_meta in decls_with_meta:
                        new_pcls.handle_function_decl(cu, decl, mw_meta)
                    for member, mw_meta in members_with_meta:
                        new_pcls.handle_member(cu, member, mw_meta)
                    uid = new_pcls.namespace + "-" + type(new_pcls).__name__
                    new_uid_to_cu[uid] = new_pcls
            else:
                raise NotImplementedError

    def build_graph(self,
                    cus: List[Union[Class, ParameterizedClass]],
                    root: Optional[Union[str, Path]] = None):
        """code dep graph:
        1. ParameterizedClass must be leaf node.
        2. dep graph must be DAG.
        3. Class must be unique.
        """
        assert self.built is False
        for cu in cus:
            if isinstance(cu, ParameterizedClass):
                assert cu.namespace is not None
        # 1. build dependency graph

        all_cus = set()  # type: Set[Class]
        cu_type_to_cu = {}  # type: Dict[Type[Class], Class]

        for cu in cus:
            if isinstance(cu,
                          Class) and not isinstance(cu, ParameterizedClass):
                cu_type = type(cu)
                cu_type_to_cu[cu_type] = cu
                if cu.namespace is None:
                    cu.namespace = extract_module_id_of_class(cu_type, root)
        uid_to_cu = OrderedDict()  # type: Dict[str, Class]
        for cu in cus:
            stack = [(cu, set())]  # type: List[Tuple[Class, Set[Type[Class]]]]
            while stack:
                cur_cu, cur_type_trace = stack.pop()
                cur_cu_type = type(cur_cu)
                cur_ns = cur_cu.namespace
                # ns should be set below
                assert cur_ns is not None
                uid_to_cu[cur_cu.uid] = cur_cu
                cur_cu_type = type(cur_cu)
                all_cus.add(cur_cu_type)
                # construct unified dependency and assign namespace for Class
                if not cur_cu.graph_inited:
                    for dep in cur_cu.get_class_deps():
                        if dep in cur_type_trace:
                            raise ValueError("cycle detected")
                        if dep not in cu_type_to_cu:
                            cu_type_to_cu[dep] = dep()
                            cu_type_to_cu[
                                dep].namespace = extract_module_id_of_class(
                                    dep, root=root)
                        cur_cu._unified_deps.append(cu_type_to_cu[dep])
                        cur_type_trace_copy = cur_type_trace.copy()
                        cur_type_trace_copy.add(dep)
                        stack.append((cu_type_to_cu[dep], cur_type_trace_copy))

                    for k, v in cur_cu._param_class.items():
                        if type(v) in cur_type_trace:
                            raise ValueError("cycle detected")
                        if v.namespace is None:
                            v.namespace = cur_ns + "." + k
                        cur_cu._unified_deps.append(v)
                        cur_type_trace_copy = cur_type_trace.copy()
                        cur_type_trace_copy.add(type(v))
                        stack.append((v, cur_type_trace_copy))
                    cur_cu._function_decls = self.cached_extract_classunit_methods(
                        cur_cu)
                    cur_cu.graph_inited = True
                else:
                    for dep in cur_cu._unified_deps:
                        if type(dep) in cur_type_trace:
                            raise ValueError("cycle detected")
                        cur_type_trace_copy = cur_type_trace.copy()
                        cur_type_trace_copy.add(type(dep))
                        stack.append((dep, cur_type_trace_copy))
        self._apply_middleware_to_cus(uid_to_cu)
        self.code_units = uid_to_cu.values()
        self.built = True

    def cached_get_cu_members(self, cu: Class):
        cu_type = type(cu)
        if cu_type not in self._get_members_cache:
            self._get_members_cache[cu_type] = cu.get_members()
        return self._get_members_cache[cu_type]

    def cached_extract_classunit_methods(self, cu: Class):
        cu_type = type(cu)
        if cu_type not in self._get_decls_cache:
            self._get_decls_cache[cu_type] = self.extract_classunit_methods(cu)
        return self._get_decls_cache[cu_type]

    def extract_classunit_methods(self, cu: Class):
        methods = []  # type: List[FunctionDecl]
        for k, v in self.cached_get_cu_members(cu):
            if hasattr(v, PCCM_FUNC_META_KEY):
                meta = getattr(v, PCCM_FUNC_META_KEY)  # type: FunctionMeta
                code_obj = getattr(cu, k)()  # type: FunctionCode

                methods.append(FunctionDecl(meta, code_obj))
        return methods

    def get_code_units(self) -> List[Class]:
        return self.code_units

    def code_generation(self, cus: List[Union[Class, ParameterizedClass]]):
        header_dict = OrderedDict()  # type: Dict[str, CodeSectionHeader]
        impl_dict = OrderedDict()  # type: Dict[str, CodeSectionImpl]
        for cu in cus:
            cu_header_dict, cu_impls_dict = self.generate_cu_code_v2(cu)
            header_dict.update(cu_header_dict)
            impl_dict.update(cu_impls_dict)
        return header_dict, impl_dict

    def code_written(self,
                     root: Union[str, Path],
                     code_dict: Dict[str, CodeSection],
                     code_fmt: Optional[CodeFormatter] = None):
        # TODO insert md5 to code to detect manual code change
        root_path = Path(root)
        all_paths = []  # type: List[Path]
        if code_fmt is None:
            code_fmt = CodeFormatter()
        for k, v in code_dict.items():
            code_to_write = code_fmt(v.to_string())
            code_path = root_path / k
            if code_path.exists():
                # read first, if same, don't write to keep file state.
                with code_path.open("r") as f:
                    code = f.read()
                if code.strip() == code_to_write.strip():
                    all_paths.append(code_path)
                    continue
            if self.verbose and code_path.exists():
                code_to_write_lines = code_to_write.strip().split("\n")
                code_old_lines = code.strip().split("\n")
                diff = difflib.unified_diff(code_old_lines,
                                            code_to_write_lines)
                print("\n".join(diff))
            code_path.parent.mkdir(exist_ok=True, parents=True)
            with code_path.open("w") as f:
                f.write(code_to_write)
            all_paths.append(code_path)
        return all_paths

    def generate_cu_code_v2(self, cu: Class, one_impl_one_file: bool = True):
        """
        TODO multiple impl one file
        generate_code will put all Class in cus to one header file and same namespace.
        headers: {
            "xx.yy.zz": content
        }
        """
        impl_dict = OrderedDict()  # type: Dict[str, CodeSectionImpl]
        code_cdefs = []  # type: List[CodeSectionClassDef]
        cu_name = cu.class_name
        assert cu.namespace is not None, cu.class_name
        includes = []  # type: List[str]
        ext_functions_decl = []  # type: List[str]

        member_functions_index_decl = []  # type: List[Tuple[int, str]]
        static_functions_index_decl = []  # type: List[Tuple[int, str]]
        ctors_index_decl = []  # type: List[Tuple[int, str]]
        dtors_index_decl = []  # type: List[Tuple[int, str]]

        impl_dict_cls = OrderedDict()  # type: Dict[str, List[str]]
        impl_only_deps = OrderedDict()  # type: Dict[str, List[Class]]
        for index, (k, v) in enumerate(self.cached_get_cu_members(cu)):
            if hasattr(v, PCCM_FUNC_META_KEY):
                meta = getattr(v, PCCM_FUNC_META_KEY)  # type: FunctionMeta
                code_obj = getattr(cu, k)()  # type: FunctionCode
                if not isinstance(code_obj, FunctionCode):
                    msg = "your func {}-{}-{} must return a FunctionCode".format(
                        cu.namespace, cu.class_name, v.__name__)
                    raise ValueError(msg)
                func_name = meta.name
                if isinstance(meta, ConstructorMeta):
                    func_name = cu_name
                elif isinstance(meta, DestructorMeta):
                    func_name = "~" + cu_name
                if not isinstance(meta, (ConstructorMeta, DestructorMeta)):
                    assert func_name != cu_name
                # if isinstance(meta, MemberFunctionMeta):
                header_only = meta.is_header_only() or code_obj.is_template()
                impl_file_name = meta.impl_loc
                if not impl_file_name:
                    if isinstance(meta, DestructorMeta):
                        impl_file_name = "{}_destructor_{}".format(
                            cu_name, func_name)
                    else:
                        impl_file_name = "{}_{}".format(cu_name, func_name)
                impl_file_name = "{}/{}/{}{}".format(
                    cu.namespace.replace(".", "/"), cu.class_name,
                    impl_file_name, meta.impl_file_suffix)
                if impl_file_name not in impl_dict_cls:
                    impl_dict_cls[impl_file_name] = []

                func_decl_str = code_obj.get_sig(func_name, meta)
                bound_name = cu_name
                if isinstance(meta, ExternalFunctionMeta):
                    bound_name = ""
                func_impl_str = code_obj.get_impl(func_name, meta, bound_name)
                if header_only:
                    func_decl_str = func_impl_str
                if isinstance(meta, MemberFunctionMeta):
                    member_functions_index_decl.append((index, func_decl_str))
                elif isinstance(meta, ExternalFunctionMeta):
                    ext_functions_decl.append(func_decl_str)
                elif isinstance(meta, ConstructorMeta):
                    ctors_index_decl.append((index, func_decl_str))
                elif isinstance(meta, StaticMemberFunctionMeta):
                    static_functions_index_decl.append((index, func_decl_str))
                elif isinstance(meta, DestructorMeta):
                    dtors_index_decl.append((index, func_decl_str))
                else:
                    raise NotImplementedError
                if not header_only:
                    impl_dict_cls[impl_file_name].append(func_impl_str)

                # handle impl-only dependency
                # TODO better code
                if impl_file_name not in impl_only_deps:
                    impl_only_deps[impl_file_name] = []
                for impl_func, cls_deps in cu._impl_only_cls_dep.items():
                    if impl_func is v:
                        for udep in cu._unified_deps:
                            if isinstance(udep, Class) and not isinstance(
                                    udep, ParameterizedClass):
                                udep_type = type(udep)
                                for cls_dep in cls_deps:
                                    if udep_type is cls_dep:
                                        impl_only_deps[impl_file_name].append(
                                            udep)
                for impl_func, pcls_deps in cu._impl_only_param_cls_dep.items(
                ):
                    if impl_func is v:
                        for udep in cu._unified_deps:
                            if isinstance(udep, ParameterizedClass):
                                for pcls_dep in pcls_deps:
                                    if udep is pcls_dep:
                                        impl_only_deps[impl_file_name].append(
                                            udep)

        cls_funcs_with_index = (member_functions_index_decl +
                                ctors_index_decl +
                                static_functions_index_decl + dtors_index_decl)
        cls_funcs_with_index.sort(key=lambda x: x[0])
        cls_funcs = [c[1] for c in cls_funcs_with_index]
        code_cls_def = cu.get_code_class_def(cu_name, ext_functions_decl,
                                             cls_funcs)
        code_cdefs.append(code_cls_def)
        includes.extend(cu.get_includes_with_dep())
        cu_typedefs = [s.to_string() for s in cu._typedefs
                       ] + cu.get_common_dependency_aliases()
        assert len(dtors_index_decl) <= 1, "only allow one dtor"
        for k, v in impl_dict_cls.items():
            if v:
                impl_includes = ["#include <{}>".format(cu.include_file)]
                impl_only_cls_alias = []
                for dep in impl_only_deps[k]:
                    impl_includes.append("#include <{}>".format(
                        dep.include_file))
                    dep_stmt = cu.get_dependency_alias(dep)
                    if dep_stmt:
                        impl_only_cls_alias.append(dep_stmt)
                code_impl = CodeSectionImpl(cu.namespace,
                                            cu_typedefs + impl_only_cls_alias,
                                            impl_includes, v)
                impl_dict[k] = code_impl
        for k, (suffix, mains) in cu._impl_mains.items():
            impl_key = "{}/{}{}".format(cu.namespace.replace(".", "/"), k,
                                        suffix)
            code_impl = CodeSectionImpl(
                "", cu_typedefs, ["#include <{}>".format(cu.include_file)],
                mains)
            impl_dict[impl_key] = code_impl

        code_header = CodeSectionHeader(cu.namespace, cu._global_codes,
                                        includes, code_cdefs)
        header_dict = {cu.include_file: code_header}
        # every cu have only one header with several impl files.
        return header_dict, impl_dict
