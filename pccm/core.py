from typing import Optional, List, Tuple, Dict, Type, Union, Set, Callable
from pathlib import Path
from pccm.constants import PCCM_FUNC_META_KEY, PCCM_MAGIC_STRING
import inspect
import abc
from pccm import graph
from ccimport import loader
from pccm.codegen import Block, generate_code, generate_code_list


class MiddlewareMeta(object):
    def __init__(self, mw_type: Type["_MW_TYPES"]):
        self.type = mw_type


class FunctionMeta(object):
    def __init__(self,
                 inline: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        self.name = name
        if attrs is None:
            attrs = []
        self.attrs = attrs  # type: List[str]
        # the inline attr is important because
        # it will determine function code location.
        # in header, or in source file.
        self.inline = inline
        # impl file location. if empty, use default location strategy
        self.impl_loc = impl_loc
        self.impl_file_suffix = impl_file_suffix
        self.mw_metas = []  # type: List[MiddlewareMeta]

    def get_pre_attrs(self) -> List[str]:
        res = self.attrs.copy()  # type: List[str]
        if self.inline:
            res.append("inline")
        return res

    def get_post_attrs(self) -> List[str]:
        return []


def get_func_meta_except(func) -> FunctionMeta:
    if not hasattr(func, PCCM_FUNC_META_KEY):
        raise ValueError(
            "you need to mark method before use middleware decorator.")
    return getattr(func, PCCM_FUNC_META_KEY)


class ConstructorMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
                         attrs=attrs,
                         name=name,
                         impl_loc=impl_loc,
                         impl_file_suffix=impl_file_suffix)


class DestructorMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 virtual: bool = True,
                 override: bool = False,
                 final: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
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
                 virtual: bool = True,
                 override: bool = False,
                 final: bool = False,
                 const: bool = False,
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
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
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
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
                 attrs: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 impl_loc: str = "",
                 impl_file_suffix: str = ".cc"):
        super().__init__(inline=inline,
                         attrs=attrs,
                         name=name,
                         impl_loc=impl_loc,
                         impl_file_suffix=impl_file_suffix)


class Argument(object):
    def __init__(self, name: str, type: str, default: Optional[str] = None):
        self.name = name
        self.type_str = type  # type: str
        self.default = default


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
                 name: str,
                 type: str,
                 default: Optional[str] = None,
                 mw_metas: Optional[List[MiddlewareMeta]] = None):
        super().__init__(name, type, default)
        if mw_metas is None:
            mw_metas = []
        self.mw_metas = mw_metas

    def to_string(self) -> str:
        if self.default is None:
            return "{} {};".format(self.type_str, self.name)
        else:
            return "{} {} = {};".format(self.type_str, self.name, self.default)


class FunctionCode(object):
    def __init__(self,
                 code: str,
                 arguments: Union[List[Argument], Tuple[Argument]] = (),
                 return_type: str = "void",
                 ctor_inits: Optional[List[Tuple[str, str]]] = None):
        self.code = code
        self.arguments = list(arguments) # type: List[Argument]
        self.return_type = return_type
        if ctor_inits is None:
            ctor_inits = []
        self.ctor_inits = ctor_inits

    def get_sig(self, name: str, meta: FunctionMeta) -> str:
        """
        pre_attrs ret_type name(args) post_attrs;
        """
        inline = meta.inline
        pre_attrs = list(set(meta.get_pre_attrs()))
        post_attrs = list(set(meta.get_post_attrs()))
        return_type = self.return_type
        if isinstance(meta, (ConstructorMeta, DestructorMeta)):
            return_type = ""
        else:
            if not inline:
                assert self.return_type != "auto" and self.return_type != "decltype(auto)"

        fmt = "{ret_type} {name}({args});"
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
        return prefix_fmt

    def get_impl(self, name: str, meta: FunctionMeta, class_name: str = ""):
        """
        pre_attrs ret_type BoundClass::name(args) post_attrs {body};
        """
        inline = meta.inline
        pre_attrs = list(set(meta.get_pre_attrs()))
        post_attrs = list(set(meta.get_post_attrs()))
        fmt = "{ret_type} {bound}{name}({args}) {ctor_inits} {post_attrs} {{"
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
        if class_name and not inline:
            bound = "{}::".format(class_name)
        arg_strs = []  # type: List[str]
        for arg in self.arguments:
            arg_fmt = "{type} {name}".format(type=arg.type_str, name=arg.name)
            if arg.default and inline:
                arg_fmt += " = {}".format(arg.default)
            arg_strs.append(arg_fmt)
        arg_str = ", ".join(arg_strs)
        # detect intent
        lines = self.code.split("\n")
        # line_start_ws = []
        # for l in lines[1:]:
        #     cnt = 0
        #     for c in l:
        #         if c == " ":
        #             cnt += 1
        #         else:
        #             break
        #     line_start_ws.append(cnt)
        # if len(line_start_ws) > 1:
        #     indent = min(line_start_ws[1:])
        # else:
        #     indent = 0
        # # fix first line
        # target_indent = 2
        # if inline:
        #     target_indent = 4
        # print(indent)
        # lines[1:] = [" " * target_indent + l[indent:] for l in lines[1:]]

        prefix_fmt = fmt.format(ret_type=return_type,
                                bound=bound,
                                name=name,
                                args=arg_str,
                                ctor_inits=ctor_inits,
                                body="\n".join(lines),
                                post_attrs=post_attrs_str)
        if pre_attrs_str:
            prefix_fmt = pre_attrs_str + " " + prefix_fmt
        block = Block(prefix_fmt, lines, "}")
        return block

    def arg(self, name: str, type: str, default: Optional[str] = None):
        name_part = name.split(",")
        for part in name_part:
            self.arguments.append(Argument(part.strip(), type, default))
        return self

    def ret(self, return_type: str):
        self.return_type = return_type
        return self

class FunctionDecl(object):
    def __init__(self, meta: FunctionMeta, code: FunctionCode):
        self.meta = meta
        self.code = code


class Class(object):
    """TODO impl-only dependency for CUDA
    TODO add build meta interface to forward some dependency to builder.
    """
    def __init__(self):
        self._members = []  # type: List[Member]
        self._param_class = {}  # type: Dict[str, ParameterizedClass]
        self._typedefs = []  # type: List[Typedef]
        self._static_consts = []  # type: List[StaticConst]
        self._code_before_class = []  # type: List[str]
        self._code_after_class = []  # type: List[str]
        self._includes = []  # type: List[str]
        self._deps = []  # type: List[Type[Class]]
        self._impl_mains = {}  # type: Dict[str, Tuple[str, List[str]]]
        self._global_codes = []  # type: List[str]
        # filled during graph building
        self._graph_inited = False  # type: bool
        self._unified_deps = []  # type: List[Class]
        self._function_decls = []  # type: List[FunctionDecl]
        self._namespace = None  # type: Optional[str]

        self._impl_only_cls_dep = {}  # type: Dict[Callable, List[Type[Class]]]
        self._impl_only_param_cls_dep = {
        }  # type: Dict[Callable, List[ParameterizedClass]]

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
                   mw_metas: Optional[List[MiddlewareMeta]] = None):
        self._members.append(Member(name, type, default, mw_metas))

    def add_dependency(self, *no_param_class_cls: Type["Class"]):
        for npcls in no_param_class_cls:
            if issubclass(npcls, ParameterizedClass):
                raise ValueError(
                    "you can't use class inherit from param class as"
                    " a dependency. use add_param_class instead.")
            self._deps.append(npcls)

    def add_param_class(self, subnamespace: str,
                        param_class: "ParameterizedClass"):
        assert subnamespace not in self._param_class
        self._param_class[subnamespace] = param_class

    def add_impl_only_dependency(self, func,
                                 *no_param_class_cls: Type["Class"]):
        func_meta = get_func_meta_except(func)
        if func_meta.inline:
            raise ValueError("inline function can't have impl-only dep")
        if func not in self._impl_only_cls_dep:
            self._impl_only_cls_dep[func] = []
        for npcls in no_param_class_cls:
            assert npcls not in self._deps
            self._impl_only_cls_dep[func].append(npcls)
        return self.add_dependency(*no_param_class_cls)

    def add_impl_only_param_class(self, func, subnamespace: str,
                                  param_class: "ParameterizedClass"):
        func_meta = get_func_meta_except(func)
        if func_meta.inline:
            raise ValueError("inline function can't have impl-only dep")
        if func not in self._impl_only_cls_dep:
            self._impl_only_cls_dep[func] = []
        self._impl_only_cls_dep[func].append(param_class)
        return self.add_param_class(subnamespace, param_class)

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

    def add_include(self, inc_path: str):
        """can be used for empty class for external dependency.
        """
        self._includes.append("#include <{}>".format(inc_path))

    @property
    def include_file(self) -> Optional[str]:
        if self.namespace is None:
            return None
        return "{}/{}.h".format("/".join(self.namespace.split(".")),
                                self.class_name)

    def get_includes_with_dep(self) -> List[str]:
        res = self._includes.copy()
        res.extend("#include <{}>".format(d.include_file)
                   for d in self._unified_deps)
        return res

    def get_code_class_def(
            self, cu_name: str, ext_decls: List[str],
            member_func_decls: List[str]) -> "CodeSectionClassDef":
        assert self.graph_inited, "you must build dependency graph before generate code"
        # generate namespace alias for class
        dep_alias = []  # type: List[str]
        for dep in self._unified_deps:
            if not isinstance(dep, ParameterizedClass):
                name_with_ns = "{}::{}".format(
                    "::".join(dep.namespace.split(".")), dep.class_name)
                ns_stmt = "using {} = {};".format(dep.class_name, name_with_ns)
                dep_alias.append(ns_stmt)
        # error in msvc.
        # for k, v in self._param_class.items():
        #     p_ns = v.namespace
        #     assert p_ns is not None
        #     ns_stmt = "namespace {} = {};".format(k,
        #                                           "::".join(p_ns.split(".")))
        #     dep_alias.append(ns_stmt)

        typedef_strs = [d.to_string() for d in self._typedefs]
        sc_strs = [d.to_string() for d in self._static_consts]
        member_def_strs = [d.to_string() for d in self._members]
        cdef = CodeSectionClassDef(cu_name, dep_alias, self._code_before_class,
                                   self._code_after_class, ext_decls,
                                   typedef_strs, sc_strs, member_func_decls,
                                   member_def_strs)
        return cdef

    # def get_impl_main_dict(self):
    #     res = {}
    #     for k, v in self._impl_mains.items():
    #         impl_k = "{}.{}"


class Stmt(object):
    def __init__(self, line: str, indent: int):
        self.line = line
        self.indent = indent

    def to_string(self) -> str:
        return " " * self.indent + self.line


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
    ClassBody {
        typedefs
        static constants
        functions
        members
    }
    CodeAfterClass
    """
    def __init__(self, class_name: str, dep_alias: List[str],
                 code_before: List[str], code_after: List[str],
                 external_funcs: List[str], typedefs: List[str],
                 static_consts: List[str], functions: List[str],
                 members: List[str]):
        self.class_name = class_name
        self.dep_alias = dep_alias
        self.code_before = code_before
        self.code_after = code_after
        self.external_funcs = external_funcs
        self.typedefs = typedefs
        self.static_consts = static_consts
        self.functions = functions
        self.members = members

    def to_block(self) -> Block:
        code_before_cls = self.dep_alias + self.code_before + self.external_funcs
        class_contents = self.typedefs + self.static_consts + self.functions + self.members
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


def extract_classunit_methods(cu: Class):
    methods = []  # type: List[FunctionDecl]
    for k, v in inspect.getmembers(cu, inspect.ismethod):
        if hasattr(v, PCCM_FUNC_META_KEY):
            meta = getattr(v, PCCM_FUNC_META_KEY)  # type: FunctionMeta
            code_obj = getattr(cu, k)()  # type: FunctionCode

            methods.append(FunctionDecl(meta, code_obj))
    return methods


def generate_cu_code_v2(cu: Class, one_impl_one_file: bool = True):
    """
    TODO multiple impl one file
    generate_code will put all Class in cus to one header file and same namespace.
    headers: {
        "xx.yy.zz": content
    }
    """
    impl_dict = {}  # type: Dict[str, CodeSectionImpl]
    code_cdefs = []  # type: List[CodeSectionClassDef]
    cu_name = cu.class_name
    assert cu.namespace is not None, cu.class_name
    includes = []  # type: List[str]
    member_functions_decl = []  # List[str]
    static_functions_decl = []  # List[str]
    ext_functions_decl = []  # List[str]
    ctors_decl = []  # List[str]
    dtors_decl = []  # List[str]
    impl_dict_cls = {}  # type: Dict[str, List[str]]
    impl_only_deps = {}  # type: Dict[str, List[Class]]
    for k, v in inspect.getmembers(cu, inspect.ismethod):
        if hasattr(v, PCCM_FUNC_META_KEY):
            meta = getattr(v, PCCM_FUNC_META_KEY)  # type: FunctionMeta
            code_obj = getattr(cu, k)()  # type: FunctionCode
            func_name = meta.name
            if isinstance(meta, ConstructorMeta):
                func_name = cu_name
            elif isinstance(meta, DestructorMeta):
                func_name = "~" + cu_name
            if not isinstance(meta, (ConstructorMeta, DestructorMeta)):
                assert func_name != cu_name
            # if isinstance(meta, MemberFunctionMeta):
            inline = meta.inline
            impl_file_name = meta.impl_loc
            if not impl_file_name:
                if isinstance(meta, DestructorMeta):
                    impl_file_name = "{}_destructor_{}".format(
                        cu_name, func_name)
                else:
                    impl_file_name = "{}_{}".format(cu_name, func_name)
            impl_file_name = "{}/{}/{}{}".format(
                cu.namespace.replace(".", "/"), cu.class_name, impl_file_name,
                meta.impl_file_suffix)
            if impl_file_name not in impl_dict_cls:
                impl_dict_cls[impl_file_name] = []

            func_decl_str = code_obj.get_sig(func_name, meta)
            bound_name = cu_name
            if isinstance(meta, ExternalFunctionMeta):
                bound_name = ""
            func_impl_str = code_obj.get_impl(func_name, meta, bound_name)
            if inline:
                func_decl_str = func_impl_str
            if isinstance(meta, MemberFunctionMeta):
                member_functions_decl.append(func_decl_str)
            elif isinstance(meta, ExternalFunctionMeta):
                ext_functions_decl.append(func_decl_str)
            elif isinstance(meta, ConstructorMeta):
                ctors_decl.append(func_decl_str)
            elif isinstance(meta, StaticMemberFunctionMeta):
                static_functions_decl.append(func_decl_str)
            elif isinstance(meta, DestructorMeta):
                dtors_decl.append(func_decl_str)
            else:
                raise NotImplementedError
            if not meta.inline:
                impl_dict_cls[impl_file_name].append(func_impl_str)

            # handle impl-only dependency
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
                                    impl_only_deps[impl_file_name].append(udep)
            for impl_func, pcls_deps in cu._impl_only_param_cls_dep.items():
                if impl_func is v:
                    for udep in cu._unified_deps:
                        if isinstance(udep, ParameterizedClass):
                            for pcls_dep in pcls_deps:
                                if udep is pcls_dep:
                                    impl_only_deps[impl_file_name].append(udep)

    cls_funcs = member_functions_decl + ctors_decl + static_functions_decl + dtors_decl  # type: List[str]
    code_cls_def = cu.get_code_class_def(cu_name, ext_functions_decl,
                                         cls_funcs)
    code_cdefs.append(code_cls_def)
    includes.extend(cu.get_includes_with_dep())
    cu_typedefs = [s.to_string() for s in cu._typedefs]
    assert len(dtors_decl) <= 1, "only allow one dtor"
    for k, v in impl_dict_cls.items():
        if v:
            impl_includes = ["#include <{}>".format(cu.include_file)]
            impl_only_cls_alias = []
            for dep in impl_only_deps[k]:
                impl_includes.append("#include <{}>".format(dep.include_file))
                if not isinstance(dep, ParameterizedClass):
                    name_with_ns = "{}::{}".format(
                        "::".join(dep.namespace.split(".")), dep.class_name)
                    ns_stmt = "using {} = {};".format(dep.class_name,
                                                      name_with_ns)
                    impl_only_cls_alias.append(ns_stmt)
            code_impl = CodeSectionImpl(cu.namespace,
                                        cu_typedefs + impl_only_cls_alias,
                                        impl_includes, v)
            impl_dict[k] = code_impl
    for k, (suffix, mains) in cu._impl_mains.items():
        impl_key = "{}/{}{}".format(cu.namespace.replace(".", "/"), k, suffix)
        code_impl = CodeSectionImpl("", cu_typedefs,
                                    ["#include <{}>".format(cu.include_file)],
                                    mains)
        impl_dict[impl_key] = code_impl

    code_header = CodeSectionHeader(cu.namespace, cu._global_codes, includes,
                                    code_cdefs)
    header_dict = {cu.include_file: code_header}
    # every cu have only one header with several impl files.
    return header_dict, impl_dict


class ParameterizedClass(Class):
    """special subclass of Class. this class isn't related to c++ template,
    so it's name isn't 'TemplateClass'
    """


class ManualClass(ParameterizedClass):
    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl):
        pass

    def handle_member(self, cu: Class, member_decl: Member):
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
    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl):
        pass

    def handle_member(self, cu: Class, member_decl: Member):
        pass


class AutoClassTransformer(object):
    """modify existing Class.
    """
    def handle(self, cu: Class):
        pass


_MW_TYPES = Union[ManualClassGenerator, AutoClassGenerator,
                  ManualClassTransformer, AutoClassTransformer]


class CodeGenerator(object):
    def __init__(self, middlewares: Optional[List[_MW_TYPES]] = None):
        if middlewares is None:
            middlewares = []
        self.middlewares = middlewares

        self.code_units = []  # type: List[Class]
        self.built = False

    def _apply_middleware_to_cus(self, uid_to_cu: Dict[str, Class]):
        # manual middlewares
        new_uid_to_cu = {}  # type: Dict[str, Class]
        for middleware in self.middlewares:
            mw_type = type(middleware)
            if isinstance(middleware, ManualClassGenerator):
                for k, cu in uid_to_cu.items():
                    decls_with_meta = []  # type: List[FunctionDecl]
                    members_with_meta = []  # type: List[Member]
                    for decl in cu._function_decls:
                        for mw_meta in decl.meta.mw_metas:
                            if mw_meta.type is mw_type:
                                decls_with_meta.append(decl)
                    for member in cu._members:
                        for mw_meta in member.mw_metas:
                            if mw_meta.type is mw_type:
                                members_with_meta.append(member)
                    if not decls_with_meta and not members_with_meta:
                        continue
                    new_pcls = middleware.create_manual_class(cu)
                    if new_pcls.namespace is None:
                        new_pcls.namespace = cu.namespace + "." + middleware.subnamespace
                    for decl in decls_with_meta:
                        new_pcls.handle_function_decl(cu, decl)
                    for member in members_with_meta:
                        new_pcls.handle_member(cu, member)
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
        uid_to_cu = {}  # type: Dict[str, Class]
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
                    for dep in cur_cu._deps:
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
                    cur_cu._function_decls = extract_classunit_methods(cur_cu)
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

    def get_code_units(self) -> List[Class]:
        return self.code_units

    def code_generation(self, cus: List[Union[Class, ParameterizedClass]]):
        header_dict = {}  # type: Dict[str, CodeSectionHeader]
        impl_dict = {}  # type: Dict[str, CodeSectionImpl]
        for cu in cus:
            cu_header_dict, cu_impls_dict = generate_cu_code_v2(cu)
            header_dict.update(cu_header_dict)
            impl_dict.update(cu_impls_dict)
        return header_dict, impl_dict

    def code_written(self, root: Union[str, Path],
                     code_dict: Dict[str, CodeSection]):
        root_path = Path(root)
        all_paths = []  # type: List[Path]
        for k, v in code_dict.items():
            code_to_write = v.to_string()
            code_path = root_path / k
            if code_path.exists():
                # read first, if same, don't write to keep file state.
                with code_path.open("r") as f:
                    code = f.read()
                if code.strip() == code_to_write.strip():
                    all_paths.append(code_path)
                    continue
            code_path.parent.mkdir(exist_ok=True, parents=True)
            with code_path.open("w") as f:
                f.write(code_to_write)
            all_paths.append(code_path)
        return all_paths


def _meta_decorator(func=None, meta: Optional[FunctionMeta] = None):
    if meta is None:
        raise ValueError("this shouldn't happen")

    def wrapper(func):
        if meta.name is None:
            meta.name = func.__name__
        if hasattr(func, PCCM_FUNC_META_KEY):
            raise ValueError(
                "you can only use one meta decorator in a function.")
        setattr(func, PCCM_FUNC_META_KEY, meta)
        return func

    if func is not None:
        return wrapper(func)
    else:
        return wrapper


def middleware_decorator(func=None, meta: Optional[MiddlewareMeta] = None):
    if meta is None:
        raise ValueError("this shouldn't happen")

    def wrapper(func):
        if not hasattr(func, PCCM_FUNC_META_KEY):
            raise ValueError(
                "you need to mark method before use middleware decorator.")
        func_meta = getattr(func, PCCM_FUNC_META_KEY)  # type: FunctionMeta
        func_meta.mw_metas.append(meta)
        return func

    if func is not None:
        return wrapper(func)
    else:
        return wrapper


def member_function(func=None,
                    inline: bool = False,
                    virtual: bool = False,
                    override: bool = False,
                    final: bool = False,
                    const: bool = False,
                    attrs: Optional[List[str]] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cc",
                    name=None):
    meta = MemberFunctionMeta(name=name,
                              inline=inline,
                              virtual=virtual,
                              override=override,
                              final=final,
                              const=const,
                              impl_loc=impl_loc,
                              impl_file_suffix=impl_file_suffix,
                              attrs=attrs)
    return _meta_decorator(func, meta)


def static_function(func=None,
                    inline: bool = False,
                    attrs: Optional[List[str]] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cc",
                    name=None):
    meta = StaticMemberFunctionMeta(
        name=name,
        inline=inline,
        attrs=attrs,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return _meta_decorator(func, meta)


def external_function(func=None,
                      inline: bool = False,
                      attrs: Optional[List[str]] = None,
                      impl_loc: str = "",
                      impl_file_suffix: str = ".cc",
                      name=None):
    meta = ExternalFunctionMeta(
        name=name,
        inline=inline,
        attrs=attrs,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return _meta_decorator(func, meta)


def constructor(func=None,
                inline: bool = False,
                attrs: Optional[List[str]] = None,
                impl_loc: str = "",
                impl_file_suffix: str = ".cc"):
    meta = ConstructorMeta(
        inline=inline,
        attrs=attrs,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return _meta_decorator(func, meta)


def destructor(func=None,
               inline: bool = False,
               virtual: bool = True,
               override: bool = False,
               final: bool = False,
               attrs: Optional[List[str]] = None,
               impl_loc: str = "",
               impl_file_suffix: str = ".cc"):
    meta = DestructorMeta(
        inline=inline,
        virtual=virtual,
        override=override,
        final=final,
        attrs=attrs,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return _meta_decorator(func, meta)
