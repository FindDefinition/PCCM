import ast
import inspect
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import pccm
import portalocker
from ccimport import loader, source_iter
from pccm.builder import build_pybind
from pccm.constants import PCCM_INLINE_LIBRARY_PATH
from pccm.core import (Class, ConstructorMeta, DestructorMeta,
                       ExternalFunctionMeta, FunctionCode, FunctionDecl,
                       MemberFunctionMeta, ParameterizedClass,
                       StaticMemberFunctionMeta, get_class_meta)
from pccm.core.parsers import BaseType, CppType, DeclSpec, QualifiedId
from pccm.middlewares.pybind import Pybind11MethodMeta
from pccm.source.core import Replace, Source, execute_modifiers
from pccm.utils import UniqueNamePool, get_qualname_of_type

PCCM_INLINE_MODULE_NAME = "inline_module"
PCCM_INLINE_FUNCTION_NAME = "inline_function"
PCCM_INLINE_INNER_FUNCTION_NAME = "inline_inner_function" # usually cuda global function

PCCM_INLINE_NAMESPACE = "inline_namespace"
PCCM_INLINE_CLASS_NAME = "InlineClass"


def gcs(*instances):
    if len(instances) == 0:
        return None
    classes = [inspect.getmro(type(x)) for x in instances]
    for x in classes[0]:
        if all(x in mro for mro in classes):
            return x


def get_base_type_string(obj):
    if isinstance(obj, int):
        return "int64_t", False
    elif isinstance(obj, float):
        return "float", False
    elif isinstance(obj, bool):
        return "bool", False
    elif isinstance(obj, str):
        return "std::string", False
    else:
        return get_qualname_of_type(type(obj)), True


class MultiTypeKindError(Exception):
    pass


class InlineBuilderPlugin:
    def handle_captured_type(self, name: str, code: FunctionCode,
                             obj: Any, user_arg: Optional[Any] = None) -> Optional[Tuple[str, str]]:
        raise NotImplementedError

    def type_conversion(self, obj: Any, user_arg: Optional[Any] = None):
        return obj

    def get_cpp_type(self, obj: Any, user_arg: Optional[Any] = None) -> str:
        raise NotImplementedError


def nested_type_analysis(obj,
                         plugin_dict: Dict[str, InlineBuilderPlugin],
                         iter_limit=10,
                         user_arg: Optional[Any] = None) -> Tuple[BaseType, BaseType]:
    if isinstance(obj, (list, set)):
        types_union: Set[CppType] = set()
        mapped_type = CppType([])
        for o in obj:
            if iter_limit < 0:
                break
            iter_limit -= 1
            type_s, type_mapped_s = nested_type_analysis(
                o, plugin_dict, iter_limit, user_arg)
            cpp_type = CppType([DeclSpec(type_s)])
            mapped_type = CppType([DeclSpec(type_mapped_s)])
            types_union.add(cpp_type)
        if len(types_union) != 1:
            raise MultiTypeKindError(
                "multiple type found, union type isn't supported.")
        if isinstance(obj, list):
            name = "vector"
        else:
            name = "unordered_set"
        mapped_type = BaseType(QualifiedId(["std", name]), [mapped_type])
        return BaseType(QualifiedId(["std", name]),
                        list(types_union)), mapped_type
    elif isinstance(obj, tuple):
        type_tuple: List[CppType] = []
        mapped_type_tuple: List[CppType] = []

        for o in obj:
            if iter_limit < 0:
                break
            iter_limit -= 1
            type_s, type_mapped_s = nested_type_analysis(
                o, plugin_dict, iter_limit, user_arg)

            mapped_type_tuple.append(CppType([DeclSpec(type_mapped_s)]))
            type_tuple.append(CppType([DeclSpec(type_s)]))
        mapped_type = BaseType(QualifiedId(["std", "tuple"]),
                               mapped_type_tuple)
        return BaseType(QualifiedId(["std", "tuple"]), type_tuple), mapped_type
    elif isinstance(obj, Mapping):
        key_types_union: Set[CppType] = set()
        val_type_union: Set[CppType] = set()
        key_mapped_type = CppType([])
        val_mapped_type = CppType([])

        for k, v in obj.items():
            if iter_limit < 0:
                break
            iter_limit -= 1
            key_type_s, key_m_s = nested_type_analysis(k, plugin_dict,
                                                       iter_limit, user_arg)
            val_type_s, val_m_s = nested_type_analysis(v, plugin_dict,
                                                       iter_limit, user_arg)
            key_mapped_type = CppType([DeclSpec(key_m_s)])
            val_mapped_type = CppType([DeclSpec(val_m_s)])

            key_types_union.add(CppType([DeclSpec(key_type_s)]))
            val_type_union.add(CppType([DeclSpec(val_type_s)]))
        if len(key_types_union) != 1:
            raise MultiTypeKindError(
                "multiple type found, union type isn't supported.")
        if len(val_type_union) != 1:
            raise MultiTypeKindError(
                "multiple type found, union type isn't supported.")
        res_mapped_type = BaseType(QualifiedId(["std", "unordered_map"]),
                                   [key_mapped_type, val_mapped_type])
        res_type = BaseType(
            QualifiedId(["std", "unordered_map"]),
            [list(key_types_union)[0],
             list(val_type_union)[0]])
        return res_type, res_mapped_type
    else:
        base_str, is_custom = get_base_type_string(obj)
        res_mapped = base_str
        if is_custom:
            res_mapped = plugin_dict[base_str].get_cpp_type(obj, user_arg)
        return BaseType(QualifiedId([base_str]),
                        []), BaseType(QualifiedId([res_mapped]), [])


class NameVisitor(ast.NodeVisitor):
    def __init__(self):
        self.contain_name = False
        self.names: List[str] = []

    def visit_Name(self, node: ast.Name):
        self.names.append(node.id)


def extract_names_from_expr(expr: str):
    tree = ast.parse(expr)
    vis = NameVisitor()
    vis.visit(tree)
    return vis.names


@dataclass
class CaptureStmt:
    name: str
    is_expr: bool
    range_pair: Tuple[int, int]
    expr_names: List[str]
    replaced_name: str
    replace_range_pairs: List[Tuple[int, int]]
    arg_name: str
    def __init__(self, name: str, is_expr: bool, range_pair: Tuple[int, int],
                 replace_range_pairs: List[Tuple[int, int]]) -> None:
        self.name = name
        self.is_expr = is_expr
        self.range_pair = range_pair
        self.replace_range_pairs = replace_range_pairs
        self.expr_names = []
        if is_expr:
            self.expr_names = extract_names_from_expr(name)
        self.replaced_name = name
        self.arg_name = name



def get_save_file(path: Path):
    pid = os.getpid()


def _nested_apply_plugin_transform(obj_type: BaseType, obj,
                                   plugins: Dict[str, InlineBuilderPlugin],
                                   user_arg: Optional[Any] = None):
    if obj_type.is_std_type():
        return obj
    qualname = obj_type.qualname
    if not qualname.startswith("std"):
        # custom type
        if qualname not in plugins:
            msg = f"can't find {qualname} in plugins, available: {plugins.keys()}"
            raise ValueError(msg)
        plugin = plugins[qualname]
        return plugin.type_conversion(obj, user_arg)

    elif qualname == "std::vector":
        res = []
        vt = obj_type.args[0].base_type
        for o in obj:
            o_res = _nested_apply_plugin_transform(vt, o, plugins, user_arg)
            res.append(o_res)
        return res
    elif qualname == "std::unordered_map":
        res = {}
        kt = obj_type.args[0].base_type
        vt = obj_type.args[1].base_type
        assert kt.is_simple_type(), "only support simple type for key"
        for (ok, ov) in obj.items():
            res[ok] = _nested_apply_plugin_transform(vt, ov, plugins, user_arg)
        return res
    elif qualname == "std::unordered_set":
        assert obj_type.is_std_type()
        return obj
    elif qualname == "std::tuple":
        res = []
        for o, ot in zip(obj, obj_type.args):
            ott = ot.base_type
            if ott.is_std_type():
                res.append(o)
            else:
                res.append(_nested_apply_plugin_transform(ott, o, plugins, user_arg))
        return tuple(res)
    else:
        raise NotImplementedError


def _expr_str_to_identifier(name: str):
    res = ""
    A = ord("A")
    Z = ord("Z")
    a = ord("a")
    z = ord("z")
    _0 = ord("0")
    _9 = ord("9")

    for s in name:
        c = ord(s)
        if (c >= A and c <= Z) or (c >= a and c <= z) or (c >= _0 and c <= _9):
            res += s
        elif s != " ":
            res += "_"
    return res


class NumpyPlugin(InlineBuilderPlugin):
    def handle_captured_type(self, name: str, code: FunctionCode,
                             obj: Any, user_arg: Optional[Any] = None) ->  Optional[Tuple[str, str]]:
        return

    def type_conversion(self, obj: Any, user_arg: Optional[Any] = None):
        return obj

    def get_cpp_type(self, obj: Any, user_arg: Optional[Any] = None) -> str:
        return "pybind11::array"


@dataclass
class ModuleMetaData:
    code: str
    deps: List[str]


_DEFAULT_PLUGINS: Dict[str, InlineBuilderPlugin] = {
    "numpy.ndarray": NumpyPlugin(),
}


class PyBind11(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("pybind11/stl.h")
        self.add_include("pybind11/pybind11.h")
        self.add_include("pybind11/numpy.h")


@pccm.pybind.bind_class_module_local
class InlineClass(Class):
    def __init__(self):
        super().__init__()
        self.add_include("vector", "unordered_map", "unordered_set", "tuple",
                         "string", "iostream", "fstream")
        self.add_dependency(PyBind11)

class _ModuleMeta:
    def __init__(self, func: Any, captures: List[CaptureStmt], capture_ctypes: List[BaseType], inner_code: str) -> None:
        self.func = func
        self.captures = captures
        self.capture_ctypes = capture_ctypes
        self.inner_code = inner_code

class InlineBuilder:
    """
    inliner.inline(...)    
    """
    def __init__(
            self,
            deps: List[Type[Class]],
            plugins: Optional[Dict[str, InlineBuilderPlugin]] = None,
            root: Optional[Path] = None,
            build_root: Optional[Path] = None,
            build_kwargs: Optional[Dict[str, Any]] = None,
            param_deps: Optional[List[pccm.ParameterizedClass]] = None,
            reload_when_code_change: bool = False) -> None:
        self.deps = deps
        if param_deps is None:
            param_deps = []
        self.param_deps = param_deps
        if plugins is None:
            self.plugins = _DEFAULT_PLUGINS
        else:
            self.plugins = plugins
        if build_kwargs is None:
            build_kwargs = {}
        self.build_kwargs = build_kwargs
        self.modules: Dict[Tuple[str, str, int], _ModuleMeta] = {}
        # self.cached_captures: Dict[Tuple[str, str, int],
        #                            List[CaptureStmt]] = {}
        # self.cached_capture_ctypes: Dict[Tuple[str, str, int],
        #                                  List[BaseType]] = {}

        self.lock = threading.Lock()

        self.dep_ids = [get_qualname_of_type(t) for t in self.deps]
        self.dep_ids.sort()
        self.used_names: Set[Tuple[str, str]] = set()
        self.root = root
        self.build_root = build_root
        self._reload_when_code_change = reload_when_code_change

    def _find_exist_module_name(self, code: str, code_hash: str, root: Path):
        cur_dep_ids = self.dep_ids
        for path in root.glob(f"_{code_hash}_*.json"):
            with path.open("r") as f:
                data = json.load(f)
            meta = ModuleMetaData(data["code"], data["deps"])
            if meta.code == code and meta.deps == cur_dep_ids:
                return path.stem
        return None

    def get_save_root(self, path: Path, root: Optional[Path] = None, build_root: Optional[Path] = None):
        if root is not None:
            relative_parts = path.resolve().parent.relative_to(root)
            import_parts = list(relative_parts.parts)
        else:
            import_parts = loader.try_capture_import_parts(path)
        if import_parts is None:
            raise NotImplementedError("you must use inline "
                                    "in a standard python project with "
                                    "pip installed.")
        if build_root is not None:
            res = build_root / "/".join(import_parts)
        else:
            res = PCCM_INLINE_LIBRARY_PATH / "/".join(import_parts)
        return res


    def handle_container_code(self, code_str: str, code: FunctionCode,
                    arg: Optional[Any]):
        code.raw(code_str)

    def create_inner_decl(self, code_str: str, container_fcode: FunctionCode, inner_fcode: FunctionCode,
                    arg: Optional[Any]) -> Optional[FunctionDecl]:
        
        return None 

    def build(self, pccm_cls: pccm.Class, mod_root: Path, name: str, timeout: float, user_arg: Optional[Any] = None):
        mod_root.mkdir(mode=0o755, parents=True, exist_ok=True)
        out_lib_path = mod_root / name
        build_dir = mod_root / name
        # out_lib_meta_path = mod_root / f"{prev_mod_name}.json"
        file_lock = mod_root / f"{name}.lock"
        # -1 is invalid for portalocker
        with portalocker.Lock(str(file_lock), timeout=timeout) as fh:
            mod = build_pybind([pccm_cls],
                                out_lib_path,
                                build_dir=build_dir,
                                **self.build_kwargs)

        return getattr(
            getattr(getattr(mod, PCCM_INLINE_NAMESPACE),
                    PCCM_INLINE_CLASS_NAME), PCCM_INLINE_FUNCTION_NAME)

    def run_func(self, func, *args, user_args: Optional[Any] = None):
        return func(*args)

    def get_base_class(self):
        return InlineClass()

    def inline(self,
               name: str,
               code: Union[str, FunctionCode],
               impl_file_suffix=".cc",
               additional_vars: Optional[Dict[str, Any]] = None,
               *,
               _frame_cnt: int = 1,
               user_arg: Optional[Any] = None,
               timeout: float=999999.0,
               disable_cache: bool = False):
        """use $var to capture python objects, use $(var.shape[0]) to capture anonymous expr.
        use different to handle different arg types.
        ~20-100us run overhead. 
        only support: 
        1. int/float/str and nested containers of int/float/str.
        2. custom type via plugins
        """
        if isinstance(code, FunctionCode):
            code_str = code.inspect_body()
        else:
            code_str = code
        if additional_vars is None:
            additional_vars = {}
        cur_frame = inspect.currentframe()
        assert cur_frame is not None
        frame = cur_frame
        while _frame_cnt > 0:
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        # del frame
        local_vars = cur_frame.f_locals.copy()
        local_vars.update(additional_vars)
        code_path = cur_frame.f_code.co_filename
        lineno = cur_frame.f_lineno
        key = (code_path, name)
        unique_key = (code_path, name, lineno)
        if self._reload_when_code_change:
            unique_key = (code_path, name, -1)
        # del cur_frame
        with self.lock:
            if not disable_cache and not self._reload_when_code_change:
                exist = unique_key in self.modules
                if key in self.used_names and not exist:
                    raise ValueError("you use duplicate name in same file.", unique_key)
            else:
                exist = False
            if not exist:
                # 1. extract captured vars
                it = source_iter.CppSourceIterator(code_str)
                # hold ranges for further replace
                all_captures: List[CaptureStmt] = []
                unique_name: Dict[str, CaptureStmt] = {}
                for pose in it.get_symbol_poses("$"):
                    it.move(pose + 1)
                    next_round = it.next_round()
                    if next_round is not None:
                        cap_name = it.source[next_round[0] +
                                             1:next_round[1]].strip()
                        rep_range = (pose, next_round[1] + 1)
                        sym_range = (next_round[0] + 1, next_round[1])
                        is_expr = True
                    else:
                        iden = it.next_identifier()
                        assert iden is not None, "you can't use $ without a identifier."
                        rep_range = (pose, iden.end)
                        cap_name = iden.name.strip()
                        sym_range = (pose + 1, iden.end)
                        is_expr = False

                    if cap_name in unique_name:
                        cap = unique_name[cap_name]
                        cap.replace_range_pairs.append(rep_range)
                    else:
                        all_captures.append(
                            CaptureStmt(cap_name, is_expr, sym_range,
                                        [rep_range]))
                        unique_name[all_captures[-1].name] = all_captures[-1]
                # 2. find captures in prev frame

                container_fcode = FunctionCode()
                inner_fcode = FunctionCode()
                # 3. inference c++ types
                args = []
                name_pool = UniqueNamePool()
                replaces: List[Replace] = []
                capture_bts: List[BaseType] = []
                for cap in all_captures:
                    if not cap.is_expr:
                        if cap.name not in local_vars:
                            raise ValueError(
                                f"can't find your capture {cap.name} in prev frame."
                            )
                        obj = local_vars[cap.name]
                        arg_name = cap.name
                    else:
                        for cap_name in cap.expr_names:
                            if cap_name not in local_vars:
                                raise ValueError(
                                    f"can't find your capture {cap_name} in prev frame."
                                )
                        # eval expr in prev frame
                        obj = eval(cap.name, local_vars)
                    # apply non-anonymous vars (expr are anonymous vars)
                    try:
                        cpp_type, mapped_cpp_type = nested_type_analysis(
                            obj, self.plugins, user_arg=user_arg)
                    except:
                        print(f"ERROR: variable {cap.name} type analysis failed.")
                        raise
                    obj = _nested_apply_plugin_transform(
                        cpp_type, obj, self.plugins, user_arg)
                    if cap.is_expr:
                        cap.replaced_name = name_pool(
                            _expr_str_to_identifier(cap.replaced_name))
                    arg_name = cap.replaced_name
                    inner_cpp_type = str(mapped_cpp_type)
                    if not cap.is_expr:
                        if not cpp_type.is_std_type():
                            # custom type, must apply plugin
                            qualname = cpp_type.qualname
                            # here we only apply handle_captured_type on non-container custom type.
                            if qualname in self.plugins:
                                plugin = self.plugins[qualname]
                                res = plugin.handle_captured_type(
                                    cap.name, container_fcode, obj, user_arg)
                                if res is not None:
                                    new_arg_name, inner_cpp_type = res
                                    arg_name = new_arg_name

                    capture_bts.append(cpp_type)
                    args.append(obj)
                    cap.arg_name = arg_name
                    container_fcode.arg(arg_name, str(mapped_cpp_type))
                    inner_fcode.arg(cap.replaced_name, inner_cpp_type)
                    for rr in cap.replace_range_pairs:
                        replace = Replace(cap.replaced_name, *rr)
                        replaces.append(replace)
                for k, v in additional_vars.items():
                    _, mapped_cpp_type = nested_type_analysis(v, self.plugins, user_arg=user_arg)
                    container_fcode.arg(k, str(mapped_cpp_type))
                    args.append(v)

                inner_code_str = execute_modifiers(code_str, replaces)

                assert inner_code_str is not None
                if self._reload_when_code_change:
                    exist = unique_key in self.modules
                    if exist:
                        prev_inner_code = self.modules[unique_key].inner_code
                        if prev_inner_code == inner_code_str:
                            module_meta = self.modules[unique_key]
                            func = module_meta.func
                            all_captures = module_meta.captures
                            all_base_types = module_meta.capture_ctypes
                            args = []
                            for cap, bt in zip(all_captures, all_base_types):
                                if not cap.is_expr:
                                    if cap.name not in local_vars:
                                        raise ValueError(
                                            f"can't find your capture {cap.name} in prev frame.")
                                    obj = local_vars[cap.name]
                                else:
                                    for cap_name in cap.expr_names:
                                        if cap_name not in local_vars:
                                            raise ValueError(
                                                f"can't find your capture {cap_name} in prev frame."
                                            )
                                    # eval expr in prev frame
                                    obj = eval(cap.name, local_vars)
                                obj = _nested_apply_plugin_transform(bt, obj, self.plugins, user_arg)
                                args.append(obj)
                            for v in additional_vars.values():
                                args.append(v)
                            return self.run_func(func, *args, user_args=user_arg)
                if isinstance(code, FunctionCode):
                    container_fcode.ret(code.return_type)
                meta = self.handle_container_code(inner_code_str, container_fcode, user_arg)
                inner_decl = self.create_inner_decl(inner_code_str, container_fcode, inner_fcode, user_arg)

                # now we have complete code. we need to determine a history build dir and use it to build library if need.
                # here we must reserve build dir because we need to rebuild when dependency change.
                if meta is None:
                    meta = StaticMemberFunctionMeta(impl_file_suffix=impl_file_suffix)
                    meta.mw_metas.append(Pybind11MethodMeta())
                decl = FunctionDecl(meta, container_fcode)
                decl.meta.name = PCCM_INLINE_FUNCTION_NAME
                # container_fcode_str = decl.inspect_impl()
                # container_fcode_hash = hashlib.sha256(code_str.encode('utf-8')).hexdigest()

                pccm_class = self.get_base_class()
                pccm_class.class_name = PCCM_INLINE_CLASS_NAME
                pccm_class.add_func_decl(decl)

                for dep in decl.code._impl_only_deps:
                    pccm_class.add_impl_only_dependency_by_name(
                        decl.meta.name, dep.get_class_type())
                if isinstance(code, FunctionCode):
                    for dep in code._impl_only_deps:
                        pccm_class.add_impl_only_dependency_by_name(
                            decl.meta.name, dep.get_class_type())

                if inner_decl is not None:
                    inner_decl.meta.name = PCCM_INLINE_INNER_FUNCTION_NAME
                    pccm_class.add_func_decl(inner_decl)
                    for dep in inner_decl.code._impl_only_deps:
                        pccm_class.add_impl_only_dependency_by_name(
                            inner_decl.meta.name, dep.get_class_type())
                    if isinstance(code, FunctionCode):
                        for dep in code._impl_only_deps:
                            pccm_class.add_impl_only_dependency_by_name(
                                inner_decl.meta.name, dep.get_class_type())

                pccm_class.add_dependency(*self.deps)
                for pdep in self.param_deps:
                    pdep_ns = pdep.namespace
                    pdep_cls_name = pdep.get_user_provided_class_name()
                    assert pdep_ns is not None, "you must provide a namespace for param dep"
                    assert pdep_cls_name is not None, "you must provide a class_name for param dep"
                    # use user-defined class name as alias
                    pccm_class.add_param_class(pdep_cls_name, pdep, pdep_cls_name)
                pccm_class.namespace = PCCM_INLINE_NAMESPACE
                mod_root = self.get_save_root(Path(code_path), self.root, self.build_root)
                func = self.build(pccm_class, mod_root, name, timeout, user_arg)
                if not disable_cache:
                    module_meta = _ModuleMeta(func, all_captures, capture_bts, inner_code_str)
                    self.modules[unique_key] = module_meta
                    # self.cached_captures[unique_key] = all_captures
                    # self.cached_capture_ctypes[unique_key] = capture_bts
                    self.used_names.add(key)
                    # breakpoint()
                return self.run_func(func, *args, user_args=user_arg)

        # module already loaded. just run it after transform.
        module_meta = self.modules[unique_key]
        func = module_meta.func
        all_captures = module_meta.captures
        all_base_types = module_meta.capture_ctypes
        args = []
        for cap, bt in zip(all_captures, all_base_types):
            if not cap.is_expr:
                if cap.name not in local_vars:
                    raise ValueError(
                        f"can't find your capture {cap.name} in prev frame.")
                obj = local_vars[cap.name]
            else:
                for cap_name in cap.expr_names:
                    if cap_name not in local_vars:
                        raise ValueError(
                            f"can't find your capture {cap_name} in prev frame."
                        )
                # eval expr in prev frame
                obj = eval(cap.name, local_vars)
            obj = _nested_apply_plugin_transform(bt, obj, self.plugins, user_arg)
            args.append(obj)
        for v in additional_vars.values():
            args.append(v)
        return self.run_func(func, *args, user_args=user_arg)


def main():
    print(_expr_str_to_identifier("a.shape[0] + 5"))
    import ast

    import numpy as np
    tree = ast.parse("a.shape[0] + b + c")
    # print(ast.dump(tree))
    aa = np.array([1], dtype=np.float32)
    a = [aa, aa]
    # print(nested_type_analysis(aa.shape))
    b = InlineBuilder([])
    for i in range(10):

        code = FunctionCode().raw(f"""
        """)
        t = time.time()
        b.inline(
            "just_a_name", f"""
        // pybind::array
        float* ptr = reinterpret_cast<float*>($a[0].mutable_data());
        float* ptr2 = reinterpret_cast<float*>($a[1].mutable_data());

        ptr[0] += 1;
        ptr2[0] += 1;
        """)
        tt = time.time() - t
        print(tt)
        print(aa[0])




if __name__ == "__main__":
    main()
