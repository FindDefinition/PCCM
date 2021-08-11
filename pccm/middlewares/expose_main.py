import ast
from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from pccm.core import (Class, CodeSectionClassDef, ConstructorMeta, EnumClass,
                       ExternalFunctionMeta, FunctionCode, FunctionDecl,
                       ManualClass, ManualClassGenerator, Member,
                       MemberFunctionMeta, MiddlewareMeta, ParameterizedClass,
                       StaticMemberFunctionMeta)
from pccm.core.markers import middleware_decorator


class ExposeMainMeta(MiddlewareMeta):
    def __init__(self):
        super().__init__(ExposeMain)


class ExposeMainHandler(ManualClass):
    def __init__(self, file_suffix: str = ".cc"):
        super().__init__()
        self.main_cu = None  # type: Optional[Class]
        self.func_decl = None  # typeL: Optional[FunctionDecl]
        self.built = False
        self.file_suffix = file_suffix

    def handle_function_decl(self, cu: Class, func_decl: FunctionDecl,
                             mw_meta: ExposeMainMeta):
        meta = func_decl.meta
        assert isinstance(
            meta, StaticMemberFunctionMeta
        ), "function expose to main must be static member function"
        assert self.main_cu is None, "you can only expose one main function"
        self.main_cu = cu
        self.func_decl = func_decl
        assert len(func_decl.code.arguments
                   ) == 0, "main function can't have any argument"

    def postprocess(self):
        if self.built:
            return
        ns = "::".join(self.main_cu.namespace.split("."))
        code = """
        {} main(){{
            return {}::{}::{}();
        }}
        """.format(self.func_decl.code.return_type, ns,
                   self.main_cu.class_name, self.func_decl.meta.name)
        self.add_impl_main("{}_main".format(self.main_cu.class_name), code,
                           self.file_suffix)
        self.built = True


class ExposeMain(ManualClassGenerator):
    def __init__(self, subnamespace: str, file_suffix: str = ".cc"):
        super().__init__(subnamespace)
        self.file_suffix = file_suffix
        self.singleton = ExposeMainHandler(file_suffix)
        self.singleton.graph_inited = True

    def create_manual_class(self, cu: Class) -> ManualClass:
        self.singleton._unified_deps.append(cu)
        return self.singleton

    def get_code_units(self) -> List[Class]:
        if self.singleton.main_cu is not None:
            self.singleton.postprocess()
            return [self.singleton]
        else:
            return []


def mark(func=None):
    meta = ExposeMainMeta()
    return middleware_decorator(func, meta)
