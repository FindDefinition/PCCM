from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from ccimport import compat

from pccm.constants import PCCM_CLASS_META_KEY, PCCM_FUNC_META_KEY
from pccm.core import (ClassMeta, ConstructorMeta, DestructorMeta,
                       ExternalFunctionMeta, FunctionMeta, MemberFunctionMeta,
                       MiddlewareMeta, StaticMemberFunctionMeta, get_class_meta)

PYTHON_OPERATORS_TO_CPP = {}


def meta_decorator(func=None, meta: Optional[FunctionMeta] = None):
    if meta is None:
        raise ValueError("this shouldn't happen")

    def wrapper(func):
        if meta.name is None:
            meta.name = func.__name__
        if meta.name in PYTHON_OPERATORS_TO_CPP:
            meta.name = PYTHON_OPERATORS_TO_CPP[meta.name]
        if hasattr(func, PCCM_FUNC_META_KEY):
            raise ValueError(
                "you can only use one meta decorator in a function.")
        setattr(func, PCCM_FUNC_META_KEY, meta)
        return func

    if func is not None:
        return wrapper(func)
    else:
        return wrapper


def class_meta_decorator(cls=None, meta: Optional[ClassMeta] = None):
    if not compat.Python3_6AndLater:
        raise NotImplementedError(
            "only python 3.6+ support class meta decorator.")
    if meta is None:
        raise ValueError("this shouldn't happen")

    def wrapper(cls):
        if meta.name is None:
            meta.name = cls.__name__
        cls_meta = get_class_meta(cls)
        if cls_meta is not None:
            cls_meta.skip_inherit = meta.skip_inherit
        else:
            setattr(cls, PCCM_CLASS_META_KEY, meta)
        return cls

    if cls is not None:
        return wrapper(cls)
    else:
        return wrapper

def append_class_mw_meta(cls=None, mw_metas: Optional[List[Any]] = None):
    if not compat.Python3_6AndLater:
        raise NotImplementedError(
            "only python 3.6+ support class meta decorator.")
    if mw_metas is None:
        raise ValueError("this shouldn't happen")

    def wrapper(cls):
        cls_meta = get_class_meta(cls)
        if cls_meta is None:
            cls_meta = ClassMeta(cls.__name__, mw_metas=mw_metas)
            setattr(cls, PCCM_CLASS_META_KEY, cls_meta)
        else:
            cls_meta.mw_metas.extend(mw_metas)
        return cls

    if cls is not None:
        return wrapper(cls)
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
        if meta is None:
            raise ValueError("this shouldn't happen")
        func_meta.mw_metas.append(meta)
        return func

    if func is not None:
        return wrapper(func)
    else:
        return wrapper


def member_function(func=None,
                    inline: bool = False,
                    constexpr: bool = False,
                    virtual: bool = False,
                    pure_virtual: bool = False,
                    override: bool = False,
                    final: bool = False,
                    const: bool = False,
                    attrs: Optional[List[str]] = None,
                    macro_guard: Optional[str] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cc",
                    header_only: Optional[bool] = None,
                    name=None):
    meta = MemberFunctionMeta(name=name,
                              inline=inline,
                              constexpr=constexpr,
                              virtual=virtual,
                              pure_virtual=pure_virtual,
                              override=override,
                              final=final,
                              const=const,
                              macro_guard=macro_guard,
                              impl_loc=impl_loc,
                              impl_file_suffix=impl_file_suffix,
                              attrs=attrs,
                              header_only=header_only)

    return meta_decorator(func, meta)


def static_function(func=None,
                    inline: bool = False,
                    constexpr: bool = False,
                    attrs: Optional[List[str]] = None,
                    macro_guard: Optional[str] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cc",
                    header_only: Optional[bool] = None,
                    name=None):
    meta = StaticMemberFunctionMeta(name=name,
                                    inline=inline,
                                    constexpr=constexpr,
                                    attrs=attrs,
                                    macro_guard=macro_guard,
                                    impl_loc=impl_loc,
                                    impl_file_suffix=impl_file_suffix,
                                    header_only=header_only)
    return meta_decorator(func, meta)


def external_function(func=None,
                      inline: bool = False,
                      constexpr: bool = False,
                      attrs: Optional[List[str]] = None,
                      macro_guard: Optional[str] = None,
                      impl_loc: str = "",
                      impl_file_suffix: str = ".cc",
                      header_only: Optional[bool] = None,
                      name=None):
    meta = ExternalFunctionMeta(name=name,
                                inline=inline,
                                constexpr=constexpr,
                                attrs=attrs,
                                macro_guard=macro_guard,
                                impl_loc=impl_loc,
                                impl_file_suffix=impl_file_suffix,
                                header_only=header_only)
    return meta_decorator(func, meta)


def constructor(func=None,
                inline: bool = False,
                constexpr: bool = False,
                attrs: Optional[List[str]] = None,
                macro_guard: Optional[str] = None,
                impl_loc: str = "",
                impl_file_suffix: str = ".cc",
                header_only: Optional[bool] = None):
    meta = ConstructorMeta(inline=inline,
                           constexpr=constexpr,
                           attrs=attrs,
                           macro_guard=macro_guard,
                           impl_loc=impl_loc,
                           impl_file_suffix=impl_file_suffix,
                           header_only=header_only)
    return meta_decorator(func, meta)


def destructor(func=None,
               inline: bool = False,
               constexpr: bool = False,
               virtual: bool = True,
               pure_virtual: bool = False,
               override: bool = False,
               final: bool = False,
               attrs: Optional[List[str]] = None,
               macro_guard: Optional[str] = None,
               impl_loc: str = "",
               impl_file_suffix: str = ".cc",
               header_only: Optional[bool] = None):
    meta = DestructorMeta(inline=inline,
                          constexpr=constexpr,
                          virtual=virtual,
                          pure_virtual=pure_virtual,
                          override=override,
                          final=final,
                          attrs=attrs,
                          macro_guard=macro_guard,
                          impl_loc=impl_loc,
                          impl_file_suffix=impl_file_suffix,
                          header_only=header_only)
    return meta_decorator(func, meta)


def skip_inherit(cls=None):
    """use this decorator when you want to add
    python base class for param class.
    """
    meta = ClassMeta(skip_inherit=True)
    return class_meta_decorator(cls, meta)
