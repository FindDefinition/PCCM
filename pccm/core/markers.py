from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pccm.constants import PCCM_FUNC_META_KEY, PCCM_MAGIC_STRING
from pccm.core import (ConstructorMeta, DestructorMeta, ExternalFunctionMeta,
                       FunctionMeta, MemberFunctionMeta, MiddlewareMeta,
                       StaticMemberFunctionMeta)

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
                              attrs=attrs)

    return meta_decorator(func, meta)


def static_function(func=None,
                    inline: bool = False,
                    constexpr: bool = False,
                    attrs: Optional[List[str]] = None,
                    macro_guard: Optional[str] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cc",
                    name=None):
    meta = StaticMemberFunctionMeta(
        name=name,
        inline=inline,
        constexpr=constexpr,
        attrs=attrs,
        macro_guard=macro_guard,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return meta_decorator(func, meta)


def external_function(func=None,
                      inline: bool = False,
                      constexpr: bool = False,
                      attrs: Optional[List[str]] = None,
                      macro_guard: Optional[str] = None,
                      impl_loc: str = "",
                      impl_file_suffix: str = ".cc",
                      name=None):
    meta = ExternalFunctionMeta(
        name=name,
        inline=inline,
        constexpr=constexpr,
        attrs=attrs,
        macro_guard=macro_guard,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return meta_decorator(func, meta)


def constructor(func=None,
                inline: bool = False,
                constexpr: bool = False,
                attrs: Optional[List[str]] = None,
                macro_guard: Optional[str] = None,
                impl_loc: str = "",
                impl_file_suffix: str = ".cc"):
    meta = ConstructorMeta(
        inline=inline,
        constexpr=constexpr,
        attrs=attrs,
        macro_guard=macro_guard,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
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
               impl_file_suffix: str = ".cc"):
    meta = DestructorMeta(
        inline=inline,
        constexpr=constexpr,
        virtual=virtual,
        pure_virtual=pure_virtual,
        override=override,
        final=final,
        attrs=attrs,
        macro_guard=macro_guard,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return meta_decorator(func, meta)
