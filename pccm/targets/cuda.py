import contextlib
import enum
from typing import Dict, List, Optional, Set, Tuple, Union

from ccimport import compat

from pccm.core import (Argument, ConstructorMeta, DestructorMeta,
                       ExternalFunctionMeta, FunctionCode, MemberFunctionMeta,
                       StaticMemberFunctionMeta, markers)

if compat.Python3_6AndLater:
    from .cuda_ptx import CacheOpLd, CacheOpSt, PTXCode, PTXContext, RegDType


class CudaMemberFunctionMeta(MemberFunctionMeta):
    def is_header_only(self):
        return super().is_header_only() or "__forceinline__" in self.attrs


class CudaStaticMemberFunctionMeta(StaticMemberFunctionMeta):
    def is_header_only(self):
        return super().is_header_only() or "__forceinline__" in self.attrs


class CudaConstructorMeta(ConstructorMeta):
    def is_header_only(self):
        return super().is_header_only() or "__forceinline__" in self.attrs


class CudaDestructorMeta(DestructorMeta):
    def is_header_only(self):
        return super().is_header_only() or "__forceinline__" in self.attrs


class CudaExternalFunctionMeta(ExternalFunctionMeta):
    def is_header_only(self):
        return self.inline or "__forceinline__" in self.attrs


def cuda_global_function(func=None,
                         inline: bool = False,
                         attrs: Optional[List[str]] = None,
                         macro_guard: Optional[str] = None,
                         impl_loc: str = "",
                         impl_file_suffix: str = ".cu",
                         launch_bounds: Optional[Tuple[int, int]] = None,
                         header_only: Optional[bool] = None,
                         name=None):
    if attrs is None:
        attrs = []
    cuda_global_attrs = attrs + ["__global__"]
    if launch_bounds is not None:
        cuda_global_attrs.append("__launch_bounds__({}, {})".format(
            launch_bounds[0], launch_bounds[1]))
    return markers.external_function(func,
                                     name=name,
                                     inline=inline,
                                     constexpr=False,
                                     macro_guard=macro_guard,
                                     impl_loc=impl_loc,
                                     impl_file_suffix=impl_file_suffix,
                                     attrs=cuda_global_attrs,
                                     header_only=header_only)


def member_function(func=None,
                    host: bool = False,
                    device: bool = False,
                    inline: bool = False,
                    forceinline: bool = False,
                    constexpr: bool = False,
                    const: bool = False,
                    attrs: Optional[List[str]] = None,
                    macro_guard: Optional[str] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cu",
                    header_only: Optional[bool] = None,
                    name=None):
    if forceinline or inline:
        assert forceinline is not inline, "can't set both inline and forceinline"
    cuda_global_attrs = []
    if forceinline:
        cuda_global_attrs.append("__forceinline__")
    if host:
        cuda_global_attrs.append("__host__")
    if device:
        cuda_global_attrs.append("__device__")
    if attrs is None:
        attrs = []
    attrs.extend(cuda_global_attrs)
    meta = CudaMemberFunctionMeta(name=name,
                                  inline=inline,
                                  constexpr=constexpr,
                                  virtual=False,
                                  override=False,
                                  final=False,
                                  const=const,
                                  macro_guard=macro_guard,
                                  impl_loc=impl_loc,
                                  impl_file_suffix=impl_file_suffix,
                                  header_only=header_only,
                                  attrs=attrs)
    return markers.meta_decorator(func, meta)


def static_function(func=None,
                    host: bool = False,
                    device: bool = False,
                    inline: bool = False,
                    forceinline: bool = False,
                    constexpr: bool = False,
                    attrs: Optional[List[str]] = None,
                    macro_guard: Optional[str] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cu",
                    header_only: Optional[bool] = None,
                    name=None):
    if forceinline or inline:
        assert forceinline is not inline, "can't set both inline and forceinline"
    cuda_global_attrs = []
    if forceinline:
        cuda_global_attrs.append("__forceinline__")
    if host:
        cuda_global_attrs.append("__host__")
    if device:
        cuda_global_attrs.append("__device__")
    if attrs is None:
        attrs = []
    attrs.extend(cuda_global_attrs)
    meta = CudaStaticMemberFunctionMeta(name=name,
                                        inline=inline,
                                        constexpr=constexpr,
                                        attrs=attrs,
                                        macro_guard=macro_guard,
                                        impl_loc=impl_loc,
                                        impl_file_suffix=impl_file_suffix,
                                        header_only=header_only)
    return markers.meta_decorator(func, meta)


def external_function(func=None,
                      host: bool = False,
                      device: bool = False,
                      inline: bool = False,
                      forceinline: bool = False,
                      constexpr: bool = False,
                      attrs: Optional[List[str]] = None,
                      macro_guard: Optional[str] = None,
                      impl_loc: str = "",
                      impl_file_suffix: str = ".cu",
                      header_only: Optional[bool] = None,
                      name=None):
    if forceinline or inline:
        assert forceinline is not inline, "can't set both inline and forceinline"
    cuda_global_attrs = []
    if forceinline:
        cuda_global_attrs.append("__forceinline__")
    if host:
        cuda_global_attrs.append("__host__")
    if device:
        cuda_global_attrs.append("__device__")
    if attrs is None:
        attrs = []
    attrs.extend(cuda_global_attrs)
    meta = CudaExternalFunctionMeta(name=name,
                                    inline=inline,
                                    constexpr=constexpr,
                                    attrs=attrs,
                                    macro_guard=macro_guard,
                                    impl_loc=impl_loc,
                                    impl_file_suffix=impl_file_suffix,
                                    header_only=header_only)
    return markers.meta_decorator(func, meta)


def constructor(func=None,
                host: bool = False,
                device: bool = False,
                inline: bool = False,
                forceinline: bool = False,
                constexpr: bool = False,
                attrs: Optional[List[str]] = None,
                macro_guard: Optional[str] = None,
                impl_loc: str = "",
                impl_file_suffix: str = ".cu",
                header_only: Optional[bool] = None,
                name=None):
    if forceinline or inline:
        assert forceinline is not inline, "can't set both inline and forceinline"
    cuda_global_attrs = []
    if forceinline:
        cuda_global_attrs.append("__forceinline__")
    if host:
        cuda_global_attrs.append("__host__")
    if device:
        cuda_global_attrs.append("__device__")
    if attrs is None:
        attrs = []
    attrs.extend(cuda_global_attrs)
    meta = CudaConstructorMeta(inline=inline,
                               constexpr=constexpr,
                               attrs=attrs,
                               macro_guard=macro_guard,
                               impl_loc=impl_loc,
                               impl_file_suffix=impl_file_suffix,
                               header_only=header_only)
    return markers.meta_decorator(func, meta)
