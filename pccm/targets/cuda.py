from typing import List, Optional

from pccm.core import markers
from pccm.core import MemberFunctionMeta, ExternalFunctionMeta, StaticMemberFunctionMeta, ConstructorMeta, DestructorMeta

class CudaMemberFunctionMeta(MemberFunctionMeta):
    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        if "__forceinline__" in res and "inline" in res:
            res.remove("inline")
        return res

class CudaStaticMemberFunctionMeta(StaticMemberFunctionMeta):
    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        if "__forceinline__" in res and "inline" in res:
            res.remove("inline")
        return res

class CudaConstructorMeta(ConstructorMeta):
    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        if "__forceinline__" in res and "inline" in res:
            res.remove("inline")
        return res

class CudaDestructorMeta(DestructorMeta):
    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        if "__forceinline__" in res and "inline" in res:
            res.remove("inline")
        return res

class CudaExternalFunctionMeta(ExternalFunctionMeta):
    def get_pre_attrs(self) -> List[str]:
        res = super().get_pre_attrs()  # type: List[str]
        if "__forceinline__" in res and "inline" in res:
            res.remove("inline")
        return res


def cuda_global_function(func=None,
                         inline: bool = False,
                         attrs: Optional[List[str]] = None,
                         impl_loc: str = "",
                         impl_file_suffix: str = ".cu",
                         name=None):
    if attrs is None:
        attrs = []
    cuda_global_attrs = attrs + ["__global__"]
    return markers.external_function(func,
                                     name=name,
                                     inline=inline,
                                     impl_loc=impl_loc,
                                     impl_file_suffix=impl_file_suffix,
                                     attrs=cuda_global_attrs)


def member_function(func=None,
                    host: bool = False,
                    device: bool = False,
                    inline: bool = False,
                    forceinline: bool = False,
                    const: bool = False,
                    attrs: Optional[List[str]] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cu",
                    name=None):
    if forceinline:
        inline = True
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
                                virtual=False,
                                override=False,
                                final=False,
                                const=const,
                                impl_loc=impl_loc,
                                impl_file_suffix=impl_file_suffix,
                                attrs=attrs)
    return markers.meta_decorator(func, meta)


def static_function(func=None,
                    host: bool = False,
                    device: bool = False,
                    inline: bool = False,
                    forceinline: bool = False,
                    attrs: Optional[List[str]] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cu",
                    name=None):
    if forceinline:
        inline = True
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
    meta = CudaStaticMemberFunctionMeta(
        name=name,
        inline=inline,
        attrs=attrs,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return markers.meta_decorator(func, meta)


def external_function(func=None,
                      host: bool = False,
                      device: bool = False,
                      inline: bool = False,
                      forceinline: bool = False,
                      attrs: Optional[List[str]] = None,
                      impl_loc: str = "",
                      impl_file_suffix: str = ".cu",
                      name=None):
    if forceinline:
        inline = True
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
    meta = CudaExternalFunctionMeta(
        name=name,
        inline=inline,
        attrs=attrs,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return markers.meta_decorator(func, meta)


def constructor(func=None,
                host: bool = False,
                device: bool = False,
                inline: bool = False,
                forceinline: bool = False,
                attrs: Optional[List[str]] = None,
                impl_loc: str = "",
                impl_file_suffix: str = ".cu",
                name=None):
    if forceinline:
        inline = True
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
    meta = CudaConstructorMeta(
        inline=inline,
        attrs=attrs,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return markers.meta_decorator(func, meta)
