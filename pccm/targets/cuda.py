from typing import List, Optional

from pccm import core


def cuda_global_function(func=None,
                         inline: bool = False,
                         attrs: Optional[List[str]] = None,
                         impl_loc: str = "",
                         impl_file_suffix: str = ".cu",
                         name=None):
    if attrs is None:
        attrs = []
    cuda_global_attrs = attrs + ["__global__"]
    return core.external_function(func,
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
    return core.member_function(func,
                                name=name,
                                inline=inline,
                                const=const,
                                virtual=False,
                                override=False,
                                final=False,
                                impl_loc=impl_loc,
                                impl_file_suffix=impl_file_suffix,
                                attrs=attrs)


def static_function(func=None,
                    host: bool = False,
                    device: bool = False,
                    inline: bool = False,
                    forceinline: bool = False,
                    attrs: Optional[List[str]] = None,
                    impl_loc: str = "",
                    impl_file_suffix: str = ".cu",
                    name=None):
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
    return core.static_function(func,
                                name=name,
                                inline=inline,
                                impl_loc=impl_loc,
                                impl_file_suffix=impl_file_suffix,
                                attrs=attrs)


def external_function(func=None,
                      host: bool = False,
                      device: bool = False,
                      inline: bool = False,
                      forceinline: bool = False,
                      attrs: Optional[List[str]] = None,
                      impl_loc: str = "",
                      impl_file_suffix: str = ".cu",
                      name=None):
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
    return core.external_function(func,
                                  name=name,
                                  inline=inline,
                                  impl_loc=impl_loc,
                                  impl_file_suffix=impl_file_suffix,
                                  attrs=attrs)


def constructor(func=None,
                host: bool = False,
                device: bool = False,
                inline: bool = False,
                forceinline: bool = False,
                attrs: Optional[List[str]] = None,
                impl_loc: str = "",
                impl_file_suffix: str = ".cu",
                name=None):
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
    return core.constructor(func,
                            name=name,
                            inline=inline,
                            impl_loc=impl_loc,
                            impl_file_suffix=impl_file_suffix,
                            attrs=attrs)
