import contextlib
from typing import List, Optional, Tuple
import  enum 
from pccm.core import (ConstructorMeta, DestructorMeta, ExternalFunctionMeta,
                       MemberFunctionMeta, StaticMemberFunctionMeta, markers, FunctionCode, Argument)


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
                         name=None):
    if attrs is None:
        attrs = []
    cuda_global_attrs = attrs + ["__global__"]
    return markers.external_function(func,
                                     name=name,
                                     inline=inline,
                                     constexpr=False,
                                     macro_guard=macro_guard,
                                     impl_loc=impl_loc,
                                     impl_file_suffix=impl_file_suffix,
                                     attrs=cuda_global_attrs)


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
    meta = CudaStaticMemberFunctionMeta(
        name=name,
        inline=inline,
        constexpr=constexpr,
        attrs=attrs,
        macro_guard=macro_guard,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
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
    meta = CudaExternalFunctionMeta(
        name=name,
        inline=inline,
        constexpr=constexpr,
        attrs=attrs,
        macro_guard=macro_guard,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
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
    meta = CudaConstructorMeta(
        inline=inline,
        constexpr=constexpr,
        attrs=attrs,
        macro_guard=macro_guard,
        impl_loc=impl_loc,
        impl_file_suffix=impl_file_suffix,
    )
    return markers.meta_decorator(func, meta)

class CudaPointerType(enum.Enum):
    Global = "Global"
    Smem = "Smem"
    Register = "Register"

class CudaPointer:
    def __init__(self, name: str, ptr_type: CudaPointerType):
        self.ptr_type = ptr_type 
        self.name = name 

class PTXCode(FunctionCode):
    """TODO
    A = code.global_ptr("", length=...)
    B = code.smem_ptr("", length=...)
    C = code.register_ptr("", length=...)
    with code.asm_if("var_name"):
        a, b, c = code.decl_reg("a, b, c")
        code.asm_assign(A + offset_0, B + offset_1)
        code.asm_assign(a, 0)
    """
    def __init__(self,
                 code: str = "",
                 arguments: Optional[List[Argument]] = None,
                 return_type: str = "void",
                 ctor_inits: Optional[List[Tuple[str, str]]] = None):
        super().__init__(code, arguments, return_type, ctor_inits)

    @contextlib.contextmanager
    def asm_block(self):
        yield None 
