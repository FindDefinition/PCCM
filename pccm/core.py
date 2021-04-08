from typing import Optional, List, Dict
from pccm.constants import PCCM_FUNC_META_KEY


class FunctionMeta(object):
    def __init__(self,
                 inline: bool = False,
                 attrs: Optional[List[str]] = None):
        if attrs is None:
            attrs = []
        self.attrs = attrs  # type: List[str]
        # the inline attr is important because
        # it will determine function code location.
        # in header, or in source file.
        self.inline = inline


class ConstructorMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 attrs: Optional[List[str]] = None):
        super().__init__(inline=inline, attrs=attrs)


class DestructorMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 virtual: bool = True,
                 attrs: Optional[List[str]] = None):
        super().__init__(inline=inline, attrs=attrs)
        self.virtual = virtual


class MemberFunctionMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 virtual: bool = False,
                 const: bool = False,
                 attrs: Optional[List[str]] = None):
        super().__init__(inline=inline, attrs=attrs)
        self.virtual = virtual
        self.const = const


class StaticMemberFunctionMeta(FunctionMeta):
    def __init__(self,
                 inline: bool = False,
                 attrs: Optional[List[str]] = None):
        super().__init__(inline=inline, attrs=attrs)


class ExternalFunctionMeta(FunctionMeta):
    """external function will be put above
    """
    def __init__(self,
                 inline: bool = False,
                 attrs: Optional[List[str]] = None):
        super().__init__(inline=inline, attrs=attrs)


class Argument(object):
    def __init__(self, name: str, type: str, default: str = ""):
        self.name = name
        self.type_str = type  # type: str
        self.default = default  # type: str


class Member(Argument):
    pass


class Class(object):
    def __init__(self):
        self._members = []  # type: List[Member]
        self._param_class = {}  # type: Dict[str, ParameterizedClass]

    def register_member(self, name: str, type: str, default: str = ""):
        self._members.append(Member(name, type, default))

    def register_param_class(self, namespace: str,
                             param_class: "ParameterizedClass"):
        self._param_class[namespace] = param_class


class ParameterizedClass(Class):
    def __init__(self):
        pass


def _meta_decorator(func=None, meta=None):
    if meta is None:
        raise ValueError("this shouldn't happen")

    def wrapper(func):
        if hasattr(func, PCCM_FUNC_META_KEY):
            raise ValueError("you can only use one meta decorator in a function.")
        setattr(func, PCCM_FUNC_META_KEY, meta)
        return func

    if func is not None:
        return wrapper(func)
    else:
        return wrapper


def member_function(func=None,
                    inline: bool = False,
                    virtual: bool = False,
                    const: bool = False,
                    attrs: Optional[List[str]] = None):
    meta = MemberFunctionMeta(inline=inline,
                              virtual=virtual,
                              const=const,
                              attrs=attrs)
    return _meta_decorator(func, meta)


def static_function(func=None,
                    inline: bool = False,
                    attrs: Optional[List[str]] = None):
    meta = StaticMemberFunctionMeta(inline=inline, attrs=attrs)
    return _meta_decorator(func, meta)


def external_function(func=None,
                      inline: bool = False,
                      attrs: Optional[List[str]] = None):
    meta = ExternalFunctionMeta(inline=inline, attrs=attrs)
    return _meta_decorator(func, meta)


def constructor(func=None,
                inline: bool = False,
                attrs: Optional[List[str]] = None):
    meta = ConstructorMeta(inline=inline, attrs=attrs)
    return _meta_decorator(func, meta)


def destructor(func=None,
               inline: bool = False,
               virtual: bool = True,
               attrs: Optional[List[str]] = None):
    meta = MemberFunctionMeta(inline=inline, virtual=virtual, attrs=attrs)
    return _meta_decorator(func, meta)


class FunctionCode(object):
    def __init__(self,
                 code: str,
                 arguments: List[Argument],
                 return_type: str = "decltype(auto)"):
        self.code = code
        self.arguments = arguments
        self.return_type = return_type
