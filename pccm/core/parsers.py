"""parsers for advanced grammar in arguments
"""
import enum 
from typing import Dict, List, Optional, Union
from lark import Lark, Transformer, v_args
import lark

SIMPLE_TYPES = set(["int", "int8_t", "int16_t", "int32_t", "int64_t", "unsigned",
            "uint8_t", "uint16_t", "uint32_t", "uint64_t", "std::intptr_t", "std::uintptr_t",
            "size_t", "std::size_t", "long", "short", "float", "double", "bool", "std::string", 
            "unsigned long", "unsigned int"])

class Attr:
    def __init__(self, name: str, value: Union[str, int, float, list]) -> None:
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"{self.name}={self.value}"


class NameWithAttrs:
    def __init__(self, name: str, attrs: List[Attr]) -> None:
        self.name = name
        self.attrs = attrs

    def __repr__(self) -> str:
        return f"{self.name}={self.attrs}"


_ARG_WITH_ATTR_GRAMMAR = """
args: arg ("," arg)*
arg: identifier | identifier attrs
attrs: "[" attr_arg ("," attr_arg)*  "]"
attr_arg: value | identifier "=" value
value: string | number | identifier | value_list | value_tuple
value_list: "[" value ("," value)*  "]"
value_tuple: "(" value ("," value)*  ")"

identifier: NAME

string : ESCAPED_STRING
number: SIGNED_NUMBER
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%import common.CNAME -> NAME

%ignore WS
"""


class AttrArgTransformer(Transformer):
    CppTypeMap = {
        "float": "double",
        "int": "int",
        "bool": "bool",
        "string": "std::string",
    }

    def attr_arg(self, args):
        if len(args) == 1:
            return Attr("", args[0])
        else:
            return Attr(args[0], args[1])

    def value(self, args):
        return args[0]

    def value_list(self, args):
        return args

    def value_tuple(self, args):
        return list(args)

    def attrs(self, args):
        return args

    def identifier(self, args):
        return args[0].value

    def arg(self, args):
        if len(args) == 1:
            return NameWithAttrs(args[0], [])
        else:
            return NameWithAttrs(args[0], args[1])

    def args(self, args):
        return args

    number = v_args(inline=True)(int)

    @v_args(inline=True)
    def string(self, s):
        return s[1:-1].replace('\\"', '"')


_ARG_WITH_ATTR_PARSER = Lark(_ARG_WITH_ATTR_GRAMMAR,
                             parser='lalr',
                             start='args',
                             transformer=AttrArgTransformer())


def arg_parser(data: str):
    """parse "x[a1, a2, a3], y[k1=a1, a2, k3=a3], z"
    nested [] are ignored.
    """
    res: List[NameWithAttrs] = _ARG_WITH_ATTR_PARSER.parse(data)
    return res

# https://alx71hub.github.io/hcb/#function-definition

_CPP_TYPE_GRAMMAR_V2 = """

type_decl: decl_spec_seq decl
decl_spec_seq: decl_spec | decl_spec decl_spec_seq
decl_spec: type_spec |
            func_spec
type_spec: trailing_type_spec |
    class_spec |
    enum_spec
trailing_type_spec:
    simple_type_spec |
    elab_type_spec |
    typename_spec |
    cv_qual
simple_type_spec: ns_sep? nested_name_spec? type_name |
    ns_sep? nested_name_spec template simple_template_id |
    char |
    char16_t |
    char32_t |
    wchar_t |
    bool |
    short |
    int |
    long |
    signed |
    unsigned |
    float |
    double |
    void |
    auto |
    decl_spec


declaractor: 

decl: ptr_decl |
      ref_decl |
      rref_decl

ptr_decl: "*" cv_qual? decl?

ref_decl: "&" decl?

rref_decl: "&&" decl?

type: decl_seq decl?

decl_seq: decl_specs (decl_specs)*

decl_specs: "const" -> const
    | "static" -> static
    | "inline" -> inline
    | "constexpr" -> constexpr
    | "mutable" -> mutable
    | "__restrict__" -> restrict
    | base_type
    | "::" -> ns_sep

base_type: qualified_id | qualified_id "<" template_args ">"

qualified_id: identifier ("::" identifier)*

template_args: template_arg ("," template_arg)*

template_arg: type | number

cv_qual: "const" -> const
        | "volatile" -> volatile

identifier: NAME
string : ESCAPED_STRING
number: SIGNED_NUMBER

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%import common.CNAME -> NAME

%ignore WS
"""


# https://en.cppreference.com/w/cpp/language/declarations
_CPP_TYPE_GRAMMAR = """
decl: ptr_decl |
      ref_decl |
      rref_decl

ptr_decl: "*" cv_qual? decl?

ref_decl: "&" decl?

rref_decl: "&&" decl?

type: decl_seq decl?

decl_seq: decl_specs (decl_specs)*

decl_specs: "const" -> const
    | "static" -> static
    | "inline" -> inline
    | "constexpr" -> constexpr
    | "mutable" -> mutable
    | "__restrict__" -> restrict
    | base_type

base_type: qualified_id | qualified_id "<" template_args ">"

qualified_id: identifier ("::" identifier)*

template_args: template_arg ("," template_arg)*

template_arg: type | number

cv_qual: "const" -> const
        | "volatile" -> volatile

identifier: NAME
string : ESCAPED_STRING
number: SIGNED_NUMBER

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%import common.CNAME -> NAME

%ignore WS
"""

class CppKeyWord(enum.Enum):
    Const = "const"
    Static = "static"
    Inline = "inline"
    Constexpr = "constexpr"
    Mutable = "mutable"
    Volatile = "volatile"

class Decl:
    def __init__(self, decl: Optional["Decl"] = None) -> None:
        self.decl = decl 
    
    def __str__(self):
        raise NotImplementedError


class QualifiedId:
    def __init__(self, names: List[str]) -> None:
        self.names = names 

    def get_dot_name(self):
        return ".".join(self.names)
    
    def get_cpp_name(self):
        return "::".join(self.names)

    def __repr__(self):
        return self.get_cpp_name()

    def __str__(self):
        return self.get_cpp_name()

class QualifiedIdDecl(Decl):
    def __init__(self, names: List[str]) -> None:
        super().__init__()
        self.names = names 

    def __str__(self):
        return "::".join(self.names)

class PtrDecl(Decl):
    def __init__(self, decl: Optional["Decl"], const: bool, volatile: bool) -> None:
        super().__init__(decl)
        self.const = const 
        self.volatile = volatile

    def __str__(self):
        res = "*"
        if self.const:
            res += " const"
        if self.volatile:
            res += " volatile"
        if self.decl is not None:
            return res + " " + str(self.decl)
        return res 
        
class RefDecl(Decl):
    def __init__(self, decl: Optional["Decl"]) -> None:
        super().__init__(decl)

    def __str__(self):
        if self.decl is None:
            return "&"
        return "& " + str(self.decl)
    
class RRefDecl(Decl):
    def __init__(self, decl: Optional["Decl"]) -> None:
        super().__init__(decl)

    def __str__(self):
        if self.decl is None:
            return "&&"
        return "&& " + str(self.decl)

class DeclSpec:
    def __init__(self, val: Union[CppKeyWord, "BaseType"]) -> None:
        self.val = val

    def __str__(self):
        if isinstance(self.val, BaseType):
            return str(self.val)
        return self.val.value


_DECL_TYPES = Union[PtrDecl, RefDecl, RRefDecl]

class CppType:
    def __init__(self, decl_seq: List[DeclSpec], decl: Optional[_DECL_TYPES] = None) -> None:
        self.decl_seq = decl_seq
        self.decl = decl

    def __str__(self):
        decl_seq_strs = [str(d) for d in self.decl_seq]
        if self.decl is None:
            return f"{' '.join(decl_seq_strs)}"
        return f"{' '.join(decl_seq_strs)} {self.decl}"

    @property 
    def base_type(self):
        for decl in self.decl_seq:
            if isinstance(decl.val, BaseType):
                return decl.val
        raise ValueError("can't find base type in your decl seq")

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: "CppType"):
        return hash(other) == hash(self)

class BaseType:
    def __init__(self, name: QualifiedId, args: List[CppType]) -> None:
        self.name = name
        self.args = args

        self._is_std_type: Optional[bool] = None

    def __str__(self):
        if not self.args:
            return str(self.name)
        arg_strs = [str(d) for d in self.args]
        return f"{self.name}<{', '.join(arg_strs)}>"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: "BaseType"):
        return hash(other) == hash(self)

    @property 
    def qualname(self):
        return self.name.get_cpp_name()

    def is_simple_type(self):
        all_args_simple = all(x.base_type.qualname in SIMPLE_TYPES for x in self.args)
        return self.qualname in SIMPLE_TYPES and all_args_simple
        
    def is_std_type(self):
        if self._is_std_type is not None:
            return self._is_std_type
        is_simple_type = self.qualname in SIMPLE_TYPES
        res = is_simple_type or (self.name.names[0] == "std" and all(x.base_type.is_std_type() for x in self.args))
        self._is_std_type = res 
        return res


class CppTypeTransformer(Transformer):
    const = lambda self, _: CppKeyWord.Const
    static = lambda self, _: CppKeyWord.Static
    inline = lambda self, _: CppKeyWord.Inline
    constexpr = lambda self, _: CppKeyWord.Constexpr
    mutable = lambda self, _: CppKeyWord.Mutable
    volatile = lambda self, _: CppKeyWord.Volatile

    number = v_args(inline=True)(int)

    @v_args(inline=True)
    def string(self, s):
        return s[1:-1].replace('\\"', '"')



    def attr_arg(self, args):
        if len(args) == 1:
            return Attr("", args[0])
        else:
            return Attr(args[0], args[1])

    def value(self, args):
        return args[0]

    def value_list(self, args):
        return args

    def value_tuple(self, args):
        return list(args)

    def attrs(self, args):
        return args

    def identifier(self, args):
        return args[0].value

    def arg(self, args):
        if len(args) == 1:
            return NameWithAttrs(args[0], [])
        else:
            return NameWithAttrs(args[0], args[1])

    def args(self, args):
        return args



if __name__ == "__main__":
    data = "x[a1, a2, a3], y[k1=a1, a2, k3=[1,2,3]], z"
    print(arg_parser(data))