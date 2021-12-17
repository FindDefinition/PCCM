"""parsers for advanced grammar in arguments
"""

from typing import List, Union
from lark import Lark, Transformer, v_args
import lark


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


if __name__ == "__main__":
    data = "x[a1, a2, a3], y[k1=a1, a2, k3=[1,2,3]], z"
    print(arg_parser(data))