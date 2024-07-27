from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.core.parsers import Attr

class Argument(object):
    def __init__(self,
                 name: str,
                 type: str,
                 default: Optional[str] = None,
                 array: Optional[Union[int, str]] = None,
                 pyanno: Optional[str] = None,
                 doc: Optional[str] = None,
                 user_attrs: Optional[List[Attr]] = None,
                 attributes: Optional[List[str]] = None,
                 userdata: Any = None):
        self.name = name.strip()
        self.type_str = str(type).strip()  # type: str
        self.default = default
        self.array = array
        self.pyanno = pyanno
        self.doc = doc
        if user_attrs is None:
            self.attrs: List[Attr] = []
        else:
            self.attrs: List[Attr] = user_attrs
        self.attributes = attributes
        self.userdata = userdata
        if pyanno is not None:
            self.pyanno = pyanno.strip()
            assert len(pyanno) != 0

class TemplateTypeArgument(object):
    def __init__(self,
                 name: str,
                 default: Optional[str] = None,
                 template: str = "",
                 packed: bool = False):
        self.name = name
        self.template = template
        self.packed = packed
        self.default = default

    def to_string(self) -> str:
        pack = ""
        if self.packed:
            if self.default:
                raise ValueError("packed arg can't have default value")
            pack = "..."
        if self.template:
            res = "{} typename{} {}".format(self.template, pack, self.name)
        else:
            res = "typename{} {}".format(pack, self.name)
        if self.default:
            res += " = {}".format(self.default)
        return res

class TemplateNonTypeArgument(object):
    def __init__(self,
                 name: str,
                 type: str,
                 default: Optional[str] = None,
                 packed: bool = False):
        self.name = name
        self.type = type
        self.packed = packed
        self.default = default

    def to_string(self) -> str:
        pack = ""
        if self.packed:
            if self.default:
                raise ValueError("packed arg can't have default value")
            pack = "..."
        res = "{}{} {}".format(self.type, pack, self.name)
        if self.default:
            res += " = {}".format(self.default)
        return res
