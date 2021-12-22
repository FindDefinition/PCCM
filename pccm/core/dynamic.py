from typing import Any, Callable, Dict, List, Tuple, Union
from pccm.core import (ConstructorMeta, DestructorMeta, ExternalFunctionMeta,
                       FunctionCode, FunctionDecl, MemberFunctionMeta,
                       StaticMemberFunctionMeta, Class, ParameterizedClass)

import os 
import threading
import types
from collections import OrderedDict
import json 
import inspect
from ccimport import source_iter
from pccm.constants import PCCM_INLINE_LIBRARY_PATH
import re 
from collections.abc import Mapping
from pccm.core.pycore import TypeHandler

_TYPE_STRING_PATTERN = re.compile(r"<class '(.*)'>")

_INLINE_CAPTURE_HOOKS: Dict[str, Callable[[FunctionCode, Any], None]] = {}
_OBJ_TYPE_HANDLERS : Dict[str, Callable[[Any], str]] = {}



def register_type_handler(*class_types):
    return TypeHandler.register_handler(_OBJ_TYPE_HANDLERS,
                                                *class_types)

def get_type_handler(obj):
    return TypeHandler.get_handler(_OBJ_TYPE_HANDLERS, obj)

def get_local_vars(frame: types.FrameType):
    code = frame.f_code
    vars_order = code.co_varnames + code.co_cellvars + code.co_freevars + tuple(
        frame.f_locals.keys())
    result_items = [(key, value) for key, value in frame.f_locals.items()]
    result_items.sort(key=lambda key_value: vars_order.index(key_value[0]))
    result_items = filter(lambda x: not isinstance(x[1], types.ModuleType),
                          result_items)
    result = OrderedDict(result_items)
    return result

def gcs(*instances):
    if len(instances) == 0:
        return None
    classes = [inspect.getmro(type(x)) for x in instances]
    for x in classes[0]:
        if all(x in mro for mro in classes):
            return x

def get_base_type_string(obj):
    if isinstance(obj, int):
        return "int"
    elif isinstance(obj, float):
        return "float"
    elif isinstance(obj, bool):
        return "bool"
    elif isinstance(obj, str):
        return "str"
    else:
        raise NotImplementedError("only support int/float/bool/str for non-container types")

def get_cls_type_string(obj_type, simple_type=False):
    type_string = str(obj_type)
    type_string = _TYPE_STRING_PATTERN.match(type_string).group(1)
    if simple_type:
        type_string = type_string.split(".")[-1]
    return type_string


def nested_type_analysis(obj,
                         iter_limit=10):
    """take a obj, produce nested dict, tuple and list type. typescript-style.
    TODO try merge history types
    TODO analysis function
    TODO better common_base
    """
    if isinstance(obj, list):
        types_union = set()
        for o in obj:
            if iter_limit < 0:
                break
            iter_limit -= 1
            type_s = nested_type_analysis(o)
            types_union.add(type_s)
        if len(types_union) == 1:
            return "{}[]".format(types_union.pop())
        elif len(types_union) == 0:
            return "[]"
        else:
            return "({})[]".format("|".join(types_union))
    elif isinstance(obj, tuple):
        type_tuple = []
        for o in obj:
            if iter_limit < 0:
                break
            iter_limit -= 1
            type_tuple.append(nested_type_analysis(o))
        return "[{}]".format(", ".join(type_tuple))
    elif isinstance(obj, Mapping):
        key_types_union = set()
        val_type_union = set()
        for k, v in obj.items():
            if iter_limit < 0:
                break
            iter_limit -= 1
            key_type_s = nested_type_analysis(k)
            val_type_s = nested_type_analysis(v)
            key_types_union.add(key_type_s)
            val_type_union.add(val_type_s)
        if len(key_types_union) == 0:
            return "Map<>"
        return "Map<{},{}>".format("|".join(key_types_union),
                                   "|".join(val_type_union))
    else:
        return get_type_string(obj)


class InlineBuilder:
    """
    inliner.inline(...)    
    """
    def __init__(self, deps: List[Class]) -> None:
        self.deps = deps 


    def inline(self, code: Union[str, FunctionCode]):
        """use $var to capture python objects
        only support: 
        1. int/float/str and nested containers of int/float/str.
        2. np.ndarray -> py::array
        3. tv.Tensor -> tv::Tensor (via pccm.libs.tvten)
        4. torch.Tensor -> torch::Tensor (via pccm.libs.pytorch)
        """
        if isinstance(code, FunctionCode):
            code_str = code.inspect_body()
        else:
            code_str = code 
        # 1. extract captured vars
        it = source_iter.CppSourceIterator(code_str)
        all_captured_idens: List[str] = []
        # hold ranges for further replace
        all_captured_ranges: List[Tuple[int, int]] = []
        for pose in it.get_symbol_poses("$"):
            it.move(pose + 1)
            iden = it.next_identifier()
            assert iden is not None, "you can't use $ without a identifier."
            all_captured_idens.append(iden.name)
            all_captured_ranges.append((pose + 1, iden.end))
        # 2. find captures in prev frame
        cur_frame = inspect.currentframe()
        assert cur_frame is not None 
        prev_frame = cur_frame.f_back 
        assert prev_frame is not None 
        local_vars = get_local_vars(prev_frame)
        del cur_frame
        del prev_frame
        # 3. inference c++ types
        for cap in all_captured_idens:
            if cap not in local_vars:
                raise ValueError(f"can't find your capture {cap} in prev frame.")
            


