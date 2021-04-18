from pccm import builder, core, middlewares
from pccm.core import Argument, Class, FunctionCode, ParameterizedClass
from pccm.core.markers import (constructor, destructor, external_function,
                               member_function, static_function)
from pccm.middlewares import pybind
from pccm.targets import cuda

def boolean(val: bool):
    return "true" if val else "false"

