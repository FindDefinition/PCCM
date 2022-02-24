import inspect
from typing import Any 

def get_members(obj: Any, no_parent: bool = True):
    """this function return member functions that keep def order.
    """
    this_cls = type(obj)
    if not no_parent:
        res = inspect.getmembers(this_cls, inspect.isfunction)
        # inspect.getsourcelines need to read file, so .__code__.co_firstlineno
        # is greatly faster than it.
        # res.sort(key=lambda x: inspect.getsourcelines(x[1])[1])
        res.sort(key=lambda x: x[1].__code__.co_firstlineno)
        return res
    parents = inspect.getmro(this_cls)[1:]
    parents_methods = set()
    for parent in parents:
        members = inspect.getmembers(parent, predicate=inspect.isfunction)
        parents_methods.update(members)

    child_methods = set(
        inspect.getmembers(this_cls, predicate=inspect.isfunction))
    child_only_methods = child_methods - parents_methods
    res = list(child_only_methods)
    # res.sort(key=lambda x: inspect.getsourcelines(x[1])[1])
    res.sort(key=lambda x: x[1].__code__.co_firstlineno)
    return res

