import sys 
import os 
import importlib.util

def project_is_editable(proj_name: str):
    """Is distribution an editable install?"""
    spec = importlib.util.find_spec(proj_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"{proj_name} not found. you need to install it by pip.")
    for path_item in sys.path:
        egg_link = os.path.join(path_item, proj_name + '.egg-link')
        if os.path.isfile(egg_link):
            return True
    return False

def project_is_installed(proj_name: str):
    """Is distribution an editable install?"""
    spec = importlib.util.find_spec(proj_name)
    if spec is None or spec.origin is None:
        return False 
    return True 

def _make_unique_name(unique_set, name, max_count=10000):
    if name not in unique_set:
        unique_set.add(name)
        return name
    for i in range(max_count):
        new_name = name + "_{}".format(i)
        if new_name not in unique_set:
            unique_set.add(new_name)
            return new_name
    raise ValueError("max count reached")


class UniqueNamePool:
    def __init__(self, max_count=10000):
        self.max_count = max_count
        self.unique_set = set()

    def __call__(self, name):
        return _make_unique_name(self.unique_set, name, self.max_count)

    def __contains__(self, key):
        return key in self.unique_set