from typing import Any, Callable, Dict, Hashable, Optional


class TypeHandler(object):
    @staticmethod
    def register_handler(global_dict, *class_types):
        def wrap_func(handler):
            for cls_type in class_types:
                global_dict[cls_type] = handler

            def new_handler(obj):
                return handler(obj)

            return handler

        return wrap_func

    @staticmethod
    def get_handler(global_dict, obj) -> Optional[Callable]:
        # 1. fast check
        obj_type = type(obj)
        if obj_type in global_dict:
            return global_dict[obj_type]
        else:
            # 2. iterate and isinstance
            for class_type, handler in global_dict.items():
                if isinstance(obj, class_type):
                    return handler
        return None

    @staticmethod
    def get_type_handler(global_dict, obj_type) -> Optional[Callable]:
        # 1. fast check
        if obj_type in global_dict:
            return global_dict[obj_type]
        else:
            # 2. iterate and isinstance
            for class_type, handler in global_dict.items():
                if issubclass(obj_type, class_type):
                    return handler
        return None
