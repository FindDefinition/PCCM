from typing import Literal, Optional, Union


class EnumClassValue(object):
    def __init__(self, value: int):
        self.__value = value

    def __int__(self):
        return self.__value

    def __index__(self) -> int:
        return self.__value

    @property
    def value(self):
        return self.__value

    @property
    def name(self) -> str:
        ...


class EnumValue(EnumClassValue):
    def __eq__(self, other: Optional["EnumValue"]) -> bool:
        ...

    def __ne__(self, other: Optional["EnumValue"]) -> bool:
        ...

    def __lt__(self, other: Union["EnumValue", int]) -> bool:
        ...

    def __gt__(self, other: Union["EnumValue", int]) -> bool:
        ...

    def __le__(self, other: Union["EnumValue", int]) -> bool:
        ...

    def __ge__(self, other: Union["EnumValue", int]) -> bool:
        ...

    def __and__(self, other: Union["EnumValue", int]) -> int:
        ...

    def __rand__(self, other: Union["EnumValue", int]) -> int:
        ...

    def __or__(self, other: Union["EnumValue", int]) -> int:
        ...

    def __ror__(self, other: Union["EnumValue", int]) -> int:
        ...

    def __xor__(self, other: Union["EnumValue", int]) -> int:
        ...

    def __rxor__(self, other: Union["EnumValue", int]) -> int:
        ...

    def __invert__(self) -> int:
        ...
