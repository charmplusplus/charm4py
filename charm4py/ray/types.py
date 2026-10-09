"""``ray.types``: as in Ray, ObjectRef here is a generic type for annotations
(``ObjectRef[int]``), not the runtime class of references (that is
``ray.ObjectRef``)."""

from typing import Generic, TypeVar

T = TypeVar('T')


class ObjectRef(Generic[T]):
    pass


__all__ = ['ObjectRef']
