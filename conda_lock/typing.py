from typing import List, Literal, Union


TKindAll = Union[Literal["explicit"], Literal["lock"], Literal["env"]]
# Unused:
TKindRendarable = Union[Literal["explicit"], Literal["lock"], Literal["env"]]
TKindNoLock = Union[Literal["explicit"], Literal["env"]]

TValidationCheck = Union[Literal["hash"], Literal["compatible"]]
