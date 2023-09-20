from typing import Final, List

from conda_lock.typing import TKindNoLock, TValidationCheck


CHOICES_VALIDATION_CHECKS: Final[List[TValidationCheck]] = ["hash", "compatible"]
CHOICES_KIND_NO_LOCK: Final[List[TKindNoLock]] = ["explicit", "env"]
