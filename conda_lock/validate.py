import pathlib
import re

from typing import Callable, Final, Optional, Tuple

from conda_lock.errors import PlatformValidationError
from conda_lock.lockfile import parse_conda_lock_file
from conda_lock.typing import TValidationCheck


# Captures the platform in the first group.
PLATFORM_PATTERN = re.compile(r"^# platform: (.*)$")


def validate_lockfile_compatible_with_sources(lockfile_path: pathlib.Path) -> None:
    """Check that all locked dependencies are compatible with source specification."""
    # TODO: pass in the lock content instead of reading for each check
    parse_conda_lock_file(lockfile_path)
    ...

    error = False
    if error:
        raise RuntimeError(
            f"Lockfile {lockfile_path} dependencies conflict with source specifications:\n"
            " {}"
        )

    print(f"Check passed: Lockfile {lockfile_path} is compatible with its sources.")


def validate_lockfile_from_hashes(lockfile_path: pathlib.Path) -> None:
    """Check that all lockfile sources still match their recorded hashes."""
    # TODO: pass in the lock content instead of reading for each check
    parse_conda_lock_file(lockfile_path)
    ...

    error = False
    if error:
        raise RuntimeError(f"Lockfile {lockfile_path} source {{}} hash mismatch.")

    print(f"Check passed: Lockfile {lockfile_path} hashes match sources.")


VALIDATION_CHECKS: Final[dict[TValidationCheck, Callable]] = {
    "compatible": validate_lockfile_compatible_with_sources,
    "hash": validate_lockfile_from_hashes,
}


def do_validate_platform(lockfile: str) -> None:
    platform_lockfile = extract_platform(lockfile)
    try:
        success, platform_sys = _do_validate_platform(platform_lockfile)
    except KeyError:
        raise RuntimeError(f"Unknown platform type in lockfile '{platform_lockfile}'.")
    if not success:
        raise PlatformValidationError(
            f"Platform in lockfile '{platform_lockfile}' is not compatible with system platform '{platform_sys}'."
        )


def _do_validate_platform(platform: str) -> Tuple[bool, str]:
    from ensureconda.resolve import platform_subdir

    determined_subdir = platform_subdir()
    return platform == determined_subdir, determined_subdir


def extract_platform(lockfile: str) -> str:
    for line in lockfile.strip().split("\n"):
        platform = _extract_platform(line)
        if platform:
            return platform
    raise RuntimeError("Cannot find platform in lockfile.")


def _extract_platform(line: str) -> Optional[str]:
    search = PLATFORM_PATTERN.search(line)
    if search:
        return search.group(1)
    return None
