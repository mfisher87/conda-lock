import re

from typing import Optional, Tuple

from conda_lock.errors import PlatformValidationError


# Captures the platform in the first group.
PLATFORM_PATTERN = re.compile(r"^# platform: (.*)$")


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
