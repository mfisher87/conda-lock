"""
Somewhat hacky solution to create conda lock files.
"""
import logging
import os
import pathlib
import re
import sys
import tempfile

from contextlib import contextmanager
from functools import partial
from types import TracebackType
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import click
import yaml

from ensureconda.api import ensureconda
from ensureconda.resolve import platform_subdir
from typing_extensions import Literal

from conda_lock.click_helpers import OrderedGroup
from conda_lock.common import (
    read_file,
    read_json,
    temporary_file_with_contents,
    write_file,
)
from conda_lock.constants.filenames import DEFAULT_LOCKFILE_NAME, DEFAULT_SOURCE_FILES
from conda_lock.errors import MissingEnvVarError, PlatformValidationError
from conda_lock.install import do_conda_install
from conda_lock.invoke_conda import PathLike, determine_conda_executable
from conda_lock.lockfile import UnknownLockfileVersion, parse_conda_lock_file
from conda_lock.lockfile.v2prelim.models import MetadataOption
from conda_lock.lookup import set_lookup_location
from conda_lock.render import do_render, render_lockfile_for_platform, run_lock
from conda_lock.typing import TKindAll
from conda_lock.validate import do_validate_platform


logger = logging.getLogger(__name__)

if not (sys.version_info.major >= 3 and sys.version_info.minor >= 6):
    print("conda_lock needs to run under python >=3.6")
    sys.exit(1)

HAVE_MAMBA = (
    ensureconda(
        mamba=True, micromamba=False, conda=False, conda_exe=False, no_install=True
    )
    is not None
)

# Captures basic auth credentials, if they exists, in the third capture group.
AUTH_PATTERN = re.compile(r"^(# pip .* @ )?(https?:\/\/)(.*:.*@)?(.*)")

# Do not substitute in comments, but do substitute in pip installable packages
# with the pattern: # pip package @ url.
PKG_PATTERN = re.compile(r"(^[^#@].*|^# pip .*)")

# Captures the domain in the third group.
DOMAIN_PATTERN = re.compile(r"^(# pip .* @ )?(https?:\/\/)?([^\/]+)(.*)")


@click.group(cls=OrderedGroup, default="lock", default_if_no_args=True)
@click.version_option()
def main() -> None:
    """To get help for subcommands, use the conda-lock <SUBCOMMAND> --help"""
    pass


TLogLevel = Union[
    Literal["DEBUG"],
    Literal["INFO"],
    Literal["WARNING"],
    Literal["ERROR"],
    Literal["CRITICAL"],
]


@main.command("lock")
@click.option(
    "--conda", default=None, help="path (or name) of the conda/mamba executable to use."
)
@click.option(
    "--mamba/--no-mamba",
    default=HAVE_MAMBA,
    help="don't attempt to use or install mamba.",
)
@click.option(
    "--micromamba/--no-micromamba",
    default=False,
    help="don't attempt to use or install micromamba.",
)
@click.option(
    "-p",
    "--platform",
    multiple=True,
    help="generate lock files for the following platforms",
)
@click.option(
    "-c",
    "--channel",
    "channel_overrides",
    multiple=True,
    help="""Override the channels to use when solving the environment. These will replace the channels as listed in the various source files.""",
)
@click.option(
    "--dev-dependencies/--no-dev-dependencies",
    is_flag=True,
    default=True,
    help="include dev dependencies in the lockfile (where applicable)",
)
@click.option(
    "-f",
    "--file",
    "files",
    default=DEFAULT_SOURCE_FILES,
    type=click.Path(),
    multiple=True,
    help="path to a conda environment specification(s)",
)
@click.option(
    "-k",
    "--kind",
    default=["lock"],
    type=str,
    multiple=True,
    help="Kind of lock file(s) to generate [should be one of 'lock', 'explicit', or 'env'].",
)
@click.option(
    "--filename-template",
    default="conda-{platform}.lock",
    help="Template for single-platform (explicit, env) lock file names. Filename must include {platform} token, and must not end in '.yml'. For a full list and description of available tokens, see the command help text.",
)
@click.option(
    "--lockfile",
    default=DEFAULT_LOCKFILE_NAME,
    help="Path to a conda-lock.yml to create or update",
)
@click.option(
    "--strip-auth",
    is_flag=True,
    default=False,
    help="Strip the basic auth credentials from the lockfile.",
)
@click.option(
    "-e",
    "--extras",
    "--category",
    default=[],
    type=str,
    multiple=True,
    help="When used in conjunction with input sources that support extras/categories (pyproject.toml) will add the deps from those extras to the render specification",
)
@click.option(
    "--filter-categories",
    "--filter-extras",
    is_flag=True,
    default=False,
    help="In conjunction with extras this will prune out dependencies that do not have the extras specified when loading files.",
)
@click.option(
    "--check-input-hash",
    is_flag=True,
    default=False,
    help="Check existing input hashes in lockfiles before regenerating lock files.  If no files were updated exit with exit code 4.  Incompatible with --strip-auth",
)
@click.option(
    "--log-level",
    help="Log level.",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.option(
    "--pdb", is_flag=True, help="Drop into a postmortem debugger if conda-lock crashes"
)
@click.option(
    "--virtual-package-spec",
    type=click.Path(),
    help="Specify a set of virtual packages to use.",
)
@click.option(
    "--update",
    multiple=True,
    help="Packages to update to their latest versions. If empty, update all.",
)
@click.option(
    "--pypi_to_conda_lookup_file",
    type=str,
    help="Location of the lookup file containing Pypi package names to conda names.",
)
@click.option(
    "--md",
    "--metadata",
    "metadata_choices",
    default=[],
    multiple=True,
    type=click.Choice([md.value for md in MetadataOption]),
    help="Metadata fields to include in lock-file",
)
@click.option(
    "--with-cuda",
    "with_cuda",
    type=str,
    default=None,
    help="Specify cuda version to use in virtual packages. Avoids warning about implicit acceptance of cuda dependencies. Ignored if virtual packages are specified.",
)
@click.option(
    "--without-cuda",
    "with_cuda",
    flag_value="",
    default=None,
    help="Disable cuda in virtual packages. Prevents accepting cuda variants of packages. Ignored if virtual packages are specified.",
)
@click.option(
    "--mdy",
    "--metadata-yaml",
    "--metadata-json",
    "metadata_yamls",
    default=[],
    multiple=True,
    type=click.Path(),
    help="YAML or JSON file(s) containing structured metadata to add to metadata section of the lockfile.",
)
@click.pass_context
def lock(
    ctx: click.Context,
    conda: Optional[str],
    mamba: bool,
    micromamba: bool,
    platform: List[str],
    channel_overrides: List[str],
    dev_dependencies: bool,
    files: List[pathlib.Path],
    kind: List[Union[Literal["lock"], Literal["env"], Literal["explicit"]]],
    filename_template: str,
    lockfile: PathLike,
    strip_auth: bool,
    extras: List[str],
    filter_categories: bool,
    check_input_hash: bool,
    log_level: TLogLevel,
    pdb: bool,
    virtual_package_spec: Optional[pathlib.Path],
    pypi_to_conda_lookup_file: Optional[str],
    with_cuda: Optional[str] = None,
    update: Optional[List[str]] = None,
    metadata_choices: Sequence[str] = (),
    metadata_yamls: Sequence[pathlib.Path] = (),
) -> None:
    """Generate fully reproducible lock files for conda environments.

    By default, a multi-platform lock file is written to conda-lock.yml.

    When choosing the "explicit" or "env" kind, lock files are written to
    conda-{platform}.lock. These filenames can be customized using the
    --filename-template argument. The following tokens are available:

    \b
        platform: The platform this lock file was generated for (conda subdir).
        dev-dependencies: Whether or not dev dependencies are included in this lock file.
        input-hash: A sha256 hash of the lock file input specification.
        version: The version of conda-lock used to generate this lock file.
        timestamp: The approximate timestamp of the output file in ISO8601 basic format.
    """
    logging.basicConfig(level=log_level)

    # Set Pypi <--> Conda lookup file location
    if pypi_to_conda_lookup_file:
        set_lookup_location(pypi_to_conda_lookup_file)

    metadata_enum_choices = set(MetadataOption(md) for md in metadata_choices)

    metadata_yamls = [pathlib.Path(path) for path in metadata_yamls]

    # bail out if we do not encounter the default file if no files were passed
    if ctx.get_parameter_source("files") == click.core.ParameterSource.DEFAULT:
        candidates = list(files)
        candidates += [f.with_name(f.name.replace(".yml", ".yaml")) for f in candidates]
        for f in candidates:
            if f.exists():
                break
        else:
            print(ctx.get_help())
            sys.exit(1)

    if pdb:
        sys.excepthook = _handle_exception_post_mortem

    if not virtual_package_spec:
        candidates = [
            pathlib.Path("virtual-packages.yml"),
            pathlib.Path("virtual-packages.yaml"),
        ]
        for c in candidates:
            if c.exists():
                logger.info("Using virtual packages from %s", c)
                virtual_package_spec = c
                break
    else:
        virtual_package_spec = pathlib.Path(virtual_package_spec)

    files = [pathlib.Path(file) for file in files]
    extras_ = set(extras)
    lock_func = partial(
        run_lock,
        environment_files=files,
        conda_exe=conda,
        platforms=platform,
        mamba=mamba,
        micromamba=micromamba,
        include_dev_dependencies=dev_dependencies,
        channel_overrides=channel_overrides,
        kinds=kind,
        lockfile_path=pathlib.Path(lockfile),
        extras=extras_,
        virtual_package_spec=virtual_package_spec,
        with_cuda=with_cuda,
        update=update,
        filter_categories=filter_categories,
        metadata_choices=metadata_enum_choices,
        metadata_yamls=metadata_yamls,
        strip_auth=strip_auth,
    )
    if strip_auth:
        with tempfile.TemporaryDirectory() as tempdir:
            filename_template_temp = f"{tempdir}/{filename_template.split('/')[-1]}"
            lock_func(filename_template=filename_template_temp)
            filename_template_dir = "/".join(filename_template.split("/")[:-1])
            for file in os.listdir(tempdir):
                lockfile_content = read_file(os.path.join(tempdir, file))
                lockfile_content = _strip_auth_from_lockfile(lockfile_content)
                write_file(lockfile_content, os.path.join(filename_template_dir, file))
    else:
        lock_func(
            filename_template=filename_template, check_input_hash=check_input_hash
        )


@main.command("install")
@click.option(
    "--conda", default=None, help="path (or name) of the conda/mamba executable to use."
)
@click.option(
    "--mamba/--no-mamba",
    default=HAVE_MAMBA,
    help="don't attempt to use or install mamba.",
)
@click.option(
    "--micromamba/--no-micromamba",
    default=False,
    help="don't attempt to use or install micromamba.",
)
@click.option(
    "--copy",
    is_flag=True,
    help=(
        "Install using `--copy` to prevent links. "
        "This is useful for building containers"
    ),
)
@click.option("-p", "--prefix", help="Full path to environment location (i.e. prefix).")
@click.option("-n", "--name", help="Name of environment.")
@click.option(
    "--auth",
    help="The auth file provided as string. Has precedence over `--auth-file`.",
    default="",
)
@click.option("--auth-file", help="Path to the authentication file.", default="")
@click.option(
    "--validate-platform/--no-validate-platform",
    default=True,
    help="Whether the platform compatibility between your lockfile and the host system should be validated.",
)
@click.option(
    "--log-level",
    help="Log level.",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.option(
    "--dev/--no-dev",
    is_flag=True,
    default=True,
    help="install dev dependencies from the lockfile (where applicable)",
)
@click.option(
    "-E",
    "--extras",
    multiple=True,
    default=[],
    help="include extra dependencies from the lockfile (where applicable)",
)
@click.argument(
    "lock-file", default=pathlib.Path(DEFAULT_LOCKFILE_NAME), type=click.Path()
)
@click.pass_context
def install(
    ctx: click.Context,
    conda: Optional[str],
    mamba: bool,
    micromamba: bool,
    copy: bool,
    prefix: Optional[str],
    name: Optional[str],
    lock_file: pathlib.Path,
    auth: Optional[str],
    auth_file: Optional[PathLike],
    validate_platform: bool,
    log_level: TLogLevel,
    dev: bool,
    extras: List[str],
) -> None:
    # bail out if we do not encounter the lockfile
    lock_file = pathlib.Path(lock_file)
    if not lock_file.exists():
        print(ctx.get_help())
        sys.exit(1)

    """Perform a conda install"""
    logging.basicConfig(level=log_level)
    _auth = (
        yaml.safe_load(auth) if auth else read_json(auth_file) if auth_file else None
    )
    _conda_exe = determine_conda_executable(conda, mamba=mamba, micromamba=micromamba)
    install_func = partial(
        do_conda_install, conda=_conda_exe, prefix=prefix, name=name, copy=copy
    )
    if validate_platform and not lock_file.name.endswith(DEFAULT_LOCKFILE_NAME):
        lockfile_contents = read_file(lock_file)
        try:
            do_validate_platform(lockfile_contents)
        except PlatformValidationError as error:
            raise PlatformValidationError(
                error.args[0] + " Disable validation with `--no-validate-platform`."
            )
    with _render_lockfile_for_install(
        lock_file, include_dev_dependencies=dev, extras=set(extras)
    ) as lockfile:
        if _auth is not None:
            with _add_auth(read_file(lockfile), _auth) as lockfile_with_auth:
                install_func(file=lockfile_with_auth)
        else:
            install_func(file=lockfile)


@main.command("render")
@click.option(
    "--dev-dependencies/--no-dev-dependencies",
    is_flag=True,
    default=True,
    help="include dev dependencies in the lockfile (where applicable)",
)
@click.option(
    "-k",
    "--kind",
    default=["explicit"],
    type=click.Choice(["explicit", "env"]),
    multiple=True,
    help="Kind of lock file(s) to generate.",
)
@click.option(
    "--filename-template",
    default="conda-{platform}.lock",
    help="Template for the lock file names. Filename must include {platform} token, and must not end in '.yml'. For a full list and description of available tokens, see the command help text.",
)
@click.option(
    "-e",
    "--extras",
    default=[],
    type=str,
    multiple=True,
    help="When used in conjunction with input sources that support extras (pyproject.toml) will add the deps from those extras to the input specification",
)
@click.option(
    "--log-level",
    help="Log level.",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.option(
    "--pdb", is_flag=True, help="Drop into a postmortem debugger if conda-lock crashes"
)
@click.option(
    "-p",
    "--platform",
    multiple=True,
    help="render lock files for the following platforms",
)
@click.argument("lock-file", default=DEFAULT_LOCKFILE_NAME)
@click.pass_context
def render(
    ctx: click.Context,
    dev_dependencies: bool,
    kind: Sequence[Union[Literal["env"], Literal["explicit"]]],
    filename_template: str,
    extras: List[str],
    log_level: TLogLevel,
    lock_file: PathLike,
    pdb: bool,
    platform: Sequence[str],
) -> None:
    """Render multi-platform lockfile into single-platform env or explicit file"""
    logging.basicConfig(level=log_level)

    if pdb:
        sys.excepthook = _handle_exception_post_mortem

    # bail out if we do not encounter the lockfile
    lock_file = pathlib.Path(lock_file)
    if not lock_file.exists():
        print(ctx.get_help())
        sys.exit(1)

    lock_content = parse_conda_lock_file(lock_file)

    do_render(
        lock_content,
        filename_template=filename_template,
        kinds=kind,
        include_dev_dependencies=dev_dependencies,
        extras=set(extras),
        override_platform=platform,
    )


def _handle_exception_post_mortem(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: Optional[TracebackType],
) -> Any:
    import pdb

    pdb.post_mortem(exc_traceback)


def _add_auth_to_line(line: str, auth: Dict[str, str]) -> str:
    matching_auths = [a for a in auth if a in line]
    if not matching_auths:
        return line
    # If we have multiple matching auths, we choose the longest one.
    matching_auth = max(matching_auths, key=len)
    replacement = f"{auth[matching_auth]}@{matching_auth}"
    return line.replace(matching_auth, replacement)


def _add_auth_to_lockfile(lockfile: str, auth: Dict[str, str]) -> str:
    lockfile_with_auth = "\n".join(
        _add_auth_to_line(line, auth) if PKG_PATTERN.match(line) else line
        for line in lockfile.strip().split("\n")
    )
    if lockfile.endswith("\n"):
        return lockfile_with_auth + "\n"
    return lockfile_with_auth


@contextmanager
def _add_auth(lockfile: str, auth: Dict[str, str]) -> Iterator[pathlib.Path]:
    lockfile_with_auth = _add_auth_to_lockfile(lockfile, auth)
    with temporary_file_with_contents(lockfile_with_auth) as path:
        yield path


def _strip_auth_from_line(line: str) -> str:
    return AUTH_PATTERN.sub(r"\1\2\4", line)


def _extract_domain(line: str) -> str:
    return DOMAIN_PATTERN.sub(r"\3", line)


def _strip_auth_from_lockfile(lockfile: str) -> str:
    lockfile_lines = lockfile.strip().split("\n")
    stripped_lockfile_lines = tuple(
        _strip_auth_from_line(line) if PKG_PATTERN.match(line) else line
        for line in lockfile_lines
    )
    stripped_domains = sorted(
        {
            _extract_domain(stripped_line)
            for line, stripped_line in zip(lockfile_lines, stripped_lockfile_lines)
            if line != stripped_line
        }
    )
    stripped_lockfile = "\n".join(stripped_lockfile_lines)
    if lockfile.endswith("\n"):
        stripped_lockfile += "\n"
    if stripped_domains:
        stripped_domains_doc = "\n".join(f"# - {domain}" for domain in stripped_domains)
        return f"# The following domains require authentication:\n{stripped_domains_doc}\n{stripped_lockfile}"
    return stripped_lockfile


@contextmanager
def _render_lockfile_for_install(
    filename: pathlib.Path,
    include_dev_dependencies: bool = True,
    extras: Optional[AbstractSet[str]] = None,
) -> Iterator[pathlib.Path]:
    """
    Render lock content into a temporary, explicit lockfile for the current platform

    Parameters
    ----------
    filename :
        Path to conda-lock.yml
    include_dev_dependencies :
        Include development dependencies in output
    extras :
        Optional dependency groups to include in output

    """
    kind = _detect_lockfile_kind(pathlib.Path(filename))
    if kind in ("explicit", "env"):
        yield filename
        return

    lock_content = parse_conda_lock_file(pathlib.Path(filename))

    platform = platform_subdir()
    if platform not in lock_content.metadata.platforms:
        suggested_platforms_section = "platforms:\n- "
        suggested_platforms_section += "\n- ".join(
            [platform, *lock_content.metadata.platforms]
        )
        suggested_platform_args = "--platform=" + " --platform=".join(
            [platform, *lock_content.metadata.platforms]
        )
        raise PlatformValidationError(
            f"The lockfile {filename} does not contain a solution for the current "
            f"platform {platform}. The lockfile only contains solutions for the "
            f"following platforms: {', '.join(lock_content.metadata.platforms)}. In "
            f"order to add support for {platform}, you must regenerate the lockfile. "
            f"Either add the following section to your environment.yml:\n\n"
            f"{suggested_platforms_section}\n\n"
            f"or add the following arguments to the conda-lock command:\n\n"
            f"{suggested_platform_args}\n\n"
        )

    # TODO: Move to LockFile
    required_env_vars: Set[str] = set()
    for channel in lock_content.metadata.channels:
        required_env_vars.update(channel.used_env_vars)
    existing_env_vars = {k for k, v in os.environ.items() if v}
    missing_env_vars = required_env_vars - existing_env_vars
    if missing_env_vars:
        msg = ", ".join(sorted(missing_env_vars))
        raise MissingEnvVarError(
            f"Cannot run render lockfile.  Missing environment variables: {msg}"
        )

    content = render_lockfile_for_platform(
        lockfile=lock_content,
        kind="explicit",
        platform=platform,
        include_dev_dependencies=include_dev_dependencies,
        extras=extras,
    )
    with temporary_file_with_contents("\n".join(content) + "\n") as path:
        yield path


class UnknownLockfileKind(ValueError):
    pass


def _detect_lockfile_kind(path: pathlib.Path) -> TKindAll:
    content = path.read_text(encoding="utf-8")
    if "@EXPLICIT" in {line.strip() for line in content.splitlines()}:
        return "explicit"
    try:
        lockfile = yaml.safe_load(content)
        if {"channels", "dependencies"} <= set(lockfile):
            return "env"
        if "version" in lockfile:
            # Version validation is handled by `lockfile.parse_conda_lock_file`
            return "lock"
        raise UnknownLockfileKind(f"Could not detect the kind of lockfile at {path}")
    except yaml.YAMLError:
        raise UnknownLockfileKind(
            f"Could not detect the kind of lockfile at {path}. Note that explicit "
            "lockfiles must contain the line '@EXPLICIT'."
        )


# Unused
def fn_to_dist_name(fn: str) -> str:
    if fn.endswith(".conda"):
        fn, _, _ = fn.partition(".conda")
    elif fn.endswith(".tar.bz2"):
        fn, _, _ = fn.partition(".tar.bz2")
    else:
        raise RuntimeError(f"unexpected file type {fn}", fn)
    return fn


if __name__ == "__main__":
    main()
