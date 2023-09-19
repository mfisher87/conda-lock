import datetime
import importlib.util
import itertools
import logging
import pathlib
import posixpath
import re
import sys

from importlib.metadata import distribution
from typing import AbstractSet, Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

import yaml

from typing_extensions import Literal

from conda_lock.common import relative_path
from conda_lock.conda_solver import solve_conda
from conda_lock.constants.filenames import DEFAULT_LOCKFILE_NAME, DEFAULT_SOURCE_FILES
from conda_lock.invoke_conda import PathLike, determine_conda_executable
from conda_lock.lockfile import parse_conda_lock_file, write_conda_lock_file
from conda_lock.lockfile.v2prelim.models import (
    GitMeta,
    InputMeta,
    LockedDependency,
    Lockfile,
    LockMeta,
    MetadataOption,
    TimeMeta,
    UpdateSpecification,
)
from conda_lock.models.channel import Channel
from conda_lock.models.lock_spec import LockSpecification
from conda_lock.pypi_solver import solve_pypi
from conda_lock.src_parser import make_lock_spec
from conda_lock.typing import TKindAll
from conda_lock.virtual_package import (
    default_virtual_package_repodata,
    virtual_package_repo_from_specification,
)


logger = logging.getLogger(__name__)

_implicit_cuda_message = """
  'cudatoolkit' package added implicitly without specifying that cuda packages
  should be accepted.
  Specify a cuda version via `--with-cuda VERSION` or via virtual packages
  to suppress this warning,
  or pass `--without-cuda` to explicitly exclude cuda packages.
"""
KIND_EXPLICIT: Literal["explicit"] = "explicit"
KIND_LOCK: Literal["lock"] = "lock"
KIND_ENV: Literal["env"] = "env"
KIND_USE_TEXT = {
    KIND_EXPLICIT: "conda create --name YOURENV --file {lockfile}",
    KIND_ENV: "conda env create --name YOURENV --file {lockfile}",
    KIND_LOCK: "conda-lock install --name YOURENV {lockfile}",
}
KIND_FILE_EXT = {
    KIND_EXPLICIT: "",
    KIND_ENV: ".yml",
    KIND_LOCK: "." + DEFAULT_LOCKFILE_NAME,
}
DEFAULT_KINDS: List[Union[Literal["explicit"], Literal["lock"]]] = [
    KIND_EXPLICIT,
    KIND_LOCK,
]

INPUT_HASH_PATTERN = re.compile(r"^# input_hash: (.*)$")


def make_lock_files(  # noqa: C901
    *,
    conda: PathLike,
    src_files: List[pathlib.Path],
    kinds: Sequence[TKindAll],
    lockfile_path: pathlib.Path = pathlib.Path(DEFAULT_LOCKFILE_NAME),
    platform_overrides: Optional[Sequence[str]] = None,
    channel_overrides: Optional[Sequence[str]] = None,
    virtual_package_spec: Optional[pathlib.Path] = None,
    update: Optional[List[str]] = None,
    include_dev_dependencies: bool = True,
    filename_template: Optional[str] = None,
    filter_categories: bool = False,
    extras: Optional[AbstractSet[str]] = None,
    check_input_hash: bool = False,
    metadata_choices: AbstractSet[MetadataOption] = frozenset(),
    metadata_yamls: Sequence[pathlib.Path] = (),
    with_cuda: Optional[str] = None,
    strip_auth: bool = False,
) -> None:
    """
    Generate a lock file from the src files provided

    Parameters
    ----------
    conda :
        Path to conda, mamba, or micromamba
    src_files :
        Files to parse requirements from
    kinds :
        Lockfile formats to output
    lockfile_path :
        Path to a conda-lock.yml to create or update
    platform_overrides :
        Platforms to solve for. Takes precedence over platforms found in src_files.
    channel_overrides :
        Channels to use. Takes precedence over channels found in src_files.
    virtual_package_spec :
        Path to a virtual package repository that defines each platform.
    update :
        Names of dependencies to update to their latest versions, regardless
        of whether the constraint in src_files has changed.
    include_dev_dependencies :
        Include development dependencies in explicit or env output
    filename_template :
        Format for names of rendered explicit or env files. Must include {platform}.
    extras :
        Include the given extras in explicit or env output
    filter_categories :
        Filter out unused categories prior to solving
    check_input_hash :
        Do not re-solve for each target platform for which specifications are unchanged
    metadata_choices:
        Set of selected metadata fields to generate for this lockfile.
    with_cuda:
        The version of cuda requested.
        '' means no cuda.
        None will pick a default version and warn if cuda packages are installed.
    metadata_yamls:
        YAML or JSON file(s) containing structured metadata to add to metadata section of the lockfile.
    """

    # initialize virtual packages
    if virtual_package_spec and virtual_package_spec.exists():
        virtual_package_repo = virtual_package_repo_from_specification(
            virtual_package_spec
        )
        cuda_specified = True
    else:
        if with_cuda is None:
            cuda_specified = False
            with_cuda = "11.4"
        else:
            cuda_specified = True

        virtual_package_repo = default_virtual_package_repodata(cuda_version=with_cuda)

    required_categories = {"main"}
    if include_dev_dependencies:
        required_categories.add("dev")
    if extras is not None:
        required_categories.update(extras)

    with virtual_package_repo:
        lock_spec = make_lock_spec(
            src_files=src_files,
            channel_overrides=channel_overrides,
            platform_overrides=platform_overrides,
            virtual_package_repo=virtual_package_repo,
            required_categories=required_categories if filter_categories else None,
        )
        original_lock_content: Optional[Lockfile] = None

        if lockfile_path.exists():
            import yaml

            try:
                original_lock_content = parse_conda_lock_file(lockfile_path)
            except (yaml.error.YAMLError, FileNotFoundError):
                logger.warning(
                    "Failed to parse existing lock.  Regenerating from scratch"
                )
                original_lock_content = None
        else:
            original_lock_content = None

        platforms_to_lock: List[str] = []
        platforms_already_locked: List[str] = []
        if original_lock_content is not None:
            platforms_already_locked = list(original_lock_content.metadata.platforms)
            update_spec = UpdateSpecification(
                locked=original_lock_content.package, update=update
            )
            for platform in lock_spec.platforms:
                if (
                    update
                    or platform not in platforms_already_locked
                    or not check_input_hash
                    or lock_spec.content_hash_for_platform(platform)
                    != original_lock_content.metadata.content_hash[platform]
                ):
                    platforms_to_lock.append(platform)
                    if platform in platforms_already_locked:
                        platforms_already_locked.remove(platform)
        else:
            platforms_to_lock = lock_spec.platforms
            update_spec = UpdateSpecification()

        if platforms_already_locked:
            print(
                f"Spec hash already locked for {sorted(platforms_already_locked)}. Skipping solve.",
                file=sys.stderr,
            )
        platforms_to_lock = sorted(set(platforms_to_lock))

        if not platforms_to_lock:
            new_lock_content = original_lock_content
        else:
            print(f"Locking dependencies for {platforms_to_lock}...", file=sys.stderr)

            fresh_lock_content = create_lockfile_from_spec(
                conda=conda,
                spec=lock_spec,
                platforms=platforms_to_lock,
                lockfile_path=lockfile_path,
                update_spec=update_spec,
                metadata_choices=metadata_choices,
                metadata_yamls=metadata_yamls,
                strip_auth=strip_auth,
            )

            if not original_lock_content:
                new_lock_content = fresh_lock_content
            else:
                # Persist packages from original lockfile for platforms not requested for lock
                packages_not_to_lock = [
                    dep
                    for dep in original_lock_content.package
                    if dep.platform not in platforms_to_lock
                ]
                lock_content_to_persist = original_lock_content.copy(
                    deep=True,
                    update={"package": packages_not_to_lock},
                )
                new_lock_content = lock_content_to_persist | fresh_lock_content

            if "lock" in kinds:
                write_conda_lock_file(
                    new_lock_content,
                    lockfile_path,
                    metadata_choices=metadata_choices,
                )
                print(
                    " - Install lock using:",
                    KIND_USE_TEXT["lock"].format(lockfile=str(lockfile_path)),
                    file=sys.stderr,
                )

        # After this point, we're working with `new_lock_content`, never
        # `original_lock_content` or `fresh_lock_content`.
        assert new_lock_content is not None

        # check for implicit inclusion of cudatoolkit
        # warn if it was pulled in, but not requested explicitly

        if not cuda_specified:
            # asking for 'cudatoolkit' is explicit enough
            cudatoolkit_requested = any(
                pkg.name == "cudatoolkit"
                for pkg in itertools.chain(*lock_spec.dependencies.values())
            )
            if not cudatoolkit_requested:
                for package in new_lock_content.package:
                    if package.name == "cudatoolkit":
                        logger.warning(_implicit_cuda_message)
                        break

        do_render(
            new_lock_content,
            kinds=[k for k in kinds if k != "lock"],
            include_dev_dependencies=include_dev_dependencies,
            filename_template=filename_template,
            extras=extras,
            check_input_hash=check_input_hash,
        )


def create_lockfile_from_spec(
    *,
    conda: PathLike,
    spec: LockSpecification,
    platforms: Optional[List[str]] = None,
    lockfile_path: pathlib.Path,
    update_spec: Optional[UpdateSpecification] = None,
    metadata_choices: AbstractSet[MetadataOption] = frozenset(),
    metadata_yamls: Sequence[pathlib.Path] = (),
    strip_auth: bool = False,
) -> Lockfile:
    """
    Solve or update specification
    """
    if platforms is None:
        platforms = []
    assert spec.virtual_package_repo is not None
    virtual_package_channel = spec.virtual_package_repo.channel

    locked: Dict[Tuple[str, str, str], LockedDependency] = {}

    for platform in platforms or spec.platforms:
        deps = _solve_for_arch(
            conda=conda,
            spec=spec,
            platform=platform,
            channels=[*spec.channels, virtual_package_channel],
            update_spec=update_spec,
            strip_auth=strip_auth,
        )

        for dep in deps:
            locked[(dep.manager, dep.name, dep.platform)] = dep

    meta_sources: Dict[str, pathlib.Path] = {}
    for source in spec.sources:
        try:
            path = relative_path(lockfile_path.parent, source)
        except ValueError as e:
            if "Paths don't have the same drive" not in str(e):
                raise e
            path = str(source.resolve())
        meta_sources[path] = source

    if MetadataOption.TimeStamp in metadata_choices:
        time_metadata = TimeMeta.create()
    else:
        time_metadata = None

    if metadata_choices & {
        MetadataOption.GitUserEmail,
        MetadataOption.GitUserName,
        MetadataOption.GitSha,
    }:
        if not importlib.util.find_spec("git"):
            raise RuntimeError(
                "The GitPython package is required to read Git metadata."
            )
        git_metadata = GitMeta.create(
            metadata_choices=metadata_choices,
            src_files=spec.sources,
        )
    else:
        git_metadata = None

    if metadata_choices & {MetadataOption.InputSha, MetadataOption.InputMd5}:
        inputs_metadata: Optional[Dict[str, InputMeta]] = {
            meta_src: InputMeta.create(
                metadata_choices=metadata_choices, src_file=src_file
            )
            for meta_src, src_file in meta_sources.items()
        }
    else:
        inputs_metadata = None

    custom_metadata = get_custom_metadata(metadata_yamls=metadata_yamls)

    return Lockfile(
        package=[locked[k] for k in locked],
        metadata=LockMeta(
            content_hash=spec.content_hash(),
            channels=[c for c in spec.channels],
            platforms=spec.platforms,
            sources=list(meta_sources.keys()),
            git_metadata=git_metadata,
            time_metadata=time_metadata,
            inputs_metadata=inputs_metadata,
            custom_metadata=custom_metadata,
        ),
    )


def do_render(
    lockfile: Lockfile,
    kinds: Sequence[Union[Literal["env"], Literal["explicit"]]],
    include_dev_dependencies: bool = True,
    filename_template: Optional[str] = None,
    extras: Optional[AbstractSet[str]] = None,
    check_input_hash: bool = False,
    override_platform: Optional[Sequence[str]] = None,
) -> None:
    """Render the lock content for each platform in lockfile

    Parameters
    ----------
    lockfile :
        Lock content
    kinds :
        Lockfile formats to render
    include_dev_dependencies :
        Include development dependencies in output
    filename_template :
        Format for the lock file names. Must include {platform}.
    extras :
        Include the given extras in output
    check_input_hash :
        Do not re-render if specifications are unchanged
    override_platform :
        Generate only this subset of the platform files
    """
    platforms = lockfile.metadata.platforms
    if override_platform is not None and len(override_platform) > 0:
        platforms = list(sorted(set(platforms) & set(override_platform)))

    if filename_template:
        if "{platform}" not in filename_template and len(platforms) > 1:
            print(
                "{platform} must be in filename template when locking"
                f" more than one platform: {', '.join(platforms)}",
                file=sys.stderr,
            )
            sys.exit(1)
        for kind, file_ext in KIND_FILE_EXT.items():
            if file_ext and filename_template.endswith(file_ext):
                print(
                    f"Filename template must not end with '{file_ext}', as this "
                    f"is reserved for '{kind}' lock files, in which case it is "
                    f"automatically added."
                )
                sys.exit(1)

    for plat in platforms:
        for kind in kinds:
            if filename_template:
                context = {
                    "platform": plat,
                    "dev-dependencies": str(include_dev_dependencies).lower(),
                    "input-hash": lockfile.metadata.content_hash,
                    "version": distribution("conda_lock").version,
                    "timestamp": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
                }

                filename = filename_template.format(**context)
            else:
                filename = f"conda-{plat}.lock"

            if pathlib.Path(filename).exists() and check_input_hash:
                with open(filename) as f:
                    previous_hash = extract_input_hash(f.read())
                if previous_hash == lockfile.metadata.content_hash.get(plat):
                    print(
                        f"Lock content already rendered for {plat}. Skipping render of {filename}.",
                        file=sys.stderr,
                    )
                    continue

            print(f"Rendering lockfile(s) for {plat}...", file=sys.stderr)
            lockfile_contents = render_lockfile_for_platform(
                lockfile=lockfile,
                include_dev_dependencies=include_dev_dependencies,
                extras=extras,
                kind=kind,
                platform=plat,
            )

            filename += KIND_FILE_EXT[kind]
            with open(filename, "w") as fo:
                fo.write("\n".join(lockfile_contents) + "\n")

            print(
                f" - Install lock using {'(see warning below)' if kind == 'env' else ''}:",
                KIND_USE_TEXT[kind].format(lockfile=filename),
                file=sys.stderr,
            )

    if "env" in kinds:
        print(
            "\nWARNING: Using environment lock files (*.yml) does NOT guarantee "
            "that generated environments will be identical over time, since the "
            "dependency resolver is re-run every time and changes in repository "
            "metadata or resolver logic may cause variation. Conversely, since "
            "the resolver is run every time, the resulting packages ARE "
            "guaranteed to be seen by conda as being in a consistent state. This "
            "makes them useful when updating existing environments.",
            file=sys.stderr,
        )


def run_lock(
    environment_files: List[pathlib.Path],
    *,
    conda_exe: Optional[PathLike],
    platforms: Optional[List[str]] = None,
    mamba: bool = False,
    micromamba: bool = False,
    include_dev_dependencies: bool = True,
    channel_overrides: Optional[Sequence[str]] = None,
    filename_template: Optional[str] = None,
    kinds: Optional[Sequence[TKindAll]] = None,
    lockfile_path: pathlib.Path = pathlib.Path(DEFAULT_LOCKFILE_NAME),
    check_input_hash: bool = False,
    extras: Optional[AbstractSet[str]] = None,
    virtual_package_spec: Optional[pathlib.Path] = None,
    with_cuda: Optional[str] = None,
    update: Optional[List[str]] = None,
    filter_categories: bool = False,
    metadata_choices: AbstractSet[MetadataOption] = frozenset(),
    metadata_yamls: Sequence[pathlib.Path] = (),
    strip_auth: bool = False,
) -> None:
    if environment_files == DEFAULT_SOURCE_FILES:
        if lockfile_path.exists():
            lock_content = parse_conda_lock_file(lockfile_path)
            # reconstruct native paths
            locked_environment_files = [
                pathlib.Path(p)
                # absolute paths could be locked for both flavours
                if pathlib.PurePosixPath(p).is_absolute()
                or pathlib.PureWindowsPath(p).is_absolute()
                else pathlib.Path(
                    pathlib.PurePosixPath(lockfile_path).parent
                    / pathlib.PurePosixPath(p)
                )
                for p in lock_content.metadata.sources
            ]
            if all(p.exists() for p in locked_environment_files):
                environment_files = locked_environment_files
            else:
                missing = [p for p in locked_environment_files if not p.exists()]
                print(
                    f"{lockfile_path} was created from {[str(p) for p in locked_environment_files]},"
                    f" but some files ({[str(p) for p in missing]}) do not exist. Falling back to"
                    f" {[str(p) for p in environment_files]}.",
                    file=sys.stderr,
                )
        else:
            long_ext_file = pathlib.Path("environment.yaml")
            if long_ext_file.exists() and not environment_files[0].exists():
                environment_files = [long_ext_file]

    _conda_exe = determine_conda_executable(
        conda_exe, mamba=mamba, micromamba=micromamba
    )
    make_lock_files(
        conda=_conda_exe,
        src_files=environment_files,
        platform_overrides=platforms,
        channel_overrides=channel_overrides,
        virtual_package_spec=virtual_package_spec,
        with_cuda=with_cuda,
        update=update,
        kinds=kinds or DEFAULT_KINDS,
        lockfile_path=lockfile_path,
        filename_template=filename_template,
        include_dev_dependencies=include_dev_dependencies,
        extras=extras,
        check_input_hash=check_input_hash,
        filter_categories=filter_categories,
        metadata_choices=metadata_choices,
        metadata_yamls=metadata_yamls,
        strip_auth=strip_auth,
    )


def render_lockfile_for_platform(  # noqa: C901
    *,
    lockfile: Lockfile,
    include_dev_dependencies: bool,
    extras: Optional[AbstractSet[str]],
    kind: Union[Literal["env"], Literal["explicit"]],
    platform: str,
) -> List[str]:
    """
    Render lock content into a single-platform lockfile that can be installed
    with conda.

    Parameters
    ----------
    lockfile :
        Locked package versions
    include_dev_dependencies :
        Include development dependencies in output
    extras :
        Optional dependency groups to include in output
    kind :
        Lockfile format (explicit or env)
    platform :
        Target platform

    """
    lockfile_contents = [
        "# Generated by conda-lock.",
        f"# platform: {platform}",
        f"# input_hash: {lockfile.metadata.content_hash.get(platform)}\n",
    ]

    categories = {
        "main",
        *(extras or []),
        *(["dev"] if include_dev_dependencies else []),
    }

    conda_deps: List[LockedDependency] = []
    pip_deps: List[LockedDependency] = []

    # ensure consistent ordering of generated file
    lockfile.toposort_inplace()

    for p in lockfile.package:
        if p.platform == platform and p.category in categories:
            if p.manager == "pip":
                pip_deps.append(p)
            elif p.manager == "conda":
                # exclude virtual packages
                if not p.name.startswith("__"):
                    conda_deps.append(p)

    def format_pip_requirement(
        spec: LockedDependency, platform: str, direct: bool = False
    ) -> str:
        if spec.source and spec.source.type == "url":
            return f"{spec.name} @ {spec.source.url}"
        elif direct:
            s = f"{spec.name} @ {spec.url}"
            if spec.hash.sha256:
                s += f"#sha256={spec.hash.sha256}"
            return s
        else:
            s = f"{spec.name} === {spec.version}"
            if spec.hash.sha256:
                s += f" --hash=sha256:{spec.hash.sha256}"
            return s

    def format_conda_requirement(
        spec: LockedDependency, platform: str, direct: bool = False
    ) -> str:
        if direct:
            # inject the environment variables in here
            return posixpath.expandvars(f"{spec.url}#{spec.hash.md5}")
        else:
            path = pathlib.Path(urlsplit(spec.url).path)
            while path.suffix in {".tar", ".bz2", ".gz", ".conda"}:
                path = path.with_suffix("")
            build_string = path.name.split("-")[-1]
            return f"{spec.name}={spec.version}={build_string}"

    if kind == "env":
        lockfile_contents.extend(
            [
                "channels:",
                *(
                    f"  - {channel.env_replaced_url()}"
                    for channel in lockfile.metadata.channels
                ),
                "dependencies:",
                *(
                    f"  - {format_conda_requirement(dep, platform, direct=False)}"
                    for dep in conda_deps
                ),
            ]
        )
        lockfile_contents.extend(
            [
                "  - pip:",
                *(
                    f"    - {format_pip_requirement(dep, platform, direct=False)}"
                    for dep in pip_deps
                ),
            ]
            if pip_deps
            else []
        )
    elif kind == "explicit":
        lockfile_contents.append("@EXPLICIT\n")

        lockfile_contents.extend(
            [format_conda_requirement(dep, platform, direct=True) for dep in conda_deps]
        )

        def sanitize_lockfile_line(line: str) -> str:
            line = line.strip()
            if line == "":
                return "#"
            else:
                return line

        lockfile_contents = [sanitize_lockfile_line(line) for line in lockfile_contents]

        # emit an explicit requirements.txt, prefixed with '# pip '
        lockfile_contents.extend(
            [
                f"# pip {format_pip_requirement(dep, platform, direct=True)}"
                for dep in pip_deps
            ]
        )

        if len(pip_deps) > 0:
            logger.warning(
                "WARNING: installation of pip dependencies is only supported by the "
                "'conda-lock install' and 'micromamba install' commands. Other tools "
                "may silently ignore them. For portability, we recommend using the "
                "newer unified lockfile format (i.e. removing the --kind=explicit "
                "argument."
            )
    else:
        raise ValueError(f"Unrecognised lock kind {kind}.")

    logging.debug("lockfile_contents:\n%s\n", lockfile_contents)
    return lockfile_contents


def update_metadata(to_change: Dict[str, Any], change_source: Dict[str, Any]) -> None:
    for key in change_source:
        if key in to_change:
            logger.warning(
                f"Custom metadata field {key} provided twice, overwriting value "
                + f"{to_change[key]} with {change_source[key]}"
            )
    to_change.update(change_source)


def convert_structured_metadata_yaml(in_path: pathlib.Path) -> Dict[str, Any]:
    with in_path.open("r") as infile:
        metadata = yaml.safe_load(infile)
    return metadata


def get_custom_metadata(
    metadata_yamls: Sequence[pathlib.Path],
) -> Optional[Dict[str, str]]:
    custom_metadata_dict: Dict[str, str] = {}
    for yaml_path in metadata_yamls:
        new_metadata = convert_structured_metadata_yaml(yaml_path)
        update_metadata(custom_metadata_dict, new_metadata)
    if custom_metadata_dict:
        return custom_metadata_dict
    return None


def _solve_for_arch(
    conda: PathLike,
    spec: LockSpecification,
    platform: str,
    channels: List[Channel],
    update_spec: Optional[UpdateSpecification] = None,
    strip_auth: bool = False,
) -> List[LockedDependency]:
    """
    Solve specification for a single platform
    """
    if update_spec is None:
        update_spec = UpdateSpecification()

    dependencies = spec.dependencies[platform]
    locked = [dep for dep in update_spec.locked if dep.platform == platform]
    requested_deps_by_name = {
        manager: {dep.name: dep for dep in dependencies if dep.manager == manager}
        for manager in ("conda", "pip")
    }
    locked_deps_by_name = {
        manager: {dep.name: dep for dep in locked if dep.manager == manager}
        for manager in ("conda", "pip")
    }

    conda_deps = solve_conda(
        conda,
        specs=requested_deps_by_name["conda"],
        locked=locked_deps_by_name["conda"],
        update=update_spec.update,
        platform=platform,
        channels=channels,
    )

    if requested_deps_by_name["pip"]:
        if "python" not in conda_deps:
            raise ValueError("Got pip specs without Python")
        pip_deps = solve_pypi(
            requested_deps_by_name["pip"],
            use_latest=update_spec.update,
            pip_locked={
                dep.name: dep for dep in update_spec.locked if dep.manager == "pip"
            },
            conda_locked={dep.name: dep for dep in conda_deps.values()},
            python_version=conda_deps["python"].version,
            platform=platform,
            allow_pypi_requests=spec.allow_pypi_requests,
            strip_auth=strip_auth,
        )
    else:
        pip_deps = {}

    return list(conda_deps.values()) + list(pip_deps.values())


def _extract_spec_hash(line: str) -> Optional[str]:
    search = INPUT_HASH_PATTERN.search(line)
    if search:
        return search.group(1)
    return None


def extract_input_hash(lockfile_contents: str) -> Optional[str]:
    for line in lockfile_contents.strip().split("\n"):
        platform = _extract_spec_hash(line)
        if platform:
            return platform
    return None
