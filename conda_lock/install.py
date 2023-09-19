import pathlib

from functools import partial

from conda_lock.common import temporary_file_with_contents
from conda_lock.invoke_conda import PathLike, _invoke_conda, is_micromamba


def do_conda_install(
    conda: PathLike,
    prefix: "str | None",
    name: "str | None",
    file: pathlib.Path,
    copy: bool,
) -> None:
    _conda = partial(_invoke_conda, conda, prefix, name, check_call=True)

    kind = "env" if file.name.endswith(".yml") else "explicit"

    if kind == "explicit":
        with open(file) as explicit_env:
            pip_requirements = [
                line.split("# pip ")[1]
                for line in explicit_env
                if line.startswith("# pip ")
            ]
    else:
        pip_requirements = []

    env_prefix = ["env"] if kind == "env" and not is_micromamba(conda) else []
    copy_arg = ["--copy"] if kind != "env" and copy else []
    yes_arg = ["--yes"] if kind != "env" else []

    _conda(
        [
            *env_prefix,
            "create",
            *copy_arg,
            "--file",
            str(file),
            *yes_arg,
        ],
    )

    if not pip_requirements:
        return

    with temporary_file_with_contents("\n".join(pip_requirements)) as requirements_path:
        _conda(["run"], ["pip", "install", "--no-deps", "-r", str(requirements_path)])
