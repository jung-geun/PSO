from pathlib import Path
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pso2keras")
except PackageNotFoundError:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomllib  # type: ignore
            except ImportError:
                import tomli as tomllib  # type: ignore

        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
            __version__ = data["project"]["version"]
    else:
        __version__ = "0.0.0"
