from __future__ import annotations

from pathlib import Path
from typing import Iterable


def resolve_allowed_path(path_value: str | Path, base_dir: Path, allowed_roots: Iterable[Path]) -> Path:
    """
    Resolve and validate a user-provided path against an allowlist of directories.
    """
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    for root in allowed_roots:
        root_resolved = root.resolve()
        try:
            candidate.relative_to(root_resolved)
            return candidate
        except ValueError:
            continue

    raise PermissionError(f"Path is outside allowed directories: {candidate}")
