"""Add MIT license header to all src/ Python files."""
from pathlib import Path

MIT_HEADER = "# Copyright (c) 2026 Santhosh -- MIT License\n# Part of Yaazhi GeoAlign OS / ISRO SIH26166\n"

src_dir = Path(__file__).resolve().parent / "src"
modified = []
for py_file in src_dir.rglob("*.py"):
    content = py_file.read_text(encoding="utf-8")
    if "Copyright (c) 2026" not in content:
        new_content = MIT_HEADER + content
        py_file.write_text(new_content, encoding="utf-8")
        modified.append(str(py_file))

print(f"Patched {len(modified)} files:")
for f in modified:
    print(f"  {f}")
