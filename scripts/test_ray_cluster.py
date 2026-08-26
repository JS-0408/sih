"""
scripts/test_ray_cluster.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ray cluster diagnostic script.

Initialises Ray and prints a formatted status table of all connected
nodes, showing IP, CPU/GPU resources, and alive status.

Exit codes:
  0 — At least one node detected (cluster healthy).
  1 — No nodes detected or Ray initialisation failed.

Usage:
  # Local single-node test:
  python scripts/test_ray_cluster.py

  # Connect to existing head node:
  RAY_HEAD=192.168.1.10 python scripts/test_ray_cluster.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from datetime import datetime

import ray


def _format_table(nodes: list[dict]) -> str:
    """Render a fixed-width ASCII table of node information."""
    headers = ["#", "Node IP", "CPUs", "GPUs", "Mem (GB)", "Alive", "Object Store (MB)"]
    rows = []
    for i, n in enumerate(nodes, start=1):
        resources = n.get("Resources", {})
        rows.append([
            str(i),
            n.get("NodeManagerAddress", "unknown"),
            str(int(resources.get("CPU", 0))),
            str(int(resources.get("GPU", 0))),
            f"{resources.get('memory', 0) / 1e9:.1f}",
            "✓" if n.get("Alive", False) else "✗",
            f"{n.get('ObjectStoreAvailableMemory', 0) / 1e6:.0f}",
        ])

    col_widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    lines = [
        sep,
        fmt.format(*headers),
        sep,
        *[fmt.format(*r) for r in rows],
        sep,
    ]
    return "\n".join(lines)


def main() -> int:
    head_ip: str = os.environ.get("RAY_HEAD", "auto")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("  Ray Cluster Diagnostic — Phase 1 Pipeline")
    print(f"  Timestamp : {timestamp}")
    print(f"  Head addr : {head_ip}")
    print("=" * 60)

    try:
        if ray.is_initialized():
            ray.shutdown()

        ray.init(
            address=head_ip,
            ignore_reinit_error=True,
            include_dashboard=False,
            logging_level=40,  # ERROR only — suppress Ray INFO spam
        )
        print(f"\n✓ Ray initialised  (version {ray.__version__})")
    except ConnectionError as exc:
        print(f"\n✗ Connection failed: {exc}")
        print(textwrap.dedent("""
            Troubleshooting:
              1. Start the head node:  ray start --head --port=6379
              2. Join worker node:     ray start --address='<HEAD_IP>:6379'
              3. Set env var:          export RAY_HEAD=<HEAD_IP>:6379
        """))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ Unexpected Ray error: {exc}")
        return 1

    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get("Alive", False)]

    print(f"\n  Total nodes : {len(nodes)}")
    print(f"  Alive nodes : {len(alive_nodes)}\n")

    if nodes:
        print(_format_table(nodes))
    else:
        print("  [!] No nodes found in cluster.")

    # Cluster-wide resource summary
    total_resources = ray.cluster_resources()
    avail_resources = ray.available_resources()
    print("\n  Cluster Resources (Total / Available)")
    print("  " + "-" * 38)
    for key in sorted(total_resources):
        total = total_resources[key]
        avail = avail_resources.get(key, 0.0)
        print(f"  {key:<20}: {total:>6.1f} / {avail:>6.1f}")

    print("\n" + "=" * 60)
    if alive_nodes:
        print(f"  STATUS: ✓ HEALTHY — {len(alive_nodes)} node(s) connected")
        print("=" * 60 + "\n")
        ray.shutdown()
        return 0
    else:
        print("  STATUS: ✗ DEGRADED — no alive nodes detected")
        print("=" * 60 + "\n")
        ray.shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(main())
