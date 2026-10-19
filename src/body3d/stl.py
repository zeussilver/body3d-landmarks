"""Minimal STL reader/writer (binary and ASCII), numpy only.

Binary layout: 80-byte header, uint32 triangle count, then per triangle 12 little-endian
float32 (normal, v1, v2, v3) and a uint16 attribute. A file is treated as binary when
its size matches that layout exactly, otherwise parsed as ASCII (some binary headers
also start with "solid").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DTYPE = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def face_normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def write(path, vertices: np.ndarray, faces: np.ndarray, header: str = "", ascii: bool = False) -> None:
    tris = np.asarray(vertices, np.float64)[faces]
    normals = face_normals(tris)
    path = Path(path)
    if ascii:
        lines = [f"solid {header}"]
        for n, t in zip(normals, tris):
            lines += [f" facet normal {n[0]:e} {n[1]:e} {n[2]:e}", "  outer loop"]
            lines += [f"   vertex {p[0]:e} {p[1]:e} {p[2]:e}" for p in t]
            lines += ["  endloop", " endfacet"]
        path.write_text("\n".join(lines + [f"endsolid {header}"]) + "\n")
        return
    rec = np.zeros(len(tris), _DTYPE)
    rec["normal"], rec["v"] = normals, tris
    with path.open("wb") as fh:
        fh.write(header.encode("ascii", "replace")[:80].ljust(80, b" "))
        fh.write(np.uint32(len(tris)).tobytes())
        fh.write(rec.tobytes())


def read_triangles(path) -> np.ndarray:
    """(F, 3, 3) float64 triangle corners."""
    raw = Path(path).read_bytes()
    if len(raw) >= 84:
        n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
        if len(raw) == 84 + n * _DTYPE.itemsize:
            return np.frombuffer(raw, _DTYPE, n, 84)["v"].astype(np.float64)
    text = raw.decode("ascii", "replace")
    if not text.lstrip().startswith("solid"):
        raise ValueError(f"{path}: neither binary STL (size mismatch) nor ASCII STL")
    verts = [line.split()[1:4] for line in text.splitlines() if line.strip().startswith("vertex")]
    if len(verts) % 3:
        raise ValueError(f"{path}: ASCII STL has {len(verts)} vertices, not a multiple of 3")
    return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


def to_indexed(tris: np.ndarray, decimals: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Weld identical corners (after rounding) into an indexed mesh: vertices (V, 3), faces (F, 3)."""
    flat = tris.reshape(-1, 3)
    _, first, inverse = np.unique(np.round(flat, decimals), axis=0, return_index=True, return_inverse=True)
    return flat[first], inverse.reshape(-1, 3)


def read(path) -> tuple[np.ndarray, np.ndarray]:
    """Indexed mesh (vertices, faces) from a binary or ASCII STL."""
    return to_indexed(read_triangles(path))
