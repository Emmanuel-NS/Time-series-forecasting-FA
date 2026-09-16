"""Geographic interpretation of the Milan grid.

The activity store indexes areas by their integer ``square_id`` (1..10000). On
its own that is enough to forecast, but not enough to *explain* anything: the
assignment asks for observed patterns to be connected to real-world causes, and
that requires knowing where each area actually is.

The official cell geometry is published as a separate file in the companion
Dataverse dataset (``doi:10.7910/DVN/QJWLFU``). This module downloads it once,
reduces the 10,000 polygons to centroids, and caches them as two small arrays.

Verified grid convention
------------------------
Centroid latitude correlates with ``(square_id - 1) // 100`` at r = 1.000 and
centroid longitude with ``(square_id - 1) % 100`` at r = 1.000. Therefore

    row = (square_id - 1) // 100      # increases northwards
    col = (square_id - 1) % 100       # increases eastwards

and ``totals.reshape(100, 100)`` displayed with ``origin="lower"`` is a
geographically correct, north-up map. This was checked rather than assumed,
because a silent vertical flip would invert every spatial interpretation.
"""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache

import numpy as np

from . import config as C

#: Companion Dataverse dataset holding ``milano-grid.geojson``.
GRID_FILE_ID = 2670806
GRID_GEOJSON = C.PROCESSED_DIR / "milano-grid.geojson"
CENTROIDS_CACHE = C.PROCESSED_DIR / "grid_centroids.npy"

#: Reference points used to describe where an area is, in decimal degrees.
LANDMARKS = {
    "Duomo (historic centre)": (45.4642, 9.1900),
    "Milano Centrale station": (45.4863, 9.2049),
    "Bocconi University": (45.4478, 9.1889),
    "Politecnico (Leonardo)": (45.4787, 9.2272),
    "San Siro stadium": (45.4781, 9.1240),
    "Linate airport": (45.4451, 9.2767),
    "Navigli nightlife district": (45.4509, 9.1743),
    "Fiera Milano City / Portello": (45.4800, 9.1550),
}


def download_grid(force: bool = False) -> "None | str":
    """Fetch ``milano-grid.geojson`` using the guestbook-signed download flow."""
    if GRID_GEOJSON.exists() and not force:
        return str(GRID_GEOJSON)

    from .ingest import _SignedUrlPool, _request

    url = _SignedUrlPool(GRID_FILE_ID).get()
    with urllib.request.urlopen(_request(url), timeout=300) as resp:
        GRID_GEOJSON.write_bytes(resp.read())
    return str(GRID_GEOJSON)


@lru_cache(maxsize=1)
def centroids() -> np.ndarray:
    """``(10000, 2)`` array of ``(longitude, latitude)`` centroids, row i = cell i+1."""
    if CENTROIDS_CACHE.exists():
        return np.load(CENTROIDS_CACHE)

    download_grid()
    payload = json.loads(GRID_GEOJSON.read_text(encoding="utf-8"))

    out = np.full((C.N_SQUARES, 2), np.nan, dtype=np.float64)
    for feature in payload["features"]:
        cell = int(feature["properties"]["cellId"])
        ring = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
        out[cell - 1] = (ring[:, 0].mean(), ring[:, 1].mean())

    if np.isnan(out).any():
        raise ValueError("grid geometry is incomplete")
    np.save(CENTROIDS_CACHE, out)
    return out


def available() -> bool:
    """True if the geometry can be used without a network call."""
    return CENTROIDS_CACHE.exists() or GRID_GEOJSON.exists()


def row_col(square_id: int) -> tuple[int, int]:
    """Grid ``(row, col)`` of an area; row increases north, col increases east."""
    index = square_id - 1
    return index // C.GRID_SIDE, index % C.GRID_SIDE


def square_id_of(row: int, col: int) -> int:
    return row * C.GRID_SIDE + col + 1


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def describe(square_id: int) -> dict:
    """Location summary for one area: coordinates and the nearest landmark."""
    lon, lat = centroids()[square_id - 1]
    row, col = row_col(square_id)
    distances = {
        name: _haversine_km(lat, lon, plat, plon)
        for name, (plat, plon) in LANDMARKS.items()
    }
    nearest = min(distances, key=distances.get)
    return {
        "square_id": square_id,
        "row": row,
        "col": col,
        "latitude": round(float(lat), 5),
        "longitude": round(float(lon), 5),
        "nearest_landmark": nearest,
        "distance_km": round(distances[nearest], 2),
    }


def describe_many(square_ids) -> "list[dict]":
    return [describe(int(s)) for s in square_ids]
