"""Experimental: Query geospatial metadata (CRS, GSD) from thermal GeoTIFF files. Requires rasterio."""

# import os
# import rasterio
# from pathlib import Path

# # Path to folder containing GeoTIFFs
# # Folder containing thermal TIFFs — this directory is inside your package
# thermal_dir = Path(__file__).parent.parent.parent / "data" / "thermal_tif"

# if not thermal_dir.exists():
#     raise FileNotFoundError(f"Folder not found: {thermal_dir}")

# tif_files = sorted(thermal_dir.glob("*.tif"))

# if not tif_files:
#     print("[WARNING] No .tif files found in", thermal_dir)

# # Find first .tif file in folder
# tif_files = [f for f in os.listdir(thermal_dir) if f.lower().endswith(".tif")]

# if not tif_files:
#     raise FileNotFoundError("No .tif files found in the folder.")

# tif_path = os.path.join(thermal_dir, tif_files[0])

# # Open GeoTIFF
# with rasterio.open(tif_path) as dataset:
#     transform = dataset.transform
#     crs = dataset.crs

#     # Pixel size (GSD)
#     gsd_x = abs(transform.a)   # meters per pixel (x direction)
#     gsd_y = abs(transform.e)   # meters per pixel (y direction)

# print(f"File: {tif_files[0]}")
# print(f"CRS: {crs}")
# print(f"GSD X: {gsd_x:.4f} m/pixel")
# print(f"GSD Y: {gsd_y:.4f} m/pixel")

# from pathlib import Path
# import rasterio

# thermal_dir = Path(__file__).parent.parent.parent / "data" / "thermal_tif"
# tif_path = sorted(thermal_dir.glob("*.tif"))[0]

# with rasterio.open(tif_path) as ds:
#     print("FILE:", tif_path.name)
#     print("CRS:", ds.crs)
#     print("TRANSFORM:", ds.transform)
#     print("RES (ds.res):", ds.res)  # (pixel width, pixel height) in CRS units
#     print("WIDTH x HEIGHT:", ds.width, "x", ds.height)
#     print("BOUNDS:", ds.bounds)
#     print("IS GEOGRAPHIC CRS:", bool(ds.crs and ds.crs.is_geographic))

#     # "GSD" assuming CRS units are meters
#     gsd_x, gsd_y = ds.res
#     print("GSD (raw units):", abs(gsd_x), abs(gsd_y), "(CRS units / pixel)")

from pathlib import Path
import rasterio
from rasterio.errors import RasterioIOError

def is_identity_transform(t):
    # Common "not georeferenced" fallback: identity or near-identity
    return (
        t.a == 1.0 and t.b == 0.0 and t.c == 0.0 and
        t.d == 0.0 and t.e == 1.0 and t.f == 0.0
    )

thermal_dir = Path(__file__).parent.parent.parent / "data" / "thermal_tif"
tifs = sorted(thermal_dir.glob("*.tif"))

if not tifs:
    raise FileNotFoundError(f"No .tif files found in {thermal_dir}")

print(f"Scanning {len(tifs)} TIFF(s) in: {thermal_dir}\n")

georef = []
not_georef = []
errors = []

for p in tifs:
    try:
        with rasterio.open(p) as ds:
            crs = ds.crs
            transform = ds.transform
            res = ds.res  # (pixel width, pixel height) in CRS units
            has_gcps = ds.gcps[0] is not None and len(ds.gcps[0]) > 0
            has_rpcs = ds.rpcs is not None

            # Consider georeferenced if any spatial reference exists:
            # - CRS + non-identity transform
            # - OR has GCPs
            # - OR has RPCs
            georef_ok = (
                (crs is not None and not is_identity_transform(transform)) or
                has_gcps or
                has_rpcs
            )

            if georef_ok:
                georef.append((p.name, crs, transform, res, has_gcps, has_rpcs))
            else:
                not_georef.append((p.name, crs, transform, res, has_gcps, has_rpcs))

    except RasterioIOError as e:
        errors.append((p.name, str(e)))

# Report
print("=== Georeferenced TIFFs ===")
if not georef:
    print("None found.\n")
else:
    for name, crs, transform, res, has_gcps, has_rpcs in georef:
        print(f"- {name}")
        print(f"  CRS: {crs}")
        print(f"  RES (CRS units/px): {res}")
        print(f"  Has GCPs: {has_gcps} | Has RPCs: {has_rpcs}")
        print(f"  Transform: {transform}\n")

print("=== Not georeferenced (identity/no CRS) ===")
print(f"{len(not_georef)} file(s)")
for name, crs, transform, res, has_gcps, has_rpcs in not_georef[:20]:
    print(f"- {name} | CRS={crs} | RES={res} | GCPs={has_gcps} | RPCs={has_rpcs}")
if len(not_georef) > 20:
    print(f"... and {len(not_georef) - 20} more\n")

if errors:
    print("\n=== Errors opening files ===")
    for name, msg in errors:
        print(f"- {name}: {msg}")
