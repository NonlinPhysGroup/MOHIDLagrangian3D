# -----------------------------
# Full Rasterization + Rugosity + KMZ Update Script with bathymetry mask
# -----------------------------
import sys
from pathlib import Path
import glob
import zipfile
import tempfile
import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import xarray as xr

# -----------------------------
# Shapefile -> NetCDF function
# -----------------------------
def Shapefile_TO_NETCDF4(shapefile, value_column, ds=None,
                         fillValue=-9999, fillValue_2=0, land_mask_shp=None):

    gdf = gpd.read_file(shapefile)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf[value_column] = gdf[value_column].astype(float)

    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS defined!")
    gdf = gdf.to_crs(epsg=4326)

    if ds is not None and "lon" in ds.coords and "lat" in ds.coords:
        lon = np.asarray(ds["lon"].values)
        lat = np.asarray(ds["lat"].values)
        lon_sorted = np.sort(lon)
        lat_sorted = np.sort(lat)
        dx = np.mean(np.diff(lon_sorted))
        dy = np.mean(np.diff(lat_sorted))
        origin_x = lon_sorted[0] - dx / 2.0
        origin_y = lat_sorted[-1] + dy / 2.0
        transform = rasterio.transform.from_origin(origin_x, origin_y, dx, dy)
        width = len(lon)
        height = len(lat)
    else:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        width = int(np.ceil((xmax - xmin) / 0.001))
        height = int(np.ceil((ymax - ymin) / 0.001))
        transform = rasterio.transform.from_origin(xmin, ymax, 0.001, 0.001)
        lon = np.linspace(xmin + 0.001/2, xmax - 0.001/2, width)
        lat = np.linspace(ymin + 0.001/2, ymax - 0.001/2, height)

    shapes_and_values = ((geom, val) for geom, val in zip(gdf.geometry, gdf[value_column]))
    raster_data = rasterize(
        shapes_and_values,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        all_touched=True,
        dtype='float32'
    )

    # Land mask
    if ds is not None and "bathymetry" in ds:
        # Make sure bathymetry matches raster_data orientation
        bathy = ds["bathymetry"].values
        # Flip lat if necessary to match raster_data (lat ascending check)
        if ds["lat"].values[0] < ds["lat"].values[-1]:
            bathy = np.flipud(bathy)
        # Apply mask
        land_mask = np.where(bathy >= 0, 0, 1)
        raster_data = np.where(np.isnan(raster_data) & (land_mask == 1), fillValue, raster_data)
        raster_data = np.where(np.isnan(raster_data) & (land_mask == 0), fillValue_2, raster_data)
    elif land_mask_shp is not None:
        land_gdf = gpd.read_file(land_mask_shp).to_crs(epsg=4326)
        land_shapes = ((geom, 1) for geom in land_gdf.geometry)
        land_mask = rasterize(
            land_shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8'
        )
        raster_data = np.where(np.isnan(raster_data) & (land_mask==1), fillValue, raster_data)
        raster_data = np.where(np.isnan(raster_data) & (land_mask==0), fillValue_2, raster_data)
    else:
        raster_data = np.where(np.isnan(raster_data), fillValue, raster_data)

    if ds is not None and "lat" in ds.coords:
        lat_arr = np.asarray(ds["lat"].values)
        if lat_arr[0] < lat_arr[-1]:
            raster_data = np.flipud(raster_data)
        lon = ds["lon"].values
        lat = ds["lat"].values
    else:
        raster_data = np.flipud(raster_data)

    da = xr.DataArray(
        raster_data,
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
        name=value_column
    )

    shapefile_folder_path = Path(shapefile).parent
    shapefile_name = Path(shapefile).stem
#    output_folder_path = Path(shapefile_folder_path) / 'output_NETCDF4'
#    output_folder_path.mkdir(parents=True, exist_ok=True)
#    output_file_path = output_folder_path / f"{shapefile_name}.nc4"

#    da.to_netcdf(
#        output_file_path,
#        format="NETCDF4",
#        encoding={value_column: {"_FillValue": fillValue}}
#    )
#    print(f"Saved NetCDF file: {output_file_path}")

    if ds is not None:
        ds[shapefile_name] = da

#    return output_file_path
    return
# -----------------------------
# Rugosity calculation
# -----------------------------
def Calculate_Rugosity(rugosity_method, output_folder, rugosity_map=None,
                        background_rugosity=0.005, use_background=True,
                        percent_in_0_1=False, fillValue_2=0,
                        fillValue=-9999,
                        template_ds=None):

    print(f'Method {rugosity_method} has used to calculated Rugosity map.')

    if template_ds is not None:
        ds = template_ds
        weighted_lin_sum = None
        weighted_pow_sum = None
        phi_weighted_sum = None
        fraction_sum = None
        rug_values = []
        phi_values = []
        fractions = []
        missing = []

        for stem, rug_val in rugosity_map.items():
            if stem not in ds:
                missing.append(stem)
                print(f"⚠️ Skipping {stem} (not in provided dataset)")
                continue

            da = ds[stem].astype("float32")
            factor = 1.0 if percent_in_0_1 else 1.0 / 100.0
            fraction = da * factor

            weighted_lin =  rug_val * fraction
            weighted_pow =  rug_val ** fraction

            # φ_i = -log2(D_i)
            phi_i = -np.log2(np.maximum(rug_val, 1e-12))
            phi_weighted = fraction * phi_i

            rug_values.append(rug_val)
            phi_values.append(phi_i)
            fractions.append(fraction)

            if fraction_sum is None:
                fraction_sum = fraction.copy(deep=True)
                weighted_lin_sum = weighted_lin.copy(deep=True)
                weighted_pow_sum = weighted_pow.copy(deep=True)
                phi_weighted_sum = phi_weighted.copy(deep=True)
            else:
                fraction_sum = fraction_sum + fraction
                weighted_lin_sum = weighted_lin_sum + weighted_lin
                weighted_pow_sum = weighted_pow_sum * weighted_pow
                phi_weighted_sum = phi_weighted_sum + phi_weighted

        if fraction_sum is None:
            print("No layers were processed (in-memory). Check rugosity_map and dataset.")
            return None

        # Compute φ_bg and missing fraction
        if use_background:
            bkg_rugosity = background_rugosity
            phi_bkg_Rugosity = -np.log2(np.maximum(background_rugosity, 1e-12))
            f_bkg = xr.where(fraction_sum < 1, 1 - fraction_sum, 0)
        else:
            bkg_rugosity = 0.0
            phi_bkg_Rugosity = 0.0
            f_bkg = xr.zeros_like(fraction_sum)

        # Compute φ̄ (include background)
        phi_bar = phi_weighted_sum + f_bkg * phi_bkg_Rugosity

        # Compute σ_φ = sqrt(Σ f_i (φ_i - φ̄)^2 + f_bg (φ_bg - φ̄)^2)
        sigma_phi_sum = f_bkg * (phi_bkg_Rugosity - phi_bar) ** 2
        for phi_i, f_i in zip(phi_values, fractions):
            sigma_phi_sum += f_i * (phi_i - phi_bar) ** 2
        #sigma_phi = np.sqrt(sigma_phi_sum)
        sigma_phi = np.sqrt(xr.where(sigma_phi_sum >= 0, sigma_phi_sum, 0))

        k_phi = xr.where((sigma_phi > 0.5) & (sigma_phi <= 1), 0.75,
                         xr.where((sigma_phi > 1.0) & (sigma_phi <= 2.0), 1.25,
                                  xr.where((sigma_phi > 2.0) & (sigma_phi <= 8.0), 2.5,
                                           xr.where(sigma_phi > 8, 3.5,
                                                    0.0))))  # default if sigma_phi <= 0.5

        k_phi = np.sqrt(xr.where(k_phi >= 0, k_phi, 0))

        # Compute D_eff = 2^{-(φ̄ - σ_φ)}
        if rugosity_method == 'Folk_ExpAvg':

            # Cap D_eff by maximum rugosity value (element-wise)
            D_eff_raw = 2 ** (-(phi_bar - k_phi * sigma_phi))
            max_rug = max(rug_values)  # scalar
            D_eff = xr.where(D_eff_raw > max_rug, max_rug, D_eff_raw)

        # Compute D_eff = D_mix (1+ k σ_φ)
        elif rugosity_method == 'Folk_LinAvg':
            D_mix = weighted_lin_sum + bkg_rugosity * f_bkg
            D_eff = D_mix * (1 + 0.25 * sigma_phi)

        # Compute D_eff = D_mix (1+ k σ_φ)
        elif rugosity_method == 'Folk_PowAvg':
            D_mix = weighted_pow_sum * bkg_rugosity ** f_bkg
            D_eff = D_mix * (1 + 0.25 * sigma_phi)
        else:
            raise ValueError('Method for calculating Rugosity is not recognized')

        # Mask handling (unchanged)
        final_rugosity = D_eff.where(fraction_sum >= 0, other=fillValue)

        ds["Intermediate_Rugosity"] = final_rugosity
        sea_mask = ds['Intermediate_Rugosity'] == fillValue_2
        final_rugosity = final_rugosity.where(~sea_mask, other=background_rugosity)

        final_rugosity.name = "Rugosity"
        ds["Rugosity"] = final_rugosity
        print("✅ Added 'Rugosity' to provided dataset (in memory).")

        # --- D50 calculation including background ---

        # Combine rugosity and fractions in lists
        rug_values_with_bkg = rug_values + [bkg_rugosity]
        fractions_with_bkg = fractions + [f_bkg]

        # Stack only non-zero layers
        rug_stack = xr.concat([xr.DataArray(r) for r in rug_values_with_bkg], dim="layer")
        frac_stack = xr.concat([xr.DataArray(f) for f in fractions_with_bkg], dim="layer")

        # Sort by rugosity ascending
        sort_idx = xr.DataArray(np.argsort(rug_stack.values), dims="layer")
        rug_sorted = rug_stack.isel(layer=sort_idx)
        frac_sorted = frac_stack.isel(layer=sort_idx)

        # Compute cumulative fraction along sorted layers
        cum_frac_sorted = frac_sorted.cumsum(dim="layer")

        # Identify first layer where cumulative fraction >= 0.5
        above_50 = cum_frac_sorted >= 0.5
        first_idx = above_50.argmax(dim="layer")
        prev_idx = xr.where(first_idx > 0, first_idx - 1, 0)

        # Vectorized adjustment of prev_idx where f2 - f1 < 1e-4
        max_layers = cum_frac_sorted.sizes["layer"]
        prev_idx_corrected = prev_idx.copy()

        for _ in range(max_layers):
            # Compute f1 for current prev_idx
            f1_current_0 = cum_frac_sorted.isel(layer=prev_idx_corrected)
            f1_current_1 = cum_frac_sorted.isel(layer=xr.where(prev_idx_corrected > 0, prev_idx_corrected - 1, 0))
            diff = f1_current_0 - f1_current_1

            # Only update prev_idx_corrected where diff < 1e-4 and prev_idx_corrected > 0
            prev_idx_corrected = xr.where((diff < 1e-4) & (prev_idx_corrected > 0),
                                          prev_idx_corrected - 1,
                                          prev_idx_corrected)

        # Recompute f1 after correction
        f1 = cum_frac_sorted.isel(layer=prev_idx_corrected)
        f2 = cum_frac_sorted.isel(layer=first_idx)
        D1 = rug_sorted.isel(layer=prev_idx_corrected)
        D2 = rug_sorted.isel(layer=first_idx)


        # Take rugosity directly if first layer fraction >= 0.5
        # Take rugosity directly if only background exists or first layer fraction >= 0.5
        D50_val = xr.where(f_bkg == 1.0, bkg_rugosity, rug_sorted.isel(layer=first_idx))

        # Interpolate only if 0.5 lies strictly between f1 and f2 and first_idx > 0
        interpolate_mask = (f1 < 0.5) & (f2 > 0.5) & (first_idx > 0) & (f_bkg != 1.0)

        # Compute D50 for interpolating cells
        if rugosity_method == 'Folk_PowAvg':
            D50_interp = np.exp(np.log(D1) + (0.5 - f1) / (f2 - f1) * (np.log(D2) - np.log(D1)))
        if rugosity_method in ['Folk_LinAvg' or 'Folk_ExpAvg']:  # Folk_LinAvg or Folk_ExpAvg
            D50_interp = D1 + (0.5 - f1) / (f2 - f1) * (D2 - D1)

        # Combine direct and interpolated
        D50_val = xr.where(interpolate_mask, D50_interp, D50_val)

        # Mask and finalize
        final_D50 = D50_val.where(fraction_sum >= 0, other=fillValue)
        final_D50 = final_D50.where(~sea_mask, other=background_rugosity)
        final_D50.name = "D50"
        ds["D50"] = final_D50

        print("✅ Added 'D50' to provided dataset (in memory).")
        if missing:
            print("Missing rugosity entries (skipped):", missing)
        return final_rugosity

    # --- File-based fallback ---
    if rugosity_map is None:
        print("No rugosity_map provided. Skipping rugosity calculation.")
        return None

    nc_folder = Path(output_folder)
    nc_files = sorted(nc_folder.glob("*.nc4"))

    weighted_lin_sum = None
    weighted_pow_sum = None
    phi_weighted_sum = None
    fraction_sum = None
    rug_values = []
    phi_values = []
    fractions = []
    missing = []

    for nc in nc_files:
        stem = nc.stem
        if stem not in rugosity_map:
            missing.append(stem)
            print(f"⚠️ Skipping {stem} (no rugosity value provided)")
            continue

        ds_file = xr.open_dataset(nc)
        varnames = list(ds_file.data_vars)
        da = ds_file[varnames[0]].astype("float32")
        ds_file.close()

        factor = 1.0 if percent_in_0_1 else 1.0 / 100.0
        fraction = da * factor

        weighted_lin = rugosity_map[stem] * fraction
        weighted_pow = rugosity_map[stem] ** fraction

        phi_i = -np.log2(np.maximum(rugosity_map[stem], 1e-12))
        phi_weighted = fraction * phi_i

        rug_values.append(rugosity_map[stem])
        phi_values.append(phi_i)
        fractions.append(fraction)

        if fraction_sum is None:
            fraction_sum = fraction.copy(deep=True)
            weighted_lin_sum = weighted_lin.copy(deep=True)
            weighted_pow_sum = weighted_pow.copy(deep=True)
            phi_weighted_sum = phi_weighted.copy(deep=True)
        else:
            fraction_sum = fraction_sum + fraction
            weighted_lin_sum = weighted_lin_sum + weighted_lin
            weighted_pow_sum = weighted_pow_sum + weighted_pow
            phi_weighted_sum = phi_weighted_sum + phi_weighted

    if fraction_sum is None:
        print("No layers were processed. Check rugosity_map and input files.")
        return None

    # Include background φ
    if use_background:
        bkg_rugosity = background_rugosity
        phi_bkg_Rugosity = -np.log2(np.maximum(background_rugosity, 1e-12))
        f_bkg = xr.where(fraction_sum < 1, 1 - fraction_sum, 0)
    else:
        bkg_rugosity = 0.0
        phi_bkg_Rugosity = 0.0
        f_bkg = xr.zeros_like(fraction_sum)

    # Compute φ̄
    phi_bar = phi_weighted_sum + f_bkg * phi_bkg_Rugosity

    # Compute σ_φ
    sigma_phi_sum = f_bkg * (phi_bkg_Rugosity - phi_bar) ** 2
    for phi_i, f_i in zip(phi_values, fractions):
        sigma_phi_sum += f_i * (phi_i - phi_bar) ** 2
    #sigma_phi = np.sqrt(sigma_phi_sum)
    sigma_phi = np.sqrt(xr.where(sigma_phi_sum >= 0, sigma_phi_sum, 0))

    k_phi = xr.where((sigma_phi > 0.5) & (sigma_phi <= 1), 0.75,
                     xr.where((sigma_phi > 1.0) & (sigma_phi <= 2.0), 1.25,
                              xr.where((sigma_phi > 2.0) & (sigma_phi <= 8.0), 2.5,
                                       xr.where(sigma_phi > 8, 3.5,
                                                0.0))))  # default if sigma_phi <= 0.5

    k_phi = np.sqrt(xr.where(k_phi >= 0, k_phi, 0))

    # Compute D_eff = 2^{-(φ̄ - σ_φ)}
    if rugosity_method == 'Folk_ExpAvg':
        # Cap D_eff by maximum rugosity value (element-wise)
        D_eff_raw = 2 ** (-(phi_bar - k_phi * sigma_phi))
        max_rug = max(rug_values)  # scalar
        D_eff = xr.where(D_eff_raw > max_rug, max_rug, D_eff_raw)

    # Compute D_eff = D_mix (1+ k σ_φ)
    elif rugosity_method == 'Folk_LinAvg':
        D_mix = weighted_lin_sum + bkg_rugosity * f_bkg
        D_eff = D_mix * (1 + 0.25 * sigma_phi)

    # Compute D_eff = D_mix (1+ k σ_φ)
    elif rugosity_method == 'Folk_PowAvg':
        D_mix = weighted_pow_sum * bkg_rugosity ** f_bkg
        D_eff = D_mix * (1 + 0.25 * sigma_phi)
    else:
        raise ValueError('Method for calculating Rugosity is not recognized')


    # Mask handling (unchanged)
    final_rugosity = D_eff.where(fraction_sum >= 0, other=fillValue)

    # Apply land/sea mask
    sample_file = nc_files[0]
    ds_sample = xr.open_dataset(sample_file)
    varname_sample = list(ds_sample.data_vars)[0]
    sea_mask = ds_sample[varname_sample] == fillValue_2
    ds_sample.close()
    final_rugosity = final_rugosity.where(~sea_mask, other=background_rugosity)

    final_rugosity.name = "Rugosity"
    output_file = nc_folder / "final_rugosity.nc4"
    final_rugosity.to_netcdf(
        output_file,
        format="NETCDF4",
        encoding={"Rugosity": {"_FillValue": fillValue}}
    )
    print(f"✅ Saved final Rugosity: {output_file}")
    if missing:
        print("Missing rugosity entries (skipped):", missing)
    return final_rugosity


# -----------------------------
# Parse KML/ KMZ polygons
# -----------------------------
def parse_kml_polygons(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    polygons = []
    for placemark in root.findall(".//Placemark", ns):
        for polygon in placemark.findall(".//Polygon", ns):
            coords_text = polygon.find(".//coordinates", ns).text
            if not coords_text:
                continue
            coords = []
            for line in coords_text.split():
                line = line.strip()
                if not line:
                    continue
                parts = [x for x in line.split(",") if x.strip() != ""]
                if len(parts) < 2:
                    continue
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            if len(coords) >= 3:
                polygons.append(Polygon(coords))
    return polygons

# -----------------------------
# Update Rugosity from KMZ polygons
# -----------------------------
def KMZ_UPDATE_RUGOSITY_POINTWISE(ds, kmz_folder, rugosity_map=None, default_rugosity=0.005):
    if "Rugosity" not in ds:
        print("⚠️  'Rugosity' variable not found — skipping KMZ rugosity update.")
        return ds  # exit gracefully, no crash

    lon_arr = np.asarray(ds["lon"].values)
    lat_arr = np.asarray(ds["lat"].values)
    nlat, nlon = len(lat_arr), len(lon_arr)

    mask = np.zeros((nlat, nlon), dtype=np.uint8)
    polygon_rug_list = []

    kmz_folder = Path(kmz_folder)
    kmz_files = sorted(list(kmz_folder.glob("*.kmz")) + list(kmz_folder.glob("*.kml")))

    for kmz_path in kmz_files:
        polygons = []
        if kmz_path.suffix.lower() == ".kmz":
            try:
                with zipfile.ZipFile(kmz_path, 'r') as kmz:
                    kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
                    if kml_files:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            kmz.extractall(tmpdir)
                            for root_dir, _, files in os.walk(tmpdir):
                                for f in files:
                                    if f.endswith(".kml"):
                                        polygons.extend(parse_kml_polygons(Path(root_dir)/f))
            except zipfile.BadZipFile:
                polygons.extend(parse_kml_polygons(kmz_path))
        else:
            polygons.extend(parse_kml_polygons(kmz_path))

        stem = kmz_path.stem
        for poly in polygons:
            rug_val_for_poly = rugosity_map.get(stem, None) if rugosity_map else None
            polygon_rug_list.append((poly, rug_val_for_poly))

    # fill mask
    for i, lat in enumerate(lat_arr):
        for j, lon in enumerate(lon_arr):
            pt = Point(lon, lat)
            for poly, _ in polygon_rug_list:
                if poly.contains(pt):
                    mask[i, j] = 1
                    break

    # Update Rugosity
    rug_values = ds["Rugosity"].values.copy()
    D50_values = ds["D50"].values.copy()
    for i in range(nlat):
        for j in range(nlon):
            if mask[i, j] == 1:
                pt = Point(lon_arr[j], lat_arr[i])
                for poly, rug_val in polygon_rug_list:
                    if poly.contains(pt):
                        rug_values[i, j] = rug_val if rug_val is not None else default_rugosity
                        D50_values[i, j] = rug_val if rug_val is not None else default_rugosity

                        break

    ds["Rugosity"].values[:] = rug_values
    ds["D50"].values[:] = D50_values
    print("✅ ds['Rugosity'] updated using KMZ polygons.")

    return ds


def conversion_rugosity_height_to_rugsity_length(ds, conversion_rugosity_height_to_length, fillValue):

    if conversion_rugosity_height_to_length is None:
        conversion_rugosity_height_to_length = 1.0
    print(f'conversion_rugosity_height_to_rugsity_length  equals to {conversion_rugosity_height_to_length}.')

    if "Rugosity" in ds:
        ds["Rugosity"] = xr.where(ds["Rugosity"] >= 0, ds["Rugosity"] / conversion_rugosity_height_to_length, fillValue)
    return ds

# -----------------------------
# Process Shapefiles + Rugosity
# -----------------------------
def process_shapefiles_and_rugosity(
        ds,
        shapefile_folder,
        value_column,
        rugosity_method=None,
        rugosity_map=None,
        kmz_folder=None,
        kmz_rugosity_map=None,
        use_background=True,
        percent_in_0_1=False,
        background_rugosity=None,
        default_kmz_rugosity=None,
        land_mask_shp="ne_10m_land//ne_10m_land.shp",
        conversion_rugosity_height_to_length=None,
        fillValue=-9999,
        fillValue_2=0,
    ):
    if ds is None or "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("A dataset with 'lon' and 'lat' coordinates must be provided.")

    shapefile_list = sorted(glob.glob(f"{shapefile_folder}/*.shp"))
    if not shapefile_list:
        print("⚠️ No shapefiles found. Exiting. use background_rugosity ")
        if use_background:
            ds["Rugosity"] = xr.where(ds["bathymetry"] >= 0, background_rugosity , fillValue)
        #return ds

    for k, shp in enumerate(shapefile_list, 1):
        print(f"\033[94mProcessing file {k}/{len(shapefile_list)}\033[0m")
        Shapefile_TO_NETCDF4(
            shp, value_column, ds=ds,
            fillValue=fillValue, fillValue_2=fillValue_2,
            land_mask_shp=land_mask_shp
        )

    Calculate_Rugosity(
        rugosity_method = rugosity_method,
        output_folder=Path(shapefile_folder)/"output_NETCDF4",
        rugosity_map=rugosity_map,
        background_rugosity=background_rugosity,
        use_background=use_background,
        percent_in_0_1=percent_in_0_1,
        fillValue_2=fillValue_2,
        fillValue=fillValue,
        template_ds=ds
    )

    if kmz_folder:
        KMZ_UPDATE_RUGOSITY_POINTWISE(ds, kmz_folder, rugosity_map=kmz_rugosity_map,
                                      default_rugosity=default_kmz_rugosity)

    conversion_rugosity_height_to_rugsity_length (ds, conversion_rugosity_height_to_length, fillValue)

    return ds

# -----------------------------
# Example usage
# -----------------------------

    # -----------------------------
    #| Sediment Type             | Description   | Typical   z0 (m)  |
    #| ------------------------- | ------------- | ----------------- |
    #| Gravas(gravel)            | coarse        | 0.01 - 0.05       |
    #| Arenas(sand)              | medium        | 0.001 - 0.005     |
    #| Finos(silt / clay)        | smooth        | 0.0001 - 0.001    |
    #| Orgánica(mud / organic)   | very  smooth  | 0.00005 - 0.0005  |
    #| ------------------------- | ------------- | ----------------- |
    #| Sediment Type             | Description   | Typical   z0 (m)  |
    #| ------------------------- | ------------- | ----------------- |
    #| Mud                       | ++++++        | 0.030 - 0.035       |
    #| Fine sand                 | ++++++        | 0.025 - 0.030     |
    #| coarse sand               | ++++++        | 0.030 - 0.035    |
    #| Gravel                    | ++++++        | 0.035 - 0.050  |
    #| ------------------------- | ------------- | ----------------- |
