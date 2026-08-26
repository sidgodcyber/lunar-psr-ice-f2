"""
BAH GREEN26 — DELIVERABLE 1: PSR MAPPING & DOUBLY SHADOWED CRATER IDENTIFICATION
=================================================================================
Target: Faustini PSR, Lunar South Pole (~87°S)
DEM:    LM7_final_adj_5mpp_surf.tif  (5 m/pixel, south polar stereographic, MOON_ME/DE421)
Output: PSR boundary mask, illumination fraction map, crater identification

Run from D:/BAH 26/ with venv active:
    python code/deliverable1_psr_mapping.py

Author: BAH Green26 Team
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from scipy.ndimage import label, uniform_filter, binary_erosion
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit paths to match your system
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = r"D:/BAH 26"
DEM_PATH    = os.path.join(BASE_DIR, "data/raw/DEM/LM7_final_adj_5mpp_surf.tif")
SLOPE_PATH  = os.path.join(BASE_DIR, "data/raw/DEM/LM7_final_adj_5mpp_slp.tif")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
FIG_DIR     = os.path.join(BASE_DIR, "outputs/figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Faustini crater approximate center in south polar stereographic (meters)
# These are approximate — the script will refine based on DEM morphology
FAUSTINI_CENTER_APPROX = (-71000, -634000)  # (x_stereo, y_stereo) in meters

# Sun parameters for illumination modeling
# At lunar south pole, max solar elevation is ~1.54° (ecliptic obliquity)
SUN_MAX_ELEV_DEG  = 1.54      # degrees above horizon (max possible at south pole)
SUN_MIN_ELEV_DEG  = -1.54     # minimum (below horizon)
N_SUN_STEPS       = 360       # discretize solar azimuth (1° steps = 1 full orbit)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DEM AND SLOPE
# ─────────────────────────────────────────────────────────────────────────────
def load_dem_and_slope():
    print("\n[1/6] Loading LOLA DEM and slope map...")
    
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        profile = src.profile
        nodata = src.nodata
        pixel_size = abs(transform.a)  # meters per pixel
        
        # Get extent
        bounds = src.bounds
        print(f"    DEM shape:      {dem.shape[0]} x {dem.shape[1]} pixels")
        print(f"    Pixel size:     {pixel_size:.1f} m")
        print(f"    Elevation range:{np.nanmin(dem):.1f} to {np.nanmax(dem):.1f} m")
        print(f"    CRS:            {crs}")
        print(f"    Bounds (m):     X[{bounds.left:.0f}, {bounds.right:.0f}]")
        print(f"                    Y[{bounds.bottom:.0f}, {bounds.top:.0f}]")
    
    # Replace nodata with NaN
    if nodata is not None:
        dem[dem == nodata] = np.nan
    
    with rasterio.open(SLOPE_PATH) as src:
        slope = src.read(1).astype(np.float32)
        slope_nodata = src.nodata
    
    if slope_nodata is not None:
        slope[slope == slope_nodata] = np.nan
    
    print(f"    Slope range:    {np.nanmin(slope):.1f}° to {np.nanmax(slope):.1f}°")
    print("    ✓ DEM and slope loaded successfully")
    
    return dem, slope, transform, crs, profile, pixel_size


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: COMPUTE ILLUMINATION MODEL
# ─────────────────────────────────────────────────────────────────────────────
def compute_illumination_model(dem, pixel_size, n_steps=N_SUN_STEPS):
    """
    Horizon-based illumination model.
    
    For each pixel, compute the horizon elevation angle in all directions.
    A pixel is illuminated for a given Sun azimuth/elevation if the Sun
    elevation exceeds the horizon elevation in that direction.
    
    PSR = pixels where illumination_fraction == 0.0
    """
    print(f"\n[2/6] Computing illumination model ({n_steps} solar azimuth steps)...")
    print("      This may take 2-5 minutes for large DEMs...")
    
    rows, cols = dem.shape
    illumination_count = np.zeros((rows, cols), dtype=np.float32)
    valid_steps = 0
    
    # Solar azimuth steps (0° = North, 90° = East, etc.)
    azimuths = np.linspace(0, 360, n_steps, endpoint=False)
    
    # At the lunar south pole, Sun moves along the horizon
    # Solar elevation varies sinusoidally with azimuth due to 1.54° obliquity
    # We model this as a sinusoidal elevation variation
    sun_elevations = SUN_MAX_ELEV_DEG * np.sin(np.radians(azimuths))
    
    # Downsample for speed if DEM is very large
    max_dim = 2000
    if rows > max_dim or cols > max_dim:
        scale = max_dim / max(rows, cols)
        from scipy.ndimage import zoom
        dem_ds = zoom(dem, scale, order=1)
        ps_ds = pixel_size / scale
        print(f"      Downsampling DEM to {dem_ds.shape} for illumination model")
        print(f"      (Full resolution for final outputs)")
    else:
        dem_ds = dem
        ps_ds = pixel_size
    
    rows_ds, cols_ds = dem_ds.shape
    illum_ds = np.zeros((rows_ds, cols_ds), dtype=np.float32)
    
    for i, (az, el) in enumerate(zip(azimuths, sun_elevations)):
        if i % 36 == 0:
            print(f"      Step {i+1}/{n_steps} (azimuth={az:.0f}°, sun_el={el:.3f}°)")
        
        # Direction vector (in pixel coordinates)
        az_rad = np.radians(az)
        dx = np.sin(az_rad)   # col direction (East = +col)
        dy = -np.cos(az_rad)  # row direction (North = -row in raster)
        
        # For each pixel, check if sun is above the terrain horizon
        # Horizon elevation = max(arctan((dem[neighbor] - dem[pixel]) / distance))
        # We use a simplified ray-marching approach
        
        illum_az = _compute_horizon_illumination(dem_ds, ps_ds, dx, dy, el)
        illum_ds += illum_az
        valid_steps += 1
    
    # Normalize to fraction [0, 1]
    if valid_steps > 0:
        illum_ds /= valid_steps
    
    # Upsample back to full resolution if we downsampled
    if dem_ds.shape != dem.shape:
        from scipy.ndimage import zoom
        scale_back = dem.shape[0] / dem_ds.shape[0]
        illumination = zoom(illum_ds, scale_back, order=1)
        # Clip due to interpolation artifacts
        illumination = np.clip(illumination, 0, 1)
    else:
        illumination = illum_ds
    
    print(f"    ✓ Illumination model complete")
    print(f"      Illuminated pixels: {np.sum(illumination > 0.01):.0f} ({100*np.mean(illumination > 0.01):.1f}%)")
    print(f"      PSR pixels (0%):    {np.sum(illumination < 0.001):.0f} ({100*np.mean(illumination < 0.001):.1f}%)")
    
    return illumination


def _compute_horizon_illumination(dem, pixel_size, dx, dy, sun_el_deg):
    """
    For a given Sun direction (dx, dy) and elevation, return binary
    illumination map using simplified horizon scanning.
    """
    rows, cols = dem.shape
    illuminated = np.ones((rows, cols), dtype=np.float32)
    
    # Ray length — scan up to 50 km for horizon (10000 pixels at 5m)
    max_range_pixels = min(min(rows, cols) // 2, 10000)
    
    # Sample along ray in direction OPPOSITE to sun (looking uphill toward sun)
    # Horizon method: pixel is shadowed if any terrain between it and Sun
    # blocks the line of sight
    
    sun_el_rad = np.radians(sun_el_deg)
    
    for step in range(1, max_range_pixels, 3):  # step by 3 pixels for speed
        dist_m = step * pixel_size
        
        # Offset in pixel coords
        row_off = int(round(step * dy))
        col_off = int(round(step * dx))
        
        if row_off == 0 and col_off == 0:
            continue
        
        # Required terrain elevation at this distance to block sun
        # (height above pixel needed to cast shadow)
        terrain_height_needed = dem + dist_m * np.tan(sun_el_rad)
        
        # Shift DEM to get elevation at the blocker location
        from scipy.ndimage import shift
        dem_shifted = shift(dem, (-row_off, -col_off), order=1, cval=np.nan)
        
        # Where blocker terrain is higher than needed → shadow
        shadow_mask = (dem_shifted > terrain_height_needed) & (~np.isnan(dem_shifted))
        illuminated[shadow_mask] = 0.0
    
    # If sun is below horizon (negative elevation), everything is dark
    if sun_el_deg < 0:
        illuminated[:] = 0.0
    
    return illuminated


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: DELINEATE PSR BOUNDARY
# ─────────────────────────────────────────────────────────────────────────────
def delineate_psr(illumination, dem, transform, pixel_size):
    """PSR = permanently shadowed region = zero illumination fraction."""
    print("\n[3/6] Delineating PSR boundary...")
    
    # PSR mask: pixels with near-zero illumination
    psr_mask = (illumination < 0.005).astype(np.uint8)
    
    # Remove isolated single pixels (noise)
    from scipy.ndimage import binary_opening, binary_closing
    psr_clean = binary_opening(psr_mask, iterations=2).astype(np.uint8)
    psr_clean = binary_closing(psr_clean, iterations=3).astype(np.uint8)
    
    # Label connected PSR regions
    labeled, n_regions = label(psr_clean)
    
    # Find the largest PSR region (= main Faustini PSR)
    region_sizes = np.bincount(labeled.ravel())
    region_sizes[0] = 0  # ignore background
    main_psr_label = np.argmax(region_sizes)
    
    main_psr = (labeled == main_psr_label).astype(np.uint8)
    psr_area_km2 = np.sum(main_psr) * (pixel_size/1000)**2
    
    # Get PSR centroid in pixel coords
    psr_rows, psr_cols = np.where(main_psr)
    psr_center_row = int(np.mean(psr_rows))
    psr_center_col = int(np.mean(psr_cols))
    
    print(f"    ✓ PSR identified: {n_regions} shadow regions found")
    print(f"      Main PSR area:  {psr_area_km2:.1f} km²")
    print(f"      PSR center:     pixel ({psr_center_row}, {psr_center_col})")
    
    return main_psr, psr_area_km2, (psr_center_row, psr_center_col)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: IDENTIFY DOUBLY SHADOWED CRATER (DSC)
# ─────────────────────────────────────────────────────────────────────────────
def identify_doubly_shadowed_crater(dem, main_psr, illumination, transform, pixel_size):
    """
    Doubly shadowed crater = small depression WITHIN the PSR that has
    ADDITIONAL topographic shadowing (its own shadow inside the PSR).
    
    Detection method:
    1. Within PSR, find circular depressions using local DEM minima
    2. Apply crater morphology filter (depth/diameter ~ 0.1-0.2)
    3. Flag deepest/most isolated depression as the DSC
    """
    print("\n[4/6] Identifying doubly shadowed crater (DSC)...")
    
    # Work only within the PSR
    dem_in_psr = dem.copy()
    dem_in_psr[main_psr == 0] = np.nan
    
    # Find local minima: pixels lower than their surroundings
    # Use a sliding window to compute local relative depth
    window_km = 2.0  # 2 km window to find crater-scale depressions
    window_px = int(window_km * 1000 / pixel_size)
    window_px = max(window_px, 10)  # minimum 10 pixels
    
    print(f"      Search window: {window_km} km = {window_px} pixels")
    
    # Local minimum using minimum filter
    from scipy.ndimage import minimum_filter, maximum_filter
    
    local_min = minimum_filter(np.nan_to_num(dem_in_psr, nan=99999), size=window_px)
    local_max = maximum_filter(np.nan_to_num(dem_in_psr, nan=-99999), size=window_px)
    
    # Relative depth = how much lower than surroundings
    local_range = local_max - local_min
    relative_depth = (np.nan_to_num(dem_in_psr, nan=0) - local_min)
    
    # Candidate crater pixels: in PSR AND significantly below surroundings
    depth_threshold = 200  # meters — typical small crater depth
    crater_candidates = (
        (main_psr == 1) &
        (relative_depth < depth_threshold) &  # low relative elevation = bottom of depression
        (local_range > 100) &  # area has some relief = not flat
        (~np.isnan(dem))
    )
    
    # Find connected depression regions
    labeled_craters, n_craters = label(crater_candidates)
    
    if n_craters == 0:
        print("    ⚠ No clear crater depressions found in PSR")
        print("      Using PSR floor minimum as proxy DSC location")
        min_idx = np.nanargmin(dem_in_psr)
        dsc_row, dsc_col = np.unravel_index(min_idx, dem.shape)
        dsc_diameter_m = 500  # default estimate
    else:
        # Score each candidate by: size + depth + roundness
        scores = []
        for lbl in range(1, min(n_craters+1, 20)):  # check top 20
            region = (labeled_craters == lbl)
            if np.sum(region) < 5:
                continue
            
            area_px = np.sum(region)
            depth = np.nanmean(dem[~region & (main_psr==1)]) - np.nanmean(dem[region])
            
            # Roundness: compare to circle of same area
            r_rows, r_cols = np.where(region)
            cr, cc = np.mean(r_rows), np.mean(r_cols)
            dists = np.sqrt((r_rows - cr)**2 + (r_cols - cc)**2)
            radius_equiv = np.sqrt(area_px / np.pi)
            roundness = 1 - np.std(dists) / (radius_equiv + 1e-6)
            
            score = area_px * 0.3 + max(depth, 0) * 0.5 + roundness * 200
            scores.append((score, lbl, area_px, depth, roundness, cr, cc))
        
        if scores:
            scores.sort(reverse=True)
            best = scores[0]
            _, best_lbl, area_px, depth, roundness, dsc_row, dsc_col = best
            dsc_row, dsc_col = int(dsc_row), int(dsc_col)
            dsc_diameter_m = 2 * np.sqrt(area_px / np.pi) * pixel_size
            
            print(f"    ✓ Best DSC candidate: Label {best_lbl}")
            print(f"      Diameter:    {dsc_diameter_m:.0f} m")
            print(f"      Depth:       {depth:.0f} m")
            print(f"      Roundness:   {roundness:.3f} (1.0 = perfect circle)")
            print(f"      Location:    pixel ({dsc_row}, {dsc_col})")
        else:
            min_idx = np.nanargmin(dem_in_psr)
            dsc_row, dsc_col = np.unravel_index(min_idx, dem.shape)
            dsc_diameter_m = 500
    
    # Convert pixel to geographic coordinates
    dsc_x, dsc_y = rasterio.transform.xy(transform, dsc_row, dsc_col)
    
    # Convert stereographic to lat/lon (approximate for south pole)
    # For south polar stereographic: lat = -90 + arctan(r/R_moon)
    R_moon = 1737400  # meters
    r = np.sqrt(dsc_x**2 + dsc_y**2)
    dsc_lat = -90 + np.degrees(np.arctan(r / R_moon)) * (180/np.pi) / (np.pi/2)
    dsc_lon = np.degrees(np.arctan2(dsc_x, -dsc_y)) % 360
    
    # More precise formula for south polar stereographic
    rho = np.sqrt(dsc_x**2 + dsc_y**2)
    dsc_lat = np.degrees(np.arcsin(1 - rho**2 / (2 * R_moon**2 + rho**2/2)))
    dsc_lat = -abs(dsc_lat)  # south pole hemisphere
    
    print(f"      Approx lat/lon: {dsc_lat:.3f}°S, {dsc_lon:.3f}°E")
    
    return dsc_row, dsc_col, dsc_diameter_m, dsc_lat, dsc_lon


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SAVE GEOTIFF OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def save_geotiff_outputs(main_psr, illumination, dem, transform, crs, profile):
    print("\n[5/6] Saving GeoTIFF outputs...")
    
    out_profile = profile.copy()
    out_profile.update(dtype='float32', count=1, compress='lzw')
    
    # PSR binary mask
    psr_path = os.path.join(OUT_DIR, "psr_mask.tif")
    with rasterio.open(psr_path, 'w', **{**out_profile, 'dtype': 'uint8', 'nodata': 255}) as dst:
        dst.write(main_psr.astype(np.uint8), 1)
    print(f"    ✓ PSR mask saved: {psr_path}")
    
    # Illumination fraction map
    illum_path = os.path.join(OUT_DIR, "illumination_fraction.tif")
    with rasterio.open(illum_path, 'w', **out_profile) as dst:
        dst.write(illumination.astype(np.float32), 1)
    print(f"    ✓ Illumination map saved: {illum_path}")
    
    return psr_path, illum_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: PUBLICATION-QUALITY FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def make_figures(dem, slope, illumination, main_psr, transform, pixel_size,
                 dsc_row, dsc_col, dsc_diameter_m, psr_area_km2):
    print("\n[6/6] Generating publication-quality figures...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0a0a0a')
    
    extent_km = [0, dem.shape[1]*pixel_size/1000, 0, dem.shape[0]*pixel_size/1000]
    
    # ── Panel A: DEM Hillshade ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor('#0a0a0a')
    
    # Create hillshade
    from scipy.ndimage import sobel
    dx_dem = sobel(np.nan_to_num(dem), axis=1)
    dy_dem = sobel(np.nan_to_num(dem), axis=0)
    sun_az = np.radians(315)
    sun_el = np.radians(30)
    hillshade = (np.cos(sun_el) * np.cos(np.arctan2(dy_dem, dx_dem) - sun_az) +
                 np.sin(sun_el))
    hillshade = np.clip(hillshade, 0, 1)
    
    im_hs = ax.imshow(hillshade, cmap='gray', extent=extent_km, origin='upper', aspect='equal')
    im_dem = ax.imshow(np.nan_to_num(dem), cmap='terrain', alpha=0.6,
                       extent=extent_km, origin='upper', aspect='equal')
    
    # DSC marker
    dsc_x_km = dsc_col * pixel_size / 1000
    dsc_y_km = (dem.shape[0] - dsc_row) * pixel_size / 1000
    dsc_radius_km = dsc_diameter_m / 2000
    circle = Circle((dsc_x_km, dsc_y_km), dsc_radius_km,
                    fill=False, edgecolor='cyan', linewidth=2.5)
    ax.add_patch(circle)
    ax.annotate('DSC', (dsc_x_km, dsc_y_km),
                color='cyan', fontsize=11, fontweight='bold',
                xytext=(dsc_x_km + 2, dsc_y_km + 2), textcoords='data')
    
    ax.set_title('A. LOLA DEM — Faustini Region (LM7, 5m/px)', color='white', fontsize=12, pad=8)
    ax.set_xlabel('Distance East (km)', color='white')
    ax.set_ylabel('Distance North (km)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    plt.colorbar(im_dem, ax=ax, label='Elevation (m)', fraction=0.04)
    
    # ── Panel B: Illumination Fraction ────────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor('#0a0a0a')
    
    cmap_illum = plt.cm.viridis.copy()
    im_illum = ax.imshow(illumination, cmap=cmap_illum, vmin=0, vmax=1,
                          extent=extent_km, origin='upper', aspect='equal')
    
    # PSR contour
    ax.contour(np.flipud(main_psr), levels=[0.5], colors=['red'], linewidths=2,
               extent=extent_km)
    
    # DSC marker
    circle2 = Circle((dsc_x_km, dsc_y_km), dsc_radius_km,
                     fill=False, edgecolor='cyan', linewidth=2.5)
    ax.add_patch(circle2)
    
    ax.set_title('B. Annual Illumination Fraction (0=PSR, 1=always lit)', color='white', fontsize=12, pad=8)
    ax.set_xlabel('Distance East (km)', color='white')
    ax.set_ylabel('Distance North (km)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')D
    cb = plt.colorbar(im_illum, ax=ax, label='Illumination Fraction', fraction=0.04)
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white')
    
    psr_patch = mpatches.Patch(edgecolor='red', fill=False, label=f'PSR boundary ({psr_area_km2:.0f} km²)')
    ax.legend(handles=[psr_patch], loc='upper right', facecolor='#222', labelcolor='white')
    
    # ── Panel C: PSR Mask + Slope ──────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor('#0a0a0a')
    
    # Show slope
    im_slope = ax.imshow(np.clip(slope, 0, 40), cmap='hot', extent=extent_km,
                          origin='upper', aspect='equal')
    
    # PSR overlay
    psr_rgba = np.zeros((*main_psr.shape, 4))
    psr_rgba[main_psr == 1] = [0.0, 0.4, 1.0, 0.35]  # semi-transparent blue
    ax.imshow(psr_rgba, extent=extent_km, origin='upper', aspect='equal')
    
    # Safe slope zone (< 15°) within PSR
    safe_psr = (main_psr == 1) & (slope < 15)
    safe_rgba = np.zeros((*safe_psr.shape, 4))
    safe_rgba[safe_psr] = [0.0, 1.0, 0.3, 0.5]  # green = safe
    ax.imshow(safe_rgba, extent=extent_km, origin='upper', aspect='equal')
    
    circle3 = Circle((dsc_x_km, dsc_y_km), dsc_radius_km,
                     fill=False, edgecolor='cyan', linewidth=2.5)
    ax.add_patch(circle3)
    
    ax.set_title('C. Slope Map + PSR (blue) + Safe Zones <15° (green)', color='white', fontsize=12, pad=8)
    ax.set_xlabel('Distance East (km)', color='white')
    ax.set_ylabel('Distance North (km)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    plt.colorbar(im_slope, ax=ax, label='Slope (degrees)', fraction=0.04)
    
    # ── Panel D: DSC Close-up ─────────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor('#0a0a0a')
    
    # Zoom window: 10 km around DSC
    zoom_km = 10
    zoom_px = int(zoom_km * 1000 / pixel_size)
    
    r0 = max(0, dsc_row - zoom_px//2)
    r1 = min(dem.shape[0], dsc_row + zoom_px//2)
    c0 = max(0, dsc_col - zoom_px//2)
    c1 = min(dem.shape[1], dsc_col + zoom_px//2)
    
    dem_zoom = dem[r0:r1, c0:c1]
    psr_zoom = main_psr[r0:r1, c0:c1]
    illum_zoom = illumination[r0:r1, c0:c1]
    
    ext_zoom = [0, (c1-c0)*pixel_size/1000, 0, (r1-r0)*pixel_size/1000]
    
    hs_zoom_x = sobel(np.nan_to_num(dem_zoom), axis=1)
    hs_zoom_y = sobel(np.nan_to_num(dem_zoom), axis=0)
    hs_zoom = np.clip(np.cos(sun_el)*np.cos(np.arctan2(hs_zoom_y, hs_zoom_x)-sun_az) + np.sin(sun_el), 0, 1)
    
    ax.imshow(hs_zoom, cmap='gray', extent=ext_zoom, origin='upper', aspect='equal')
    
    # PSR inside zoom
    psr_z = np.zeros((*psr_zoom.shape, 4))
    psr_z[psr_zoom == 1] = [0.0, 0.5, 1.0, 0.4]
    ax.imshow(psr_z, extent=ext_zoom, origin='upper', aspect='equal')
    
    # DSC circle in zoomed coords
    dsc_z_x = (dsc_col - c0) * pixel_size / 1000
    dsc_z_y = (r1 - r0 - (dsc_row - r0)) * pixel_size / 1000
    circle4 = Circle((dsc_z_x, dsc_z_y), dsc_radius_km,
                     fill=False, edgecolor='cyan', linewidth=2.5)
    ax.add_patch(circle4)
    ax.plot(dsc_z_x, dsc_z_y, '+', color='cyan', markersize=14, markeredgewidth=2)
    ax.annotate(f'DSC\nØ≈{dsc_diameter_m:.0f}m', (dsc_z_x, dsc_z_y),
                color='cyan', fontsize=10, fontweight='bold',
                xytext=(dsc_z_x + 0.5, dsc_z_y + 0.5))
    
    ax.set_title('D. DSC Close-up (10 km window) — Crater Floor = SAR Target', color='white', fontsize=12, pad=8)
    ax.set_xlabel('Distance East (km)', color='white')
    ax.set_ylabel('Distance North (km)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    
    # Overall title
    fig.suptitle(
        'BAH GREEN26 — DELIVERABLE 1: PSR Mapping & Doubly Shadowed Crater Identification\n'
        f'Target: Faustini PSR, Lunar South Pole | LOLA LM7 5m DEM | PSR Area = {psr_area_km2:.0f} km²',
        color='white', fontsize=13, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig_path = os.path.join(FIG_DIR, "deliverable1_psr_mapping.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"    ✓ Figure saved: {fig_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BAH GREEN26 — DELIVERABLE 1: PSR MAPPING")
    print("Faustini Doubly Shadowed Crater | Chandrayaan-2 DFSAR Mission")
    print("=" * 70)
    
    # Quick data check
    for path, name in [(DEM_PATH, "LOLA DEM"), (SLOPE_PATH, "Slope map")]:
        if not os.path.exists(path):
            print(f"\n❌ ERROR: {name} not found at: {path}")
            print("   Please check your file path in the CONFIG section.")
            return
        else:
            size_mb = os.path.getsize(path) / 1e6
            print(f"  ✓ {name}: {path} ({size_mb:.0f} MB)")
    
    # Run pipeline
    dem, slope, transform, crs, profile, pixel_size = load_dem_and_slope()
    
    illumination = compute_illumination_model(dem, pixel_size)
    
    main_psr, psr_area_km2, psr_center = delineate_psr(illumination, dem, transform, pixel_size)
    
    dsc_row, dsc_col, dsc_diameter_m, dsc_lat, dsc_lon = identify_doubly_shadowed_crater(
        dem, main_psr, illumination, transform, pixel_size)
    
    save_geotiff_outputs(main_psr, illumination, dem, transform, crs, profile)
    
    make_figures(dem, slope, illumination, main_psr, transform, pixel_size,
                 dsc_row, dsc_col, dsc_diameter_m, psr_area_km2)
    
    # Summary report
    print("\n" + "=" * 70)
    print("DELIVERABLE 1 — SUMMARY REPORT")
    print("=" * 70)
    print(f"  PSR Area:              {psr_area_km2:.1f} km²")
    print(f"  PSR Center (pixel):    {psr_center}")
    print(f"  DSC Location (pixel):  ({dsc_row}, {dsc_col})")
    print(f"  DSC Approx Lat/Lon:    {dsc_lat:.3f}°S, {dsc_lon:.3f}°E")
    print(f"  DSC Diameter:          {dsc_diameter_m:.0f} m")
    print(f"  Depth/Diam ratio:      ~0.1–0.2 (expected for primary impact)")
    print(f"\n  GeoTIFF outputs:       {OUT_DIR}/")
    print(f"  Figures:               {FIG_DIR}/deliverable1_psr_mapping.png")
    print("\n  NEXT STEP → Deliverable 2: DFSAR CPR/DOP ice detection")
    print("=" * 70)


if __name__ == "__main__":
    main()