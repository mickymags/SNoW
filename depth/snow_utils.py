import xarray as xr 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray # Registers the .rio accessor
import requests
import subprocess

# Define a sinusoidal function
def sine(t, A, B, C, phi):
    """
    y = A * sin(2pi * t/ B + phi) + C

    Parameters 
    ---
    t = time 
    A = Amplitude 
    B = Period 
    phi = phase shift 
    C = vertical phase
    """
    return A * np.sin(2 * np.pi * t / B + phi) + C
    

# Define a function to calculate the coefficient of determination (also known as R squared)
def r_sq(real_data, fit_data):
    """
    A function to calculate r^2. 
    R^2 is equivalent to 1 - (SS_res/SS_tot), where SS_res is 
    the sum of the squared residuals and SS_tot is the total 
    sum of squares

    Parameters 
    ---
    real_data : np.ndarray 
        - an array containing the actual data
    fit_data : np.ndarray
        - an array containing values that were 
            synthesized with a program such as curve_fit
    """

    squared_errors = []
    squared_devs = []
    mean = np.mean(real_data)

    for j in range(len(real_data)):

        # Obtain real and fitted values
        real = real_data[j]
        fitted = fit_data[j]

        # Calculate Squared Residual
        error = real - fitted 
        sq_error = error **2 
        squared_errors.append(sq_error)

        # Calculate Squares 
        deviation = real - mean 
        squared_dev = deviation ** 2 
        squared_devs.append(squared_dev)

    ss_res = np.array(squared_errors).sum()
    ss_tot = np.array(squared_devs).sum()

    r_sq = 1 - ss_res/ss_tot

    return r_sq

def open_temperature_subset(filepath): 

    temp_ds = xr.open_dataset(filepath)

    temp_dataset = temp_ds["2m_temperature_daily_mean"]

    return temp_dataset

def open_snowdepth_subset(filepath): 
    ds = xr.open_dataset(filepath)

    snow_depth = ds["snow_depth_daily_mean"]

    return snow_depth

def open_swe_dataset(filepath): 
    ds = xr.open_dataset(filepath)

    swe = ds["snow_depth_water_equivalent_daily_mean"]

    return swe

def filter_2_area(data, study_area):
    z = 0

    return z

def simple_merge(
        input_dir: list[Path], 
        output_file : Path, 
        nodata: float | None = None
) -> None: 

    output_filename = output_file
    #output_filename = str(input_dir) + '/output_file' 
    #print(f'output_filename:  {output_filename}')
    path_dir = Path(input_dir)

    input_files = [
        tif for tif in path_dir.glob('*.tif')
        if tif.name != Path(output_filename).name
    ]

    # If there are no input files, throw an error 
    if not input_files: 
        raise FileNotFoundError(f"No .tif files found in {input_dir}")

    merge_command = [
        "gdal_merge.py",
        "-o",
        str(output_filename)
    ]

    # Only add nodata arguments if a value was provided 
    if nodata is not None: 
        merge_command.extend([
            "-n",
            str(nodata),
            "-a_nodata",
            str(nodata)
        ])

    # Add input files 
    merge_command.extend(
        str(tif) for tif in input_files
    )

    subprocess.run(merge_command, check = True)

def elevation_stats(geom, dem_path, crs = "EPSG:4326"): 
    """
    This function will extract the minimum, maximum, mean, and 
    standard deviation values from a DEM over a certain area of interest, which 
    is defined via a shapefile. 
    """
    dem = rioxarray.open_rasterio(
        dem_path, 
        masked=True, 
        chunks={"band": 1, "x": 2048, "y": 2048},
        ).squeeze("band", drop=True)

    # Read only the polygons bounding box, and turn everything outside the polygon into a NaN
    dem = dem.rio.clip_box(*geom.bounds, crs = crs)
    dem = dem.rio.clip([geom], crs=crs, drop=True, all_touched=False)

    return_dict = {
        "mean": float(dem.mean()), 
        "min": float(dem.min()),
        "max": float(dem.max()),
        "stdev": float(dem.std())
    }

    return return_dict

def elevation_histogram(dem, name, out_folder, geom, bins = 200):
    dem = rioxarray.open_rasterio(
        dem, 
        masked=True,
        chunks = {"band": 1, "x": 2048, "y": 2048}
    ).squeeze("band", drop=True)

    dem = dem.rio.clip_box(*geom.bounds, crs = "EPSG:4326")
    dem = dem.rio.clip([geom], crs="EPSG:4326", drop=True)

    vals = dem.values.ravel()
    vals = vals[~np.isnan(vals)]   # drop pixels outside the polygon

    fig, ax = plt.subplots(figsize = (8, 5))
    ax.hist(vals, bins = bins, color = "tab:gray", edgecolor = "black")
    ax.axvline(vals.mean(), color="tab:red", linestyle = "--",
                label = f"mean = {vals.mean():.0f} m")
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("Number of Pixels")
    ax.set_title(f"GLO-30 elevation in {name}")
    ax.legend() 
    ax.grid()

    fig.savefig(f"{out_folder}/{name}_elevation_hist.png", dpi = 200, bbox_inches = "tight")
    plt.close(fig)

def download_glo30(xmin, ymin, xmax, ymax, output_dir): 
    tiles = []

    output_dir = Path(output_dir)

    #lat_prefix = f"N{int(ymin):02d}"
    #lon_prefix = f"W{abs((int(xmin)-1)):03d}"

    xdiff = xmax - xmin 
    ydiff = ymax - ymin 

    xmin_floor = int(np.floor(xmin))
    ymin_floor = int(np.floor(ymin))
    xmax_ceil =  int(np.floor(xmax))
    ymax_ceil =  int(np.floor(ymax))

    tiles = []

    xspace = np.linspace(xmin_floor, xmax_ceil, (xmax_ceil - xmin_floor + 1))
    yspace = np.linspace(ymin_floor, ymax_ceil, (ymax_ceil - ymin_floor + 1))

    for y in yspace: 
        lat_prefix = f"N{int(y):02d}"
        for x in xspace:
            lon_prefix = f"W{abs(int(x)):03d}"
            tile_name = (
                f"Copernicus_DSM_COG_10_"f"{lat_prefix}_00_"f"{lon_prefix}_00_DEM"
            )

            #print(tile_name)
            if tile_name not in tiles:
                tiles.append(tile_name)


    for u in tiles:

        # Download URL 
        url = (
            f"https://copernicus-dem-30m.s3.amazonaws.com/"
            f"{u}/{u}.tif"
        )

        output_file = output_dir / f"{u}.tif"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk: 
                    f.write(chunk)


    #print(lon_prefix)


# Define input and output directories 

# Input ERA5 Directory 
my_era5_directory = Path('/home/mmaganini/Documents/ess601/era5_project/era5_data/')

# Input Shapefile Directory 
my_shp = Path('/home/mmaganini/Documents/ess601/era5_project/study_area/drive-download-20261018T215517Z-1-001/HUC6_Updated.shp')

# Read shapefile
huc6 = gpd.read_file(my_shp)

# Convert to Global shape_file 
huc6 = huc6.to_crs("EPSG:4326")

# # Define watershed names 
watershed_names = {
    0: "Copper River",
    1: "Tanana River",
    2: "Prudhoe Bay"
}



"""
output_directory = '/home/mmaganini/Documents/ess601/era5_project/era5_output_data'

for era5_file in sorted(my_era5_directory.iterdir()):
    print('='*60)
    print(f'Opening {era5_file}')
    year = str(era5_file)[-7:-3]
    print(year)

    snow_depth_ds = open_snowdepth_subset(era5_file).copy()
    temp_ds = open_temperature_subset(era5_file).copy()

    # Snow Depth Subset
    snow_depth_subset = snow_depth_ds.rio.set_spatial_dims(
        x_dim="longitude",
        y_dim="latitude"
    )
    snow_depth_subset = snow_depth_ds.rio.write_crs("EPSG:4326")

    snow_depth_subset = snow_depth_subset.assign_coords(
        longitude = (
            (snow_depth_subset.longitude + 180) % 360
        ) - 180
    )

    # Sort the longitude coordinate after conversion 
    snow_depth_subset = snow_depth_subset.sortby("longitude")


    #print(snow_depth_subset.coords)

    # Temperature Subset
    temp_subset = temp_ds.rio.set_spatial_dims(
        x_dim = "longitude",
        y_dim = "latitude"
    )

    #print(temp_subset.coords)
    temp_subset = temp_subset.rio.write_crs("EPSG:4326")

    temp_subset = temp_subset.assign_coords(
        longitude = (
            (temp_subset.longitude + 180) % 360
        ) - 180
    )

    temp_subset = temp_subset.sortby("longitude")

    for watershed_idx, watershed_name in watershed_names.items(): 

        watershed_geom = [
            huc6.geometry.iloc[watershed_idx]
        ]

        
        # Clip snow depth dataset to watershed of interest and call it sd_clipped (snow depth clipped)
        sd_clipped = snow_depth_subset.rio.clip(
            watershed_geom,
            crs = huc6.crs, 
            drop = True
        )

        # Clip temperature dataset to watershed of interest
        temp_clipped = temp_subset.rio.clip(
            watershed_geom, 
            crs = huc6.crs, 
            drop = True
        )

        # Construct snow depth output_name 
        output_file = Path(output_directory) / (
            f"{watershed_name}_{year}.nc"
        )

        print(output_file)

        watershed_ds = xr.merge([sd_clipped, temp_clipped], 
                                compat = "no_conflicts")

        # Save the NetCDF file 
        watershed_ds.to_netcdf(output_file)




for watershed_idx, watershed_name in watershed_names.items(): 

    geometry = [
        huc6.geometry.iloc[watershed_idx]
    ]

    #snow_depth = 
    #print(watershed_idx)
    #print(watershed_name)

if len(huc6) != 3: 
    raise ValueError(
        f"Expected 3 watershed features, found {len(huc6)}"
    )


huc6_columns = huc6.columns
name_col = huc6["name"]
#print(huc6_columns)
#print(name_col)


snow_depth_datasets = []
temp_datasets = []

for filepath in my_era5_directory.glob("*nc"):
    snow_depth = open_snowdepth_subset(filepath)
    temp = open_temperature_subset(filepath)

    snow_depth_datasets.append(snow_depth)
    temp_datasets.append(temp)

#print(snow_depth_datasets[0])
#for j in my_directory.iterdir():
#    ds = open_snowdepth_subset(j)
#    datasets.append(ds)
#    break
"""
