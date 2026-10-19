import xarray as xr 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray # Registers the .rio accessor

def open_temperature_subset(filepath): 

    temp_ds = xr.open_dataset(filepath)

    temp_dataset = temp_ds["2m_temperature_daily_mean"]

    return temp_dataset

def open_snowdepth_subset(filepath): 
    ds = xr.open_dataset(filepath)

    snow_depth = ds["snow_depth_daily_mean"]

    return snow_depth

def filter_2_area(data, study_area):
    z = 0

    return z


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


"""

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
