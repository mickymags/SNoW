import argparse 
import snow_utils 
import data_analysis
import xarray as xr
import geopandas as gpd
from pathlib import Path
import rioxarray # Registers the .rio accessor 
import os 
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from matplotlib.patches import Patch

def main(): 

    parser = argparse.ArgumentParser(
        description = "This file will take a folder as input that contains ERA5 files in netCDF format \
        extract the snow depth subdataset, and spatially subset them to a region of interest, which is \
        given as a .shp format. "
    )

    parser.add_argument(
        "--netcdf_folder", 
        help = "String containing absolute path to folder containing netCDF files"
    )

    parser.add_argument(
        "--shapefile",
        help = "shapefile containing"
    )

    parser.add_argument(
        "--shapefile_metadata",
        help = "list containing the order of AOI names if the shapefile contains more than one polygon. \
            This must be a comma-separated list with no spaces and also end with a comma (e.g. Copper_River,Prudhoe,Tanana,)"
    )

    args = parser.parse_args()

    my_era5_dir = Path(args.netcdf_folder)
    my_shp = Path(args.shapefile)
    my_shp_metadata = args.shapefile_metadata

    # Read Shapefile 
    huc6 = gpd.read_file(my_shp)

    # Convert to EPSG:4326 to match the projection of the ERA5 data
    huc6 = huc6.to_crs("EPSG:4326")

    print(len(huc6), huc6.crs)
    print(huc6.columns.tolist())
    print(huc6.geom_type)
    print(huc6.to_crs("EPSG:4326").bounds)
    for i, g in enumerate(huc6.to_crs("EPSG:4326").geometry):
        print(i, g.geom_type, len(getattr(g, "geoms", [g])), "part(s)")

    pattern = str(my_era5_dir) + '/*.nc'
    print(pattern)

    f = sorted(glob.glob(pattern))[0]

    raw = xr.open_dataset(f)
    print("raw       ", raw.longitude.values[:3], np.diff(raw.longitude.values)[:3])

    t = snow_utils.open_temperature_subset(f)
    print("subset fn ", t.longitude.values[:3], np.diff(t.longitude.values)[:3])

    t = t.rio.write_crs("EPSG:4326").assign_coords(longitude=((t.longitude + 180) % 360) - 180).sortby("longitude")
    print("converted ", t.longitude.values[:3])

    print("transform ", t.rio.transform())   # what rio.clip will use


    #############################################################################
    #  Define watershed names by parsing through the list given by the user     #
    #############################################################################
    #print(f'my_shp_metadata {my_shp_metadata}')

    md_list = list(my_shp_metadata)

    aois = []

    aoi = ''

    for j in md_list: 
        if j != ',':
            aoi += j
        else: 
            aois.append(aoi)
            aoi = ''

    print('\n')
    print(f'aois: {aois}')

    watershed_names = {
        0: aois[0],
        1: aois[1], 
        2: aois[2]
    }

    aoi_labels = []

    for k in aois:
        k_new = k.replace('_', ' ')
        aoi_labels.append(k_new)

    #####################################################
    # Step 1: Spatially Subset netCDF files             #
    #####################################################
    spatial_subset_folder = str(my_era5_dir.parent) + '/subset_data/'
    """
    print('='*70)
    print('Step 1: Spatially subset NetCDF files')


    # If the folder doesn't exist, make it 
    os.makedirs(spatial_subset_folder, exist_ok = True)

    years = []

    # for each netcdf file in the input folder, spatially subset it and append it to the new directory
    for era5_file in sorted(my_era5_dir.iterdir()):
        year = str(era5_file)[-7:-3]
        if year not in years:
            years.append(year)
            print(f"Processing {year} ERA5 data...")

        ############# Snow Depth Dataset ########################
        snow_depth_ds = snow_utils.open_snowdepth_subset(era5_file).copy()

        n_before = int(snow_depth_ds.isnull().sum())

        # Remove all values greater than 10 
        snow_depth_ds = snow_depth_ds.where(snow_depth_ds < 10.0)

        n_after = int(snow_depth_ds.isnull().sum())

        #print(f"Nans before: {n_before}")
        #print(f"Nans after: {n_after}")
        print(f"Removed {n_after - n_before} glacier cell values from Snow Depth Dataset.")

        # Snow Depth Subset Using rio.set_spatial_dims
        snow_depth_subset = snow_depth_ds.rio.set_spatial_dims(
            x_dim = "longitude", 
            y_dim = "latitude"
        )
        snow_depth_subset = snow_depth_ds.rio.write_crs("EPSG:4326")
        snow_depth_subset = snow_depth_subset.assign_coords(
            longitude = (
                (snow_depth_subset.longitude + 180) % 360) - 180
        )
        # Sort the longitude coordinate after conversion 
        snow_depth_subset = snow_depth_subset.sortby("longitude")

        # Clip to longitudes in our region to avoid longitude wraparound error
        snow_depth_subset = snow_depth_subset.sel(longitude=slice(-170, -120))

        #################### Snow Water Equivalent Dataset #####################
        swe_ds = snow_utils.open_swe_dataset(era5_file).copy()

        num_before = int(swe_ds.isnull().sum())

        swe_ds = swe_ds.where(swe_ds != 10.0)

        num_after = int(swe_ds.isnull().sum())

        print(f"Removed {num_after - num_before} glacier cell values from SWE Dataset.")
              
        swe_subset = swe_ds.rio.set_spatial_dims(
            x_dim = "longitude", 
            y_dim = "latitude"
        )

        swe_subset = swe_subset.rio.write_crs("EPSG:4326")

        swe_subset = swe_subset.assign_coords(
            longitude = (
                (swe_subset.longitude + 180) % 360) - 180
        )

        # Sort the longitude coordinates after conversion 
        swe_subset = swe_subset.sortby("longitude")

        # Clip to longitudes in our region to avoid longitude wraparound error 
        swe_subset = swe_subset.sel(longitude=slice(-170, -120))


        ################## Temp Dataset ##################
        temp_ds = snow_utils.open_temperature_subset(era5_file).copy() 

        temp_subset = temp_ds.rio.set_spatial_dims(
            x_dim = "longitude", 
            y_dim = "latitude"
        )

        temp_subset = temp_subset.rio.write_crs("EPSG:4326")

        temp_subset = temp_subset.assign_coords(
            longitude = (
                (temp_subset.longitude + 180) % 360
            ) - 180
        )

        temp_subset = temp_subset.sortby("longitude")

        # Clip to longitudes in our region to avoid longitude wraparound error
        temp_subset = temp_subset.sel(longitude=slice(-170, -120))

        ################ Snow Flag Dataset #########################

        # This dataset will be used to measure the snow-on and snow-off dates. 

        snow_flag = swe_subset.copy(deep=True)

        #snow_flag["snow_depth_water_equivalent"] = (
        #    snow_flag["snow_depth_water_equivalent"] > 0.01 
        #).astype("int8")

        swe = snow_flag
        glacier = swe >= 10.0
        snow = (swe > 0.1) & ~glacier 

        valid = swe.notnull()

        snow_flag = xr.Dataset({
            "glacier": glacier.where(valid),
            "snow_flag": snow.where(valid),
        })
        #snow_flag["snow_depth_water_equivalent"] = (swe > 0.1).where(swe.notnull())
        #snow_flag["snow_present"] = (swe> 0.1).astype()

        print(type(snow_flag))
        print(list(snow_flag.data_vars))

        print(f'Length of snow flag for {year} is: {len(snow_flag)}')

        # Sanity check 
        if year == years[0] and era5_file == sorted(my_era5_dir.iterdir())[0]:
            print("lon spacing", np.unique(np.round(np.diff(temp_subset.longitude.values), 3)))
            print("transform:", temp_subset.rio.transform())

        for watershed_idx, watershed_name in watershed_names.items(): 

            watershed_geom = [
                huc6.geometry.iloc[watershed_idx]
            ]

            # Clip snow depth ds to watershed of interest
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

            swe_clipped = swe_subset.rio.clip(
                watershed_geom, 
                crs = huc6.crs, 
                drop = True
            )

            snow_flag_clipped = snow_flag.rio.clip(
                watershed_geom, 
                crs = huc6.crs, 
                drop = True
            )

            # Merge temperature, snow depth, and snow water equivalent datasets together 
            watershed_ds = xr.merge([sd_clipped, temp_clipped, swe_clipped, snow_flag_clipped], 
                                    compat = "no_conflicts")

            # Construct Output file name
            output_file = Path(spatial_subset_folder) / (
                f"{watershed_name}_{year}.nc"
            )

            # Save the netCDF file 
            watershed_ds.to_netcdf(output_file)
    """

    ############# Delete Later ############
    print('Diagnostic')
    gdf = gpd.read_file(my_shp).to_crs("EPSG:4326")
    print(type(gdf.bounds))
    #print(gdf.bounds)

    bboxes = []
    for j in range(gdf.shape[0]): 
        bboxes.append(gdf.geometry.iloc[j].bounds)

    print(f'bboxes: {bboxes}')
    #bbox0 = gdf.geometry.iloc[0].bounds() 
    #bbox1 = gdf.geometry.iloc[1].bounds()
    #bbox2 = gdf.goemtry.iloc[2].bounds()

    #src = sorted(glob.glob(spatial_subset_folder + '/'))

    ################## Delete above later #########

    ############################################################
    #       Step 2: Download DEM data                          #
    ############################################################
    print('='*70)
    print('\n')
    print('='*70)
    print('Step 2: Download DEM data')

    dem_dir = str(Path(spatial_subset_folder).parent) + '/dem_data'
    os.makedirs(dem_dir, exist_ok=True)
    print(dem_dir)

    for f in range(len(bboxes)): 
        my_bbox = bboxes[f]
        my_aoi = aois[f]

        minx = my_bbox[0]
        miny = my_bbox[1]
        maxx = my_bbox[2]
        maxy = my_bbox[3]

        dem_folder_name = dem_dir + f'/{my_aoi}/'
        os.makedirs(dem_folder_name, exist_ok=True)
        #print(f'Downloading DEM tiles to {dem_folder_name}')
        #my_tiles = snow_utils.download_glo30(minx, miny, maxx, maxy, dem_folder_name)
        #print(f'tiles for {my_aoi}: {my_tiles}')

    # Merge DEMs for each watershed together
    print('\n')
    print('Merge DEMs for each watershed together')
    for d in range(len(bboxes)):
        my_aoi = aois[d]
        print(f'Merging DEMs for {my_aoi} watershed')
        dem_folder_name = dem_dir + f'/{my_aoi}'
        out_fname = dem_folder_name + f'/{my_aoi}_merged_dem.tif'

        """
        print(f"Writing merged DEM to {out_fname}")
        snow_utils.simple_merge(
            dem_folder_name,
            out_fname,
            nodata=None
        )
        """


    # Obtain min, max, mean, and standard deviation values for elevation

    print("Obtaining DEM stats and creating histograms")
    for b in range(len(bboxes)): 
        my_aoi_name = aois[b]
        my_folder = dem_dir + f'/{my_aoi_name}'
        out_fname = f'{my_aoi_name}_merged_dem.tif'
        my_file = my_folder + f'/{out_fname}'

        my_aoi_shp = gdf.geometry.iloc[b]

        my_stats = snow_utils.elevation_stats(my_aoi_shp, my_file)

        print(f"stats for {my_aoi_name}")
        print(my_stats)

        print(f"Creating histogram for {my_aoi_name}")
        histogram_folder = str(Path(spatial_subset_folder).parent) + '/dem_histograms'
        os.makedirs(histogram_folder, exist_ok=True)

        my_hist = snow_utils.elevation_histogram(my_file, my_aoi_name, histogram_folder, my_aoi_shp) #my_folder, my_aoi_shp)
        print(f"Wrote histogram to {histogram_folder}/{my_aoi_name}_elevation_hist.png")

    ####################################################################
    # Step 3: Combine datasets to obtain decadal time series           #
    ####################################################################

    print('='*70)
    print('\n')
    print('='*70)
    print('Step 3: Combine datasets to obtain decadal time series')


    combined_watershed_data = []

    for j in range(len(aois)):
        print(f"Processing {aois[j]} data")
        ws_data = data_analysis.load_watershed_data(str(aois[j]), spatial_subset_folder)
        combined_watershed_data.append(ws_data)

        #raw = ws_data["2m_temperature_daily_mean"]

        #lon_min = -175
        #lat_min = 90 
        #lon_max = -110 
        #lat_max = 0 

        #print(raw.sizes)
        #print(raw.isel(time=0).notnull().sum().values)
        #print(raw.latitude.values, raw.longitude.values)
        #print(raw.isnull().all(["time"]).sum().values)

        t = raw.mean(dim=[d for d in raw.dims if d != "time"])
        #print(t.isnull().sum().values)
        #print(temp_C.isnull())

    #print('bboxes',bboxes)
    ##################################################################
    #           Step 4 : Data Operations                             #
    ##################################################################
    print('='*70)
    print('\n')
    print('='*70)
    print("Step 4: Conducting Operations on datasets")

    ############### Step 4a: Spatial Reduction ######################
    # This will give us a single data point for each time index #

    print('\nStep 4a: Spatially reducing data to conduct timeseries')

    ############# Step 4a Part 1: Creating a Time Series for each watershed ################

    # Create a folder to store time series figures
    sr_fig_folder = str(my_era5_dir.parent) + '/time_series_figures/' # Name the folder
    os.makedirs(sr_fig_folder, exist_ok=True)                         # Actually make the folder

    # Create empty lists where the spatially reduced datasets will be appended
    sd_sr = []    # Snow depth Spatially Reduced
    temp_sr = []  # Temperature Spatially Reduced 
    swe_sr = []   # Snow Water Equivalent Spatially Reduced

    # For each watershed...
    for k in range(len(combined_watershed_data)): 
        print(f"Loading {aois[k]} data")

        # Obtain merged watershed dataset
        ds = combined_watershed_data[k]

        # Extract and spatially reduce temp, snow depth, and swe subsets
        sd_sub = data_analysis.spatial_reduce(ds, "snow_depth_daily_mean")
        temp_sub = data_analysis.spatial_reduce(ds, "2m_temperature_daily_mean")
        swe_sub = data_analysis.spatial_reduce(ds, "snow_depth_water_equivalent_daily_mean")

        # Examine the snow flag dataset 
        #sd_grid = ds[""]

        sd_sr.append(sd_sub)
        temp_sr.append(temp_sub)
        swe_sr.append(swe_sub)

    print('plotting time series...')
    # For each dataset in sd_sr and temp_sr:
    for m in range(len(sd_sr)):
        print(m)

        sd_oi = sd_sr[m]         # Snow depth dataset of interest
        temp_oi = temp_sr[m]     # Temperature dataset of interest
        swe_oi = swe_sr[m]       # Snow Water Equivalent of interest 

        print(len(sd_oi))
        print(len(temp_oi))
        print(len(swe_oi))

        dates = np.linspace(0, len(sd_oi), len(sd_oi)) 

        if m == 0: 
            watershed = aois[0]

        if m == 1:
            watershed = aois[1]

        if m == 2: 
            watershed = aois[2]

        my_label = aoi_labels[m]

        temp_C = temp_oi - 273.15

        # Smooth Snow Depth, Temperature, and SWE data 
        sd_smooth = sd_oi.rolling(time = 30, center = True, min_periods = 1).mean() 
        temp_smooth = temp_C.rolling(time = 30, center = True, min_periods = 1).mean() 
        swe_smooth = swe_oi.rolling(time = 30, center = True, min_periods = 1).mean()

        print(len(dates), len(sd_smooth), len(temp_smooth), len(swe_smooth))
        print(np.isnan(temp_smooth).sum())

        sd_fig_fname = sr_fig_folder + f'{watershed}_snow_depth.png'

        print("creating subplots")
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # left axis 
        ax1.plot(sd_smooth['time'], sd_smooth, color="blue", label = "Smoothed Snow Depth Data")
        ax1.set_ylabel('Snow Depth (m)')
        ax1.grid()
        ax1.set_xlabel('Year')

        ############ CURVE FITTING #############

        # convert time and snow depth values to floats
        sd_vals = sd_smooth.values.astype(float)
        t_days = np.arange(len(sd_vals), dtype=float)

        ### Initial guess for curve fitting ###
        amplitude_guess = 1 
        period_guess = 365 
        phase_guess = np.pi/2
        vert_guess = np.mean(sd_vals)

        p0 = [amplitude_guess, period_guess, phase_guess, vert_guess]

        popt, pcov = curve_fit(snow_utils.sine, t_days, sd_vals, p0=p0)

        print('\n')
        print(f'fit params')
        print('='*40)

        print(f'Amplitude for {aois[m]} is {popt[0]:0.2f}')
        print(f'Period for {aois[m]} is {popt[1]:0.2f}')
        print(f'Phase Shift for {aois[m]} is {popt[2]:0.2f}')
        print(f'Vertical Phase Shift for {aois[m]} is {popt[3]:0.2f}')

        fitted = snow_utils.sine(t_days, *popt)

        # Calculate R Squared
        rsquare = snow_utils.r_sq(sd_smooth, fitted)

        print(f' Rsquared for {aois[m]} is: {rsquare}')

        # Right axis 
        ax1.plot(sd_smooth["time"], fitted, color = 'k', linestyle = '--', label = 'Sinusoidal Fit')
        ax1.legend(loc='upper right')
        ax1.text(0.025, 0.95, f'$R^2$: {rsquare:0.3f}', transform = ax1.transAxes, fontsize = 18, 
                fontweight = 'bold', va = 'top', 
                bbox = dict(boxstyle='round', facecolor = 'white', alpha = 0.75))
        
        #ax2.plot(temp_smooth["time"], temp_smooth, color = "tab:red") #dates
        #ax2.set_ylabel('Temperature (C)', color = "tab:red")
        #ax2.tick_params(axis="y", labelcolor="tab:red")
        #ax2.invert_yaxis()

        #ax2.set_xlabel("Days since Jan 1, 2016")
        #fig.tight_layout()



        print("Saving Fig")
        plt.title(f"Snow Depth and Temperature in {my_label} since 2016")
        #plt.xlabel("Days since jan 1, 2016")
        plt.savefig(sd_fig_fname, dpi=300, bbox_inches = "tight")
        plt.close(fig)

    ####################### Step 4a Part 2: Comparing with SNOTEL data ################
    print('='*70)
    print("Step 4 Part 2: Comparing with SNOTEL data")

    # Define Degree and Minute Values for SNOTEL stations. 
    points = {
        "Prudhoe_Bay": (snow_utils.dms(68, 8),  -1 * snow_utils.dms(149, 29)),
        "Copper_River": (snow_utils.dms(61, 11),  -1 * snow_utils.dms(145, 39)),
        "Tanana_River": (snow_utils.dms(65, 5), -1 * snow_utils.dms(145, 52)),
    }

    # 2016
    swep_16 = []
    swec_16 = []
    swet_16 = []

    # 2017
    swep_17 = []
    swec_17 = []
    swet_17 = []

    # 2018
    swep_18 = []
    swec_18 = []
    swet_18 = []

    # 2019
    swep_19 = []
    swec_19 = []
    swet_19 = []

    # 2020
    swep_20 = []
    swec_20 = []
    swet_20 = []

    # 2021
    swep_21 = []
    swec_21 = []
    swet_21 = []

    # 2022
    swep_22 = []
    swec_22 = []
    swet_22 = []

    # 2023
    swep_23 = []
    swec_23 = []
    swet_23 = []

    # 2024
    swep_24 = []
    swec_24 = []
    swet_24 = []

    # 2025
    swep_25 = []
    swec_25 = []
    swet_25 = []

    # For each watershed...
    for p in range(len(combined_watershed_data)):

        # Extract the Watershed
        that_aoi = aois[p]

        # extract the latitude and longitude associated with each Watershed
        lat, lon = points[aois[p]]

        # Extract the merged snow depth, SWE, and temp dataset associated with the AOI
        ds = combined_watershed_data[p]

        # IF the longitude is outside bounds
        if float(ds.longitude.max()) > 180:
            lon = lon % 360 
            
        # Extract the SWE variable only associated with the ERA5 Grid Cell overlaying the SNOTEL data
        swe_point = ds["snow_depth_water_equivalent_daily_mean"].sel(latitude=lat, longitude=lon, method='nearest')




        # For each SWE data point:
        for y in range(len(swe_point)):
            if 0 < y < 365:
                if that_aoi == 'Prudhoe_Bay':
                    swep_16.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_16.append(swe_point[y])
                else: 
                    swet_16.append(swe_point[y])
            if 365 < y < 730: 
                if that_aoi == 'Prudhoe_Bay':
                    swep_17.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_17.append(swe_point[y])
                else: 
                    swet_17.append(swe_point[y])
            if 730 < y < 1095: 
                if that_aoi == 'Prudhoe_Bay':
                    swep_18.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_18.append(swe_point[y])
                else: 
                    swet_18.append(swe_point[y])

            if 1095 < y < 1460: 
                if that_aoi == 'Prudhoe_Bay':
                    swep_19.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_19.append(swe_point[y])
                else: 
                    swet_19.append(swe_point[y])

            if 1460 < y < 1825:
                if that_aoi == 'Prudhoe_Bay':
                    swep_20.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_20.append(swe_point[y])
                else: 
                    swet_20.append(swe_point[y])

            if 1825 < y < 2190:
                if that_aoi == 'Prudhoe_Bay':
                    swep_21.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_21.append(swe_point[y])
                else: 
                    swet_21.append(swe_point[y])

            if 2190 < y < 2555: 
                if that_aoi == 'Prudhoe_Bay':
                    swep_22.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_22.append(swe_point[y])
                else: 
                    swet_22.append(swe_point[y])   

            if 2555 < y < 2920:
                if that_aoi == 'Prudhoe_Bay':
                    swep_23.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_23.append(swe_point[y])
                else: 
                    swet_23.append(swe_point[y])   

            if 2920 < y < 3285: 
                if that_aoi == 'Prudhoe_Bay':
                    swep_24.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_24.append(swe_point[y])
                else: 
                    swet_24.append(swe_point[y])

            if y > 3285:              
                if that_aoi == 'Prudhoe_Bay':
                    swep_25.append(swe_point[y])
                elif that_aoi == 'Copper_River': 
                    swec_25.append(swe_point[y])
                else: 
                    swet_25.append(swe_point[y])

        
    #peak_swe_prudhoe = np.max(np.array(swep_16))
    #print(f'Peak SWE for prudhoe: {peak_swe_prudhoe}')
    print(f"Length of array for {that_aoi} is {len(swe_point)}")
    print('='*70)

    print(len(swep_16))
    print(len(swet_16))
    print(len(swec_16))

    print(swep_16[:10])
    print(swec_16[:10])
    print(swet_16[:10])

    print(f'type is {type(swep_16)}')

    max_swe_prudhoe = []
    max_swe_tanana = []
    max_swe_copper = []

    arg_swe_prudhoe = []
    arg_swe_tanana = []
    arg_swe_copper = []

    # Mean and Standard Dev, 2016. 
    m_swep_16 = np.nanmax(np.array(swep_16))
    a_swep_16 = np.argmax(np.array(swep_16))
    m_swec_16 = np.nanmax(np.array(swec_16))
    a_swec_16 = np.argmax(np.array(swec_16))
    m_swet_16 = np.nanmax(np.array(swet_16))
    a_swet_16 = np.argmax(np.array(swet_16))

    # Mean and Standard Dev, 2017. 
    m_swep_17 = np.nanmax(np.array(swep_17))
    a_swep_17 = np.argmax(np.array(swep_17))
    m_swec_17 = np.nanmax(np.array(swec_17))
    a_swec_17 = np.argmax(np.array(swec_17))
    m_swet_17 = np.nanmax(np.array(swet_17))
    a_swet_17 = np.argmax(np.array(swet_17))

    # Mean and Standard Dev, 2018. 
    m_swep_18 = np.nanmax(np.array(swep_18))
    a_swep_18 = np.argmax(np.array(swep_18))
    m_swec_18 = np.nanmax(np.array(swec_18))
    a_swec_18 = np.argmax(np.array(swec_18))
    m_swet_18 = np.nanmax(np.array(swet_18))
    a_swet_18 = np.argmax(np.array(swet_18))

    # Mean and Standard Dev, 2019. 
    m_swep_19 = np.nanmax(np.array(swep_19))
    a_swep_19 = np.argmax(np.array(swep_19))
    m_swec_19 = np.nanmax(np.array(swec_19))
    a_swec_19 = np.argmax(np.array(swec_19))
    m_swet_19 = np.nanmax(np.array(swet_19))
    a_swet_19 = np.argmax(np.array(swet_19))

    # Mean and Standard Dev, 2020. 
    m_swep_20 = np.nanmax(np.array(swep_20))
    a_swep_20 = np.argmax(np.array(swep_20))
    m_swec_20 = np.nanmax(np.array(swec_20))
    a_swec_20 = np.argmax(np.array(swec_20))
    m_swet_20 = np.nanmax(np.array(swet_20))
    a_swet_20 = np.argmax(np.array(swet_20))

    # Mean and Standard Dev, 2021. 
    m_swep_21 = np.nanmax(np.array(swep_21))
    a_swep_21 = np.argmax(np.array(swep_21))
    m_swec_21 = np.nanmax(np.array(swec_21))
    a_swec_21 = np.argmax(np.array(swec_21))
    m_swet_21 = np.nanmax(np.array(swet_21))
    a_swet_21 = np.argmax(np.array(swet_21))

    # Mean and Standard Dev, 2021. 
    m_swep_22 = np.nanmax(np.array(swep_22))
    a_swep_22 = np.argmax(np.array(swep_22))
    m_swec_22 = np.nanmax(np.array(swec_22))
    a_swec_22 = np.argmax(np.array(swec_22))
    m_swet_22 = np.nanmax(np.array(swet_22))
    a_swet_22 = np.argmax(np.array(swet_22))

    # Mean and Standard Dev, 2021. 
    m_swep_23 = np.nanmax(np.array(swep_23))
    a_swep_23 = np.argmax(np.array(swep_23))
    m_swec_23 = np.nanmax(np.array(swec_23))
    a_swec_23 = np.argmax(np.array(swec_23))
    m_swet_23 = np.nanmax(np.array(swet_23))
    a_swet_23 = np.argmax(np.array(swet_23))

    # Mean and Standard Dev, 2024. 
    m_swep_24 = np.nanmax(np.array(swep_24))
    a_swep_24 = np.argmax(np.array(swep_24))
    m_swec_24 = np.nanmax(np.array(swec_24))
    a_swec_24 = np.argmax(np.array(swec_24))
    m_swet_24 = np.nanmax(np.array(swet_24))
    a_swet_24 = np.argmax(np.array(swet_24))

       # Mean and Standard Dev, 2021. 
    m_swep_25 = np.nanmax(np.array(swep_25))
    a_swep_25 = np.argmax(np.array(swep_25))
    m_swec_25 = np.nanmax(np.array(swec_25))
    a_swec_25 = np.argmax(np.array(swec_25))
    m_swet_25 = np.nanmax(np.array(swet_25))
    a_swet_25 = np.argmax(np.array(swet_25))


    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_16)
    arg_swe_prudhoe.append(a_swep_16)
    max_swe_copper.append(m_swec_16)
    arg_swe_copper.append(a_swec_16)
    max_swe_tanana.append(m_swet_16)
    arg_swe_tanana.append(a_swet_16)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_17)
    arg_swe_prudhoe.append(a_swep_17)
    max_swe_copper.append(m_swec_17)
    arg_swe_copper.append(a_swec_17)
    max_swe_tanana.append(m_swet_17)
    arg_swe_tanana.append(a_swet_17)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_18)
    arg_swe_prudhoe.append(a_swep_18)
    max_swe_copper.append(m_swec_18)
    arg_swe_copper.append(a_swec_18)
    max_swe_tanana.append(m_swet_18)
    arg_swe_tanana.append(a_swet_18)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_19)
    arg_swe_prudhoe.append(a_swep_19)
    max_swe_copper.append(m_swec_19)
    arg_swe_copper.append(a_swec_19)
    max_swe_tanana.append(m_swet_19)
    arg_swe_tanana.append(a_swet_19)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_20)
    arg_swe_prudhoe.append(a_swep_20)
    max_swe_copper.append(m_swec_20)
    arg_swe_copper.append(a_swec_20)
    max_swe_tanana.append(m_swet_20)
    arg_swe_tanana.append(a_swet_20)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_21)
    arg_swe_prudhoe.append(a_swep_21)
    max_swe_copper.append(m_swec_21)
    arg_swe_copper.append(a_swec_21)
    max_swe_tanana.append(m_swet_21)
    arg_swe_tanana.append(a_swet_21)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_22)
    arg_swe_prudhoe.append(a_swep_22)
    max_swe_copper.append(m_swec_22)
    arg_swe_copper.append(a_swec_22)
    max_swe_tanana.append(m_swet_22)
    arg_swe_tanana.append(a_swet_22)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_23)
    arg_swe_prudhoe.append(a_swep_23)
    max_swe_copper.append(m_swec_23)
    arg_swe_copper.append(a_swec_23)
    max_swe_tanana.append(m_swet_23)
    arg_swe_tanana.append(a_swet_23)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_24)
    arg_swe_prudhoe.append(a_swep_24)
    max_swe_copper.append(m_swec_24)
    arg_swe_copper.append(a_swec_24)
    max_swe_tanana.append(m_swet_24)
    arg_swe_tanana.append(a_swet_24)

    # Append Max and Day of Max to Respective Lists
    max_swe_prudhoe.append(m_swep_25)
    arg_swe_prudhoe.append(a_swep_25)
    max_swe_copper.append(m_swec_25)
    arg_swe_copper.append(a_swec_25)
    max_swe_tanana.append(m_swet_25)
    arg_swe_tanana.append(a_swet_25)

    mean_swemax_prudhoe = np.nanmean(np.array(max_swe_prudhoe)) 
    mean_swemax_copper = np.nanmean(np.array(max_swe_copper))
    mean_swemax_tanana = np.nanmean(np.array(max_swe_tanana))

    max_swemax_prudhoe = np.nanmax(np.array(max_swe_prudhoe))
    max_swemax_copper = np.nanmax(np.array(max_swe_copper))
    max_swemax_tanana = np.nanmax(np.array(max_swe_tanana))

    min_swemax_prudhoe = np.nanmin(np.array(max_swe_prudhoe))
    min_swemax_copper = np.nanmin(np.array(max_swe_copper))
    min_swemax_tanana = np.nanmin(np.array(max_swe_tanana))

    std_swemax_prudhoe = np.nanstd(np.array(max_swe_prudhoe))
    std_swemax_copper = np.nanstd(np.array(max_swe_copper))
    std_swemax_tanana = np.nanstd(np.array(max_swe_tanana))


    mean_sweday_prudhoe = np.mean(np.array(arg_swe_prudhoe))
    mean_sweday_copper = np.mean(np.array(arg_swe_copper))
    mean_sweday_tanana = np.mean(np.array(arg_swe_tanana))

    max_sweday_prudhoe = np.max(np.array(arg_swe_prudhoe))
    max_sweday_copper = np.max(np.array(arg_swe_copper))
    max_sweday_tanana = np.max(np.array(arg_swe_tanana))

    min_sweday_prudhoe = np.min(np.array(arg_swe_prudhoe))
    min_sweday_copper = np.min(np.array(arg_swe_copper))
    min_sweday_tanana = np.min(np.array(arg_swe_tanana))

    std_sweday_prudhoe = np.std(np.array(arg_swe_prudhoe))
    std_sweday_copper = np.std(np.array(arg_swe_copper))
    std_sweday_tanana = np.std(np.array(arg_swe_tanana))

    pru_sno_mean = 413.1 / 1000
    pru_sno_std = 44.8 / 1000
    pru_sno_min = 376.3 / 1000
    pru_snow_max = 527.2 / 1000

    tan_sno_mean = 198.4 / 1000
    tan_sno_std = 56.1 / 1000
    tan_sno_min = 106.7 / 1000
    tan_sno_max = 315.0 / 1000

    cop_sno_mean = 579.5 / 1000
    cop_sno_std = 121.4 / 1000
    cop_sno_min = 447.0 / 1000
    cop_sno_max = 886.5/ 1000

    pru_doy_mean = 128 # May 08 
    pru_doy_std = 11 
    pru_doy_min = 110 #Apr 20
    pru_doy_max = 140 # May 20 

    tan_doy_mean = 98 # Apr 08 
    tan_doy_std = 14.7 
    tan_doy_min = 66 # Mar 07 
    tan_doy_max = 111 # Apr 21

    cop_doy_mean = 114 # Apr 24 
    cop_doy_std = 6.2 
    cop_doy_min = 101 # Apr 11 
    cop_doy_max = 121 #May 01


    swe_stats = [
        snow_utils.box_stats(mean_swemax_prudhoe, std_swemax_prudhoe, min_swemax_prudhoe, max_swemax_prudhoe),
        snow_utils.box_stats(pru_sno_mean, pru_sno_std, pru_sno_min, pru_sno_max), 
        snow_utils.box_stats(mean_swemax_copper, std_swemax_copper, min_swemax_copper, max_swemax_copper),
        snow_utils.box_stats(cop_sno_mean, cop_sno_std, cop_sno_min, cop_sno_max)
    ]

    print(f'Mean SWEmax Prudhoe: {mean_swemax_prudhoe:0.3f}')
    print(f'Mean SWEmax Copper: {mean_swemax_copper:0.3f}')
    print(f'Mean SWEMax Tanana: {mean_swemax_tanana:0.3f}')

    print(f'Max SWEmax Prudhoe: {mean_swemax_prudhoe:0.3f}')
    print(f'Max SWEmax Copper: {mean_swemax_copper:0.3f}')
    print(f'Max SWEMax Tanana: {mean_swemax_tanana:0.3f}')

    print(f'Mean DayMax Prudhoe: {mean_sweday_prudhoe:0.0f}')
    print(f'Mean DayMax Copper: {mean_sweday_copper:0.0f}')
    print(f'Mean DayMax Tanana: {mean_sweday_tanana:0.0f}')

    #print(f'Max Prudhoe 2016: {m_swep_16:0.3f}')
    #print(f'Day of Max Prudhoe 2016: {a_swep_16:0f}')

    #print(f'Max Copper 2016: {m_swec_16:0.3f}')
    #print(f'Day of Max Copper 2016: {a_swec_16:0.3f}')

    #print(f'Max Tanana 2016: {m_swet_16:0.3f}')
    #print(f'Day of Max Tanana 2016: {a_swet_16:0.3f}')

    ####################### Step 4a Part 3: Combined Time Series ################
    print('Creating a single plot showing the temperature and snow depth for all watersheds')
    print(f'len sd_sr: {len(sd_sr)}')

    combined_fig_name = sr_fig_folder + 'combined_time_series.png'

    # Create colors for different watersheds
    line_colors = ['red', 'black', 'blue']

    fig, ax1 = plt.subplots(figsize = (10, 5))     # Create Subplot Figure
    ax2 = ax1.twinx() 


    for m in range(len(sd_sr)):
        #my_aoi = aois[m]
        my_label = aoi_labels[m]
        my_sd = sd_sr[m]
        my_temp = temp_sr[m]
        my_color = line_colors[m]

        temp_C = my_temp - 273.15 

        my_sd_smooth = my_sd.rolling(time = 30, center = True, min_periods = 1).mean()
        my_temp_smooth = temp_C.rolling(time = 30, center = True, min_periods = 1).mean()

        # We will use dotted lines for snow depth and solid lines for temperature
        temp_label = f'{my_label} temperature'
        snow_label = f'{my_label} snow_depth'

        ax1.plot(my_sd_smooth["time"], my_sd_smooth, color = my_color, linestyle = 'dotted', label = snow_label)
        ax2.plot(my_temp_smooth["time"], my_temp_smooth, color = my_color, linestyle = 'solid', label = temp_label)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Save Figure and Close
    ax1.set_ylabel('Snow depth (m)')               # Set Left y-label
    ax1.tick_params(axis="y")                      # Add ticks on left y-label
    ax1.grid(True, alpha = 0.4)
    ax1.set_xlabel('Year')

    ax2.invert_yaxis()
    ax2.grid(False)
    ax2.set_ylabel('Temperature (°C)')

    # Create a legend from both axes 
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(h1 + h2, l1 + l2, loc = 'upper center', 
               bbox_to_anchor = (0.5, -0.12), ncol = 3, frameon = False)

    ax1.set_title(f"Snow Depth and Temperature since 2016")
    plt.grid()
    plt.savefig(combined_fig_name, dpi = 300, bbox_inches = "tight")

    ################ Step 4a Part 4: Master Plot #####################
    print("Step 4a Part 4: Combined Display of Snow depth, SWE, and snow-on snow-off dates")
    swesd_fig_name = sr_fig_folder + 'swe_and_sd_time_series.png'

    fig2, axa = plt.subplots(figsize = (10, 5))     # Create Subplot Figure
    axb = axa.twinx() 

    for q in range(len(sd_sr)):
        #this_aoi = aois[q]
        this_label = aoi_labels[q]
        this_sd = sd_sr[q]
        this_swe = swe_sr[q]
        this_color = line_colors[q]

        this_sd_smooth = this_sd.rolling(time=30, center = True, min_periods = 1).mean()
        this_swe_smooth = this_swe.rolling(time=30, center = True, min_periods = 1).mean() 

        this_sd_label = f'{this_label} snow depth'
        this_swe_label = f'{this_label} snow water equivalent'

        axa.plot(this_sd_smooth["time"], this_sd_smooth, color = this_color, linestyle = 'dotted', label = this_sd_label)
        axb.plot(this_swe_smooth["time"], this_swe_smooth, color = this_color, linestyle = 'dashed', label = this_swe_label)
        axa.xaxis.set_major_locator(mdates.YearLocator())
        axa.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axa.set_ylabel('Snow depth (m)')               # Set Left y-label
    axa.tick_params(axis="y")                      # Add ticks on left y-label
    axa.grid(True, alpha = 0.4)
    axa.set_xlabel('Year')

    axa.grid(False)
    axb.set_ylabel('Snow Water Equivalent (m)')


    # Create a legend from both axes 
    ha, la = axa.get_legend_handles_labels()
    hb, lb = axb.get_legend_handles_labels()

    axa.legend(ha + hb, la + lb, loc = 'upper center', 
               bbox_to_anchor = (0.5, -0.1), ncol = 3, frameon = False)

    axa.set_title(f"Snow Depth and Snow Water Equivalent")
    plt.grid()
    plt.tight_layout()
    plt.savefig(swesd_fig_name, dpi = 300, bbox_inches = "tight")

    print(f"Snow Depth and SWE figure printed to {swesd_fig_name}")

    ################ Step 4a Part 4:  Snow-on Snow-off date #########################
   # snow_flag = snow_flag["snow_flag"]

    #print(f' Length of snow_flag for {year} is {len(snow_flag)}')

    #break()

    ################ Step 4a Part 5: Analysis of Trends ################ 
    print("Step 4a Part 5: Analysis of Trends")
    
    # Now we want to extract each individual year time series. 
    # For each yearly watershed time series, we want to...
    # find the Julian Day of the maximum snow_depth 
    # find the Julian Day of the minimum temperature 
    # Find the time lag between the minimum temperature and max snow depth

    # Create arrays to append the min and max days into 

    copper_temp_min_days = []
    copper_snow_max_days = []
    tanana_temp_min_days = []
    tanana_snow_max_days = []
    prudhoe_temp_min_days = []
    prudhoe_snow_max_days = []

    # Snow depth water year arrays for each watershed (WY2017..WY2026)
    copper_sd_wys  = [[] for _ in range(10)]
    prudhoe_sd_wys = [[] for _ in range(10)]
    tanana_sd_wys  = [[] for _ in range(10)]

    # Temperature water year arrays
    copper_t_wys  = [[] for _ in range(10)]
    prudhoe_t_wys = [[] for _ in range(10)]
    tanana_t_wys  = [[] for _ in range(10)]

    watersheds = ['Copper_River', 'Prudhoe_Bay', 'Tanana_River']

    # For each file in subsetted data folder 
    print(f'Min/Max analysis')

    for u in sorted(Path(spatial_subset_folder).iterdir()): 
        #print(u)
        splitter = str(u).split('/')[-1]
        shed = splitter[:-8]
        year = splitter[-7:-3]

        #print(f'watershed: {shed}')
        count = int(year) - 2017
        #print(f'count: {count}')
        #print(year)

        # Load the individual dataset
        data = data_analysis.load_individual_data(u)

        # Extract the depth dataset and run a spatial reducer on it 
        depth_redux = data_analysis.spatial_reduce(data, "snow_depth_daily_mean")
        #print(shed, year, len(depth_redux))

        # Extract the temperature dataset and run a spatial reducer on it
        t_redux = data_analysis.spatial_reduce(data, "2m_temperature_daily_mean")

        #print(f'type depth redux: {type(depth_redux)}')
        #print(f'type temp redux: {type(t_redux)}')


        # For each entry in the depth_redux and t_redux array...
        for i in range(len(depth_redux)): 
            # If we are before October 1st, 
            if i < 274:                     
                # If the year is 2016, do nothing 
                if int(year) == 2016:
                    pass
                else: 
                    if shed not in watersheds: 
                        raise ValueError('watershed not in list!')
                    if shed == 'Copper_River':
                        # Append the depth and temp values to their respective arrays
                        sd_list = copper_sd_wys[count]
                        sd_list.append(depth_redux[i])

                        t_list = copper_t_wys[count]
                        t_list.append(t_redux[i])
                    if shed == 'Prudhoe_Bay':
                        prudhoe_sd_wys[count].append(depth_redux[i])
                        prudhoe_t_wys[count].append(t_redux[i])
                    if shed == 'Tanana_River': 
                        tanana_sd_wys[count].append(depth_redux[i])
                        tanana_t_wys[count].append(t_redux[i])
            # If we are after October 1st: 
            else:
                # If the year is 2026, do nothing 
                if int(year) == 2026:
                    pass 
                else: 
                    # Identify watershed 
                    if shed not in watersheds:
                        raise ValueError("Watershed not found")
                    if shed == 'Copper_River':
                        # Append the depth and temp_values to their respective arrays
                        copper_sd_wys[count+1].append(depth_redux[i])
                        copper_t_wys[count+1].append(t_redux[i])
                    if shed == 'Tanana_River':
                        tanana_sd_wys[count+1].append(depth_redux[i])
                        tanana_t_wys[count+1].append(t_redux[i])
                    if shed == 'Prudhoe_Bay':
                        prudhoe_sd_wys[count+1].append(depth_redux[i])
                        prudhoe_t_wys[count+1].append(t_redux[i])   

    wys = ['WY2017', 'WY2018', 'WY2019', 'WY2020', 'WY2021', 'WY2022', 'WY2023', 'WY2024', 'WY2025', 'WY2026']

    copper_max_depth_days= []
    copper_min_temp_days = []
    tanana_max_depth_days = []
    tanana_min_temp_days = []
    prudhoe_max_depth_days = []
    prudhoe_min_temp_days = []

    # For each of three watersheds...
    for k in watersheds: 
        # For each water year...
        for w in range(len(copper_sd_wys)):
            if k == 'Copper_River':
                max_depth_day = np.argmax(np.array(copper_sd_wys[w]))
                min_temp_day = np.argmin(np.array(copper_t_wys[w]))

                copper_max_depth_days.append(max_depth_day)
                copper_min_temp_days.append(min_temp_day)

            if k == 'Tanana_River':
                max_depth_day = np.argmax(np.array(tanana_sd_wys[w]))
                min_temp_day = np.argmin(np.array(tanana_t_wys[w]))

                tanana_max_depth_days.append(max_depth_day)
                tanana_min_temp_days.append(min_temp_day)

            if k == 'Prudhoe_Bay':
                max_depth_day = np.argmax(np.array(prudhoe_sd_wys[w]))
                min_temp_day = np.argmin(np.array(prudhoe_t_wys[w]))

                prudhoe_max_depth_days.append(max_depth_day)
                prudhoe_min_temp_days.append(min_temp_day)

    print(f't max depth days: {tanana_max_depth_days}')
    print(f't min temp days: {tanana_min_temp_days}')

    tanana_mean_sdmax = np.array(tanana_max_depth_days).mean()
    tanana_dev_sdmax = np.array(tanana_max_depth_days).std()
    tanana_mean_tmin = np.array(tanana_min_temp_days).mean() 
    tanana_dev_tmin = np.array(tanana_min_temp_days).std() 

    copper_mean_sdmax = np.array(copper_max_depth_days).mean()
    copper_dev_sdmax = np.array(copper_max_depth_days).std()
    copper_mean_tmin = np.array(copper_min_temp_days).mean() 
    copper_dev_tmin = np.array(copper_min_temp_days).std() 

    prudhoe_mean_sdmax = np.array(prudhoe_max_depth_days).mean()
    prudhoe_dev_sdmax = np.array(prudhoe_max_depth_days).std()
    prudhoe_mean_tmin = np.array(prudhoe_min_temp_days).mean() 
    prudhoe_dev_tmin = np.array(prudhoe_min_temp_days).std() 

    print('='*70)
    print('min/max results')
    print(f'copper average minimum temp day: {int(copper_mean_tmin)}')
    print(f'copper average max depth day: {int(copper_mean_sdmax)}')
    print(f'copper average delay: {int(copper_mean_sdmax - copper_mean_tmin)}')
    print('\n')
    print(f'copper stdev min temp day: {int(copper_dev_tmin)}')
    print(f'copper stdev max depth day: {int(copper_dev_sdmax)}')
    print('--')

    print(f'tanana average minimum temp day: {int(tanana_mean_tmin)}')
    print(f'tanana average max depth day: {int(tanana_mean_sdmax)}')
    print(f'tanana average delay: {int(tanana_mean_sdmax - tanana_mean_tmin)}')
    print('\n')
    print(f'tanana stdev min temp day: {int(tanana_dev_tmin)}')
    print(f'tanana stdev max depth day: {int(tanana_dev_sdmax)}')
    print('--')

    print(f'prudhoe average minimum temp day: {int(prudhoe_mean_tmin)}')
    print(f'prudhoe average max depth day: {int(prudhoe_mean_sdmax)}')
    print(f'prudhoe average delay: {int(prudhoe_mean_sdmax - prudhoe_mean_tmin)}')
    print('\n')

    print(f'prudhoe stdev min temp day: {int(prudhoe_dev_tmin)}')
    print(f'prudhoe stdev max depth day: {int(prudhoe_dev_sdmax)}')
    print('='*70)




    


    #for j in range(len(sd_wys)): 
    #    max_depth_day = np.argmax(np.array(sd_wys[j]))
    #    min_temp_day = np.argmin(np.array(t_wys[j]))

    #    print(f'stats for {wys[j]}')
    #    print(max_depth_day)
    #    print(min_temp_day)


        #max_depth_17 = np.argmax(np.array(sd_wy_17))
        #min_t_17 = np.argmin(np.array(t_wy_17))

        #print(f'Max depth WY 2016: {max_depth_17}')
        #print(f'Min temp WY 2017: {min_t_17} ')

        #depth_max = depth_redux.argmax().item()
        #temp_min = t_redux.argmin().item()

        # Extract the watershed and year from the file name
        #splitter = str(u).split('/')[-1]
        #this_watershed = splitter[:-8]
        #this_year = splitter[-7:-3]

        
        #print(f'Julian Day of min temp for {this_watershed} on {this_year}: {temp_min}')
        #print(f'Julian Day of Maximum snow depth for {this_watershed} on {this_year}: {depth_max}')
        #print('\n')

    #copper_mean_snow_day = np.array(copper_snow_max_days).mean() 
    #copper_mean_temp_day = np.array(copper_temp_min_days).mean()
    #tanana_mean_snow_day = np.array(tanana_snow_max_days).mean() 
    #tanana_mean_temp_day = np.array(tanana_temp_min_days).mean()
    #prudhoe_mean_snow_day = np.array(prudhoe_snow_max_days).mean()
    #prudhoe_mean_temp_day = np.array(prudhoe_snow_max_days).mean()

    #print(f'Mean Julian Day of Max Snow Depth in Prudhoe Bay Watershed: {prudhoe_mean_snow_day}')



    #############################################################
    #               Step 4B: Temporal Reduction                 #
    #############################################################
    # In this step we will create a sngle map for each year, and for each watershed 
    # to display the average snow depth. We will also make one for the whole time series

    #### Part 1: Create a spatial representation of the entire time series for each watershed ###

    # Create a folder to host geotiff files
    geotiff_folder = str(Path(spatial_subset_folder).parent) + '/geotiffs'
    os.makedirs(geotiff_folder, exist_ok=True)

    # Create a folder to host png image files 
    png_folder = str(Path(spatial_subset_folder).parent) + '/images'
    os.makedirs(png_folder, exist_ok=True)

    for r in range(len(combined_watershed_data)):
        watershed_data = combined_watershed_data[r]
        #print(len(watershed_data))
        my_watershed = aois[r]

        # Temporally Reduced Snow Depth 
        sd_temp_redux = data_analysis.temporal_reduce(watershed_data, "snow_depth_daily_mean")

        ############### Make Tif files  ######################
        geotiff_path = geotiff_folder + f'/{my_watershed}_redux_2017_2026.tif'
        print(f'Writing Temp Redux GeoTiff to: {geotiff_path}')
        data_analysis.da_to_geotiff(sd_temp_redux, geotiff_path)

        ################## Make PNG files #######################
        stem = f'{my_watershed}_2017_2026'
        png_path = png_folder + f'/{stem}.png'
        
        print(f'Writing Temp Redux info from {geotiff_path} PNG to: {png_path}')
        data_analysis.geotiff_2_png(geotiff_path, png_path, stem)                    # Create PNG from Geotiff file

    ##### Step 2: Create Geotiffs and pngs for individual years and individual watersheds #######
    for k in Path(spatial_subset_folder).iterdir(): 
        #print(k)
        #print(type(k))

        # Load individual watershed as xarray
        ind_sd = data_analysis.load_individual_data(k)

        # Temporally reduce individual dataset 
        ind_redux = data_analysis.temporal_reduce(ind_sd, "snow_depth_daily_mean")

        tif_fname = geotiff_folder + '/' + str(k).split('/')[-1][:-3] + '.tif'
        #print(tif_fname)

        print(f'Writing Temp Redux GeoTiff to {tif_fname}')
        data_analysis.da_to_geotiff(ind_redux, tif_fname)


    



    



    ####################################################################
    #               Step 2c: Temporally Reducing Data                  # 
    #                                                                  #
    #         This will produce a map for each 
    ####################################################################

        
    #print(f"Processing {aois[0]} data")
    #copper = data_analysis.load_watershed_data(str(aois[0]), spatial_subset_folder)

    #print(f"Processing {aois[1]} data")
    #tanana = data_analysis.load_watershed_data(str(aois[1]), spatial_subset_folder)

    #print(f"Processing aoi")
    #prudhoe = data_analysis.load_watershed_data(str(aois[2]), spatial_subset_folder)

    #print(f'type copper: {type(copper)}')


if __name__ == "__main__":
    main()