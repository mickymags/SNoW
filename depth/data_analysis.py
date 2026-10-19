from pathlib import Path
import xarray as xr

input_data_dir = Path('/home/mmaganini/Documents/ess601/era5_project/era5_output_data')

for era5_subset in sorted(input_data_dir.iterdir()):
    print(str(era5_subset))



#for j in input_data_dir():
