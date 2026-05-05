# svd env
import xarray as xr
import os 
import numpy as np
from tqdm import tqdm
import pandas as pd 
import cdsapi
import torch
import pickle

start = 1940
end = 2025 # exclusive
hrs_per_year = 1460 # 6hrly

variables = {'slp': ('/mnt/data/sonia/cyclone/0.25/slp/slp', 'slp', None),
             'u_component_of_wind': ('/mnt/data/sonia/cyclone/0.25/wind_500hpa/wind_500hpa', 'u', 500), # 500
             'v_component_of_wind': ('/mnt/data/sonia/cyclone/0.25/wind_500hpa/wind_500hpa', 'v', 500), # 500
             'temperature': ('/mnt/data/sonia/cyclone/0.25/temperature/temperature', 't', 925), # 925
             'specific_humidity': ('/mnt/data/sonia/cyclone/0.25/humidity/humidity', 'q', 500), # 500
            }

out_dir = '/mnt/data/sonia/climax-data/train-raw'
os.makedirs(out_dir, exist_ok=True)
data = np.memmap(os.path.join(out_dir, 'train_data.dat'), dtype=np.float32, mode='w+', 
               shape=((end-start)*hrs_per_year, len(variables), 721, 1440))

for yr in tqdm(range(1940, 2025)):
    yr_vars = []
    for i, (var_name, (var_path, short_name, level)) in enumerate(variables.items()):
        ds = xr.open_dataset(f'{var_path}.{yr}.nc')
        if level is not None:
            ds = ds.sel(pressure_level=level)
        ds = ds.transpose("time", "lat", "lon") # just in case
        print(var_name, ds.to_array().shape, ds)
        data[(yr-start)*hrs_per_year:(yr-start+1)*hrs_per_year, i] = \
            ds[short_name].values[:hrs_per_year].astype(np.float32) # skip last day if leap year
        
    if yr % 10 == 0:
        data.flush()

dataset_mmap.flush()


# constants file
nc_path = os.path.join(out_dir, 'constants.nc')

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'land_sea_mask',
            'geopotential', 
            'soil_type',
        ],
        'year': '1940',
        'month': '01',
        'day': '01',
        'time': '00:00',
        'format': 'netcdf',
        'grid': '0.25/0.25', # 721x1440
    },
    nc_path
)

ds = xr.open_dataset(nc_path).load()

# Squeeze out the single time dimension so arrays are exactly 2D: (lat, lon)
if "time" in ds.dims:
    ds = ds.squeeze("time")
if "valid_time" in ds.coords or "valid_time" in ds.dims:
    ds = ds.drop_vars("valid_time", errors="ignore")

# Extract the numpy arrays using ERA5's default short names
# Cast to float32 to save memory and match standard PyTorch tensor precision
aurora_static_dict = {
    "lsm": ds["lsm"].values.astype(np.float32),
    "z": ds["z"].values.astype(np.float32),
    "slt": ds["slt"].values.astype(np.float32)
}

# Save as a pickle file
static_out_path = os.path.join(out_dir, 'aurora-0.25-static.pickle')
with open(static_out_path, "wb") as f:
    pickle.dump(aurora_static_dict, f)

print(f"Aurora static variables saved to {static_out_path}")
print(f"Shapes: z={aurora_static_dict['z'].shape}, lsm={aurora_static_dict['lsm'].shape},", 
        f"slt={aurora_static_dict['slt'].shape}")
