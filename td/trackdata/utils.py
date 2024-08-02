
# Compute means and standard deviations of the features in the training set
from td.utils import get_config
import pandas as pd
from pathlib import Path

def compute_means_and_stds():
    
    df_list = get_df_list(slice(0, 8))

    df = pd.concat(df_list)

    mean_lat = df.lat.mean()
    std_lat = df.lat.std()

    mean_lon = df.lon.mean()
    std_lon = df.lon.std()

    mean_vmax = df.vmax.mean()
    std_vmax = df.vmax.std()

    mean_pmin = df.pmin.mean()
    std_pmin = df.pmin.std()

    mean_rmw = df.rmw.mean()
    std_rmw = df.rmw.std()

    mean_r18 = df.r18.mean()
    std_r18 = df.r18.std()

    return {
        "lat": (mean_lat, std_lat), 
        "lon": (mean_lon, std_lon), 
        "vmax": (mean_vmax, std_vmax), 
        "pmin": (mean_pmin, std_pmin), 
        "rmw": (mean_rmw, std_rmw), 
        "r18": (mean_r18, std_r18)
            }

def get_df_list(slice_range):
    
    df_list = []
    for i in range(slice_range.start, slice_range.stop):
        data_path = Path(get_config().data_folder)
        data_file = data_path / f"IRIS_NA_1000Y_n{i}.txt"
        df = pd.read_csv(data_file, sep=" ", header=0)
        df = [d for _, d in df.groupby("#tcid")]
        df_list.extend(df)
    return df_list