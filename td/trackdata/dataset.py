from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrackData(Dataset):
    def __init__(self, user_cfg, fold="train"):
        
        self.data_folder = Path(user_cfg.data_folder)

        if fold=="train": # First 8 files
            self.df_list = self.get_df_list(slice(0, 8))
        elif fold=="valid": # Next 1 file
            self.df_list = self.get_df_list(slice(8, 9))
        elif fold=="test": # Last 1 file
            self.df_list = self.get_df_list(slice(9, 10))
        else:
            KeyError, "fold must be one of 'train', 'val', 'test'"

        self.get_stats()

        self.max_steps = 200 # max([len(d) for d in df_list]) #211
        
    def __len__(self):
        return len(self.df_list)


    def __getitem__(self,idx):
        
        data = self.df_list[idx].copy()
        data = self.norm(data)

        ind_max = min(self.max_steps, len(data))

        y = np.nan * np.zeros((6, self.max_steps))
        y[0, :ind_max] = data.lon.values[:ind_max]
        y[1, :ind_max] = data.lat.values[:ind_max]
        y[2, :ind_max] = data.vmax.values[:ind_max]
        y[3, :ind_max] = data.pmin.values[:ind_max]
        y[4, :ind_max] = data.rmw.values[:ind_max]
        y[5, :ind_max] = data.r18.values[:ind_max]
        
        # month encoded with sine and cosine
        
        month_sin = np.sin(2*np.pi*data.month.values[0]/12)
        month_cos = np.cos(2*np.pi*data.month.values[0]/12)
        

        ind_lon_lat = int(ind_max//2)
        lon, lat = self.random_noise_lat_lon(data.lon.values[ind_lon_lat], data.lat.values[ind_lon_lat])
        
        x = np.array([month_sin, month_cos, lon, lat])
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y

    def random_noise_lat_lon(self, lon:int, lat:int):

        lon_noise = np.random.normal(loc = 0, scale=0.005, size=(1,))
        lat_noise = np.random.normal(loc = 0, scale=0.005, size=(1,))
        
        return lon + lon_noise[0], lat + lat_noise[0]

    
    def get_stats(self):

        stats = self.compute_means_and_stds()

        self.lat_stats = stats["lat"]
        self.lon_stats = stats["lon"]
        self.vmax_stats = stats["vmax"]
        self.pmin_stats = stats["pmin"]
        self.rmw_stats = stats["rmw"]
        self.r18_stats = stats["r18"]

    def norm(self, df):
        min_lat = -90
        max_lat = 90
        min_lon = -180
        max_lon = 180

        df.lat = (df.lat - min_lat) / (max_lat - min_lat)
        df.lon = (df.lon - min_lon) / (max_lon - min_lon)
        df.vmax = (df.vmax - self.vmax_stats[0]) / self.vmax_stats[1]
        df.pmin = (df.pmin - self.pmin_stats[0]) / self.pmin_stats[1]
        df.rmw = (df.rmw - self.rmw_stats[0]) / self.rmw_stats[1]
        df.r18 = (df.r18 - self.r18_stats[0]) / self.r18_stats[1]
        return df

    def denorm(self, df):
        df.lat = df.lat * self.lat_stats[1] + self.lat_stats[0]
        df.lon = df.lon * self.lon_stats[1] + self.lon_stats[0]
        df.vmax = df.vmax * self.vmax_stats[1] + self.vmax_stats[0]
        df.pmin = df.pmin * self.pmin_stats[1] + self.pmin_stats[0]
        df.rmw = df.rmw * self.rmw_stats[1] + self.rmw_stats[0]
        df.r18 = df.r18 * self.r18_stats[1] + self.r18_stats[0]
        return df
    
    def get_df_list(self, slice_range):
    
        df_list = []
        for i in range(slice_range.start, slice_range.stop):
            data_path = self.data_folder
            data_file = data_path / f"IRIS_NA_1000Y_n{i}.txt"
            df = pd.read_csv(data_file, sep=" ", header=0)
            df = [d for _, d in df.groupby("#tcid")]
            df_list.extend(df)
        return df_list

    def compute_means_and_stds(self,):
    
        df_list = self.get_df_list(slice(0, 8))

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