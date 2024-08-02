from pathlib import Path
import numpy as np
import pandas as pd
import torch
from td.trackdata.utils import compute_means_and_stds, get_df_list
from torch.utils.data import Dataset


class TrackData(Dataset):
    def __init__(self, data_folder, train=True):
        
        self.data_folder = Path(data_folder)
        if train:
            self.df_list = get_df_list(slice(0, 8))
        else:
            self.df_list = get_df_list(slice(8, 10))

        self.get_stats()

        self.max_steps = 200 # max([len(d) for d in df_list]) #211
        
    def __len__(self):
        return len(self.df_list)


    def __getitem__(self,idx):
        
        data = self.df_list[idx]
        
        data = self.norm(data)

        ind_max = min(self.max_steps, len(data))

        

        y = -99 + np.zeros((6, self.max_steps))

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

        lon_noise = np.random.normal(0, 0.1, 1)
        lat_noise = np.random.normal(0, 0.1, 1)
        
        return lon + lon_noise[0], lat + lat_noise[0]

    
    def get_stats(self):

        stats = compute_means_and_stds()

        self.lat_stats = stats["lat"]
        self.lon_stats = stats["lon"]
        self.vmax_stats = stats["vmax"]
        self.pmin_stats = stats["pmin"]
        self.rmw_stats = stats["rmw"]
        self.r18_stats = stats["r18"]

    def norm(self, df):
        df.lat = (df.lat - self.lat_stats[0]) / self.lat_stats[1]
        df.lon = (df.lon - self.lon_stats[0]) / self.lon_stats[1]
        df.vmax = (df.vmax - self.vmax_stats[0]) / self.vmax_stats[1]
        df.pmin = (df.pmin - self.pmin_stats[0]) / self.pmin_stats[1]
        df.rmw = (df.rmw - self.rmw_stats[0]) / self.rmw_stats[1]
        df.r18 = (df.r18 - self.r18_stats[0]) / self.r18_stats[1]
        return df

    def unnorm(self, df):
        df.lat = df.lat * self.lat_stats[1] + self.lat_stats[0]
        df.lon = df.lon * self.lon_stats[1] + self.lon_stats[0]
        df.vmax = df.vmax * self.vmax_stats[1] + self.vmax_stats[0]
        df.pmin = df.pmin * self.pmin_stats[1] + self.pmin_stats[0]
        df.rmw = df.rmw * self.rmw_stats[1] + self.rmw_stats[0]
        df.r18 = df.r18 * self.r18_stats[1] + self.r18_stats[0]
        return df
        