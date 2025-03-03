# import necessary packages

import datetime
t0=datetime.datetime.now()
import numpy as np
import xarray as xr
from glob import glob
import random
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import math
from scipy import stats
from scipy.stats import rankdata
import dask.array as da
from sklearn.neighbors import KDTree

with open("output2.txt", "a") as f:
    print("This code is running now", file=f)
  
############################################ Sample Data
samplenat = xr.open_dataset('hadam3p_anz_....nc') # import dataset
g_lat = samplenat['global_latitude0'].values
g_lon = samplenat['global_longitude0'].values
lat = samplenat['latitude0'].values
lon = samplenat['longitude0'].values

# directory
mdname = '.......'
d = '.....'

############################################ SOME FUNCTIONS
def read_model_data():
    md = xr.open_dataset(d+'....nc')
    md = md['precipitation'].assign_coords(latitude=lat, longitude=lon)
    md = md.stack(days=('ensemble_number','time'))
    md = md.reset_index('days')
    return md

def read_mswep():
    m = xr.open_dataset(mdname+'rf_mswep.nc')
    m = m['precipitation']
    m = m.assign_coords(latitude=lat, longitude=lon)
    m = m.stack(days=('ensemble_no','match'))
    m = m.reset_index('days')
    return m

def get_matchingper(a,b):
    a_values = a.values
    b_values = b.values

    a_mask = ~np.isnan(a_values)
    b_mask = ~np.isnan(b_values)

    a_values_filtered = a_values[a_mask]
    b_values_filtered = b_values[b_mask]

    tree = KDTree(b_values_filtered.reshape(-1, 1))

    bmp = np.empty(len(a_values), dtype=np.float64)
    bmp[:] = np.nan

    for i in range(len(a_values)):
        if a_mask[i]:
            _, ind = tree.query(a_values[i].reshape(1, -1), k=1)
            bmp[i] = b_values_filtered[ind]

    bmp = xr.DataArray(bmp, dims=('days',), name='bm percentiles', coords={'days': range(len(a_values))})
    return bmp


with open("output2.txt", "a") as f:
    print("Moving to main part", file=f)

################################################ MAIN FUNCTION
def do_bc(model_data, mst):
    t0=datetime.datetime.now()
    zz = 0
    
    mds = model_data
    ms = mst
    
    bc_md = np.full([mds.shape[0], mds.shape[1], mds.shape[2]], np.nan)
    cvs_a = np.full([mds.shape[0], mds.shape[1], mds.shape[2]], np.nan)
        
    def antcalculate_percentile(value):
        if not np.isnan(value):
            p_rank = stats.percentileofscore(non_nan_mds, value, kind='rank')
            return p_rank
        else:
            return np.nan
    
    for i in range(ants.shape[0]):
        for j in range(ants.shape[1]):
            pers_a = np.full([mds.shape[2]], np.nan)
                        
            mds1 = mds[i,j,:]
            ms1 = ms[i,j,:]
        
            non_nan_mds = mds1[~np.isnan(mds1)]
            
            if len(non_nan_mds) > 0:
                ants_p = np.vectorize(antcalculate_percentile)(mds1)
                ants_p = xr.DataArray(ants_p, dims=ants1.dims, coords=ants1.coords)
        
                mswep = np.array(ms1)
                ants_p = np.array(ants_p)
                dummy_mswep = np.array(ants_p)
        
                non_nan_indices_p = ~np.isnan(ants_p)
                non_nan_indices_ms = ~np.isnan(mswep)
        
                percentiles = np.percentile(mswep[non_nan_indices_ms], ants_p[non_nan_indices_p])
                nearest_indices = np.abs(mswep[non_nan_indices_ms, np.newaxis] - percentiles).argmin(axis=0)
                closest_values = mswep[non_nan_indices_ms][nearest_indices]
                dummy_mswep[non_nan_indices_p] = closest_values
        
                mswep_p = xr.DataArray(dummy_mswep, dims=('days',), name='precipitation', coords={'days': range(len(ants_p))})
        
                cval = ants1 - mswep_p
                bc_ant[i,j,:] = ants1 - cval ##### cavl means correction values
                pers_a[:] = ants_p
                cvs_a[i,j,:] = cval
                
            else:
                bc_ant[i,j,:] = np.nan
                pers_a[:] = np.nan
                cvs_a[i,j,:] = np.nan
            
            pers_a = xr.DataArray(pers_a, dims=['days'], coords={'days': range(len(ants1))})
            cval = xr.DataArray(cvs_a[i,j,:], dims=['days'], coords={'days': range(len(ants1))}) 
            a_new_dataset = xr.Dataset({'CVals': cval, 'percentiles': pers_a})
            
        zz = zz+1
        t1 = datetime.datetime.now()
        with open("output2.txt", "a") as f:
            print(zz,"DONE time--->", t1-t0 , file=f)
    
    ############ changing xarrays
    bclat = ants['latitude'].values
    bclon = ants['longitude'].values
    a_bcdays = np.arange(0,bc_ant.shape[2])
    bc_ant = xr.DataArray(bc_ant, dims=('latitude','longitude','days',), name='precipitation',
                                coords={'latitude':bclat, 'longitude':bclon, 'days':a_bcdays})
    
    cvalss = xr.DataArray(cvs_a, dims=('latitude','longitude','days',), name='precipitation',
                                coords={'latitude':bclat, 'longitude':bclon, 'days': a_bcdays})
    
    bc_ant = xr.Dataset({'precipitation': bc_ant, 'CVals': cvalss})
    
    
    return bc_ant

################################################ Main code 
modeldata = read_model_data()
msp = read_mswep()

modeldata_bc = do_bc(modeldata, msp)
    
t2 = datetime.datetime.now()
    
filename= 'BC_modeldata_som43_.nc'
modeldata_bc.to_netcdf(filename)
with open("output2.txt", "a") as f:
    print(s,"Done saving modeldata bias corrected", t2-t0 , file=f)
    
            
    
  
    

