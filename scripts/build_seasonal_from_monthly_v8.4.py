#!/usr/bin/env python
# coding: utf-8

# Compute one seasonal 3S-GEOPROF-COMB granule from monthly granules
# Run "python build_seasonal_from_monthly_v8.4.py -h" for usage
# Version 0.8.4
# Leah Bertrand, University of Colorado
# leah.bertrand@colorado.edu

import os
import sys
import argparse
import re
from glob import glob
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs         
import cartopy.feature as cf
from dask.distributed import Client, LocalCluster
from dask import delayed, compute

from cloudsat_util_10 import *

class MyArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

def print_cluster_info(client):
    info=client.scheduler_info()
    n_workers=len(info['workers'])
    n_cores=sum(client.ncores().values())
    n_threads=sum(client.nthreads().values())
    memory=sum([w['memory_limit'] for w in info['workers'].values()])/1073741824 #bytes per Gibibyte
    print(f'cluster has {n_cores} cores with {n_workers} workers\
, {n_threads//n_workers} threads per worker, and {memory/n_workers:.2f} GiB memory/worker')
    
def nudge_by_seasons(time,dt=-1):
    '''shift by dt seasons and round to the start of the month'''
    ts = pd.to_datetime((time+pd.Timedelta(days=dt*90)).data).to_period('M').to_timestamp()
    da = xr.DataArray(data=ts,dims=['time'],coords={'time':[ts]})
    return da
    
def construct_parser():
    '''construct argparser from build_coverandheight_grid_v8.4.py'''
    #parser with args that should always be present
    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument("-p",help="path to radar and lidar granule folders",dest='parent_dir',
                               default='/pl/active/$USER/ftp.cloudsat.cira.colostate.edu')
    shared_parser.add_argument("-r",help="name of folder containing radar granules", default='2B-GEOPROF.P1_R05',dest='radar_folder')
    shared_parser.add_argument("-l",help="name of folder containing lidar granules", default='2B-GEOPROF-LIDAR.P2_R05',dest='lidar_folder')
    shared_parser.add_argument("-k",help="input data availability required for an orbit to be included",
                               choices=['radar','lidar','both'], default='both',dest='keep_granule')
    shared_parser.add_argument("-o",help="use only one kind of the included granules in calculation",
                               choices=['radar','lidar','both'], default='both',dest='include_only')
    shared_parser.add_argument("-m",help="lidar threshold to be considered cloud (0-100)",type=int, choices=range(0,101),dest='lidar_thresh', default=50,metavar='LIDAR_THRESH')
    shared_parser.add_argument("-n",help="radar threshold to be considered cloud (0-40)",type=int, choices=range(0,41),dest='radar_thresh', default=20,metavar='RADAR_THRESH')
    shared_parser.add_argument("-g",help="grid spacing in degrees",type=float, default=2.5, dest='grid_spacing')
    shared_parser.add_argument("-f",help="output filename suffix", default='', dest='fname_suffix')
    shared_parser.add_argument("-d",help="path to resources directory with ancillary data files", dest='resource_dir',default=os.path.expandvars('/projects/$USER/cloudsat_work/resources'))
    shared_parser.add_argument("--na",action="store_true",help="don't mask bins where lidar is likely attenuated",dest='no_attenuate_lidar')
    shared_parser.add_argument("--nb",action="store_true",help="don't build intermediates netCDF4 files, read existing intermediates",dest='no_build_intermediates')
    shared_parser.add_argument("--nd", action="store_true",help="don't delte intermediate netCDF4 files on completion",dest='no_delete_intermediates')
    shared_parser.add_argument("--outdir",help="base path in which to save output .nc and .pngs", default='.')
    shared_parser.add_argument("--nworkers",help="number of workers in Dask LocalCluster", default=32,type=int,dest='num_workers')
    shared_parser.add_argument("--threads-per-worker",help="number of threads per worker in Dask LocalCluster",default=1,type=int,dest='threads_per_worker')

    #top-level parser
    parser        = MyArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                            fromfile_prefix_chars='@',
                                            description='compute one output granule of the global-gridded 3S-GEOPROF-COMB product. Read arguments from a file by specifying "@args.txt".',
                                            epilog='example: ../build_coverandheight_grid_v8.4.py sub --help',
                                            prog='build_coverandheight_grid_v8.4.py')
    subparsers    = parser.add_subparsers(dest='file_mode',help="input granule file storage hierarchy")

    #add subparser for simple hierarchy, inherit shared arguments
    small_parser  = subparsers.add_parser('sub',parents=[shared_parser],epilog='example: ../build_coverandheight_grid_v8.4.py -p path/to/granules -r 2B-GEOPROF -l 2B-GEOPROF-LIDAR -g 10',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                          help="process all granules in folders organized by product")

    #add subparser for complex hierarchy, inherit shared arguments
    full_parser   = subparsers.add_parser('full',parents=[shared_parser],epilog='example: ../build_coverandheight_grid_v8.4.py --year=2010 --month=06 -g 10',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                          help="process a time range of granules from a copy of the CloudSat DPC archive organized by product/year/day")
    full_parser.add_argument("--year",help="YY or YYYY year to process",required=True)
    full_parser.add_argument("--month",help="MM month to process (incompatible with dayfi/dayla)")
    full_parser.add_argument("--dayfi",type=int,help="first Julian day to include (incompatible with month)")
    full_parser.add_argument("--dayla",type=int,help="last Julian day to include (incompatible with month). Note dayfi and dayla should be in the same month, or else cloud cover by type will fail.")
    full_parser.add_argument("--scratchdir",help="where to save intermediate netCDF4 files", default=os.path.expandvars('/scratch/alpine/$USER/cloudsat_intermediates'))
    full_parser.add_argument("--localdir",help="local (fast) temporary storage directory for Dask LocalCluster worker files", default='.')
    return parser
    
def assign_time(ds):
    '''Adds a time dimension to build_coverandheight_grid_v8.4.py monthly output files'''
    #get command line args from attrs
    parser = construct_parser()
    cmd    = ds.attrs['from_command']
    opts   = parser.parse_args(cmd.split()[1:-1])
    #get datetime from --month and --year args
    time = datetime.strptime(f"{int(opts.month):02d} {opts.year}",'%m %Y')
    #assign number of granules as a data var
    ds = ds.assign(num_granules=ds.attrs['num_granules'])
    #assign first and last profiles as data vars
    first_time  = pd.to_datetime(ds.attrs['time_range'][:20])
    last_time = pd.to_datetime(ds.attrs['time_range'][24:-5])
    ds = ds.assign({'first_time':first_time,'last_time':last_time})
    #assign time as a coordinate
    return ds.assign_coords(time=time)

def get_seasonal_dst(parent_dir,folder):
    '''open a folder of runs at monthly frequency and process and resample to season'''

    #open monthly data record
    path = os.path.join(parent_dir,folder,'out')
    files = sorted(glob(path+'/*.nc'))
    ds = xr.open_mfdataset(files,preprocess=assign_time,combine='nested',concat_dim='time',parallel=True)

    #aggregate counts variables
    ds_int  = ds.drop_vars(['cloud_fraction_on_levels','cloud_cover_in_column','first_time','last_time'])
    ds_sea  = ds_int.resample(time='QS-DEC').sum().fillna(0)
    ds_sea  = ds_sea.assign({'first_time':ds['first_time'].resample(time='QS-DEC').min(),
                             'last_time':ds['last_time'].resample(time='QS-DEC').max()})
    
    #put back occurrence variables
    ds_sea = ds_sea.assign(cloud_fraction_on_levels=100*ds_sea.cloud_counts_on_levels/ds_sea.total_counts_on_levels)
    ds_sea = ds_sea.assign(cloud_cover_in_column=100*ds_sea.cloud_counts_in_column/ds_sea.total_counts_in_column)
    
    #put back attributes and convert back types
    for var in ds_sea:
        ds_sea[var].attrs = ds[var].attrs
        ds_sea[var].data  = ds_sea[var].data.astype(ds[var].dtype)
    
    #get the maximum number of granules per season
    #add start of initial timestep
    times  = xr.concat([nudge_by_seasons(ds_sea.time[0],-1),ds_sea.time],dim='time')
    #max granules = duration of season // 100 minutes
    ds_sea = ds_sea.assign(max_num_granules=times.diff('time',label='upper')//pd.Timedelta(minutes=100))

    return ds_sea

def savefig(dsi,kind,fname,outdir):
    '''save plot for total cloud cover'''
    instr  = {'radarlidar':'CloudSat+CALIPSO','radaronly':'CloudSat','lidaronly':'CALIPSO'}
    dx     = dsi.attrs['resolution_lon']
    f_miss = dsi.attrs['num_granules']/dsi.attrs['max_num_granules']
    fig = plt.figure(num=None, figsize=(8, 5), dpi=150, edgecolor='k')
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    doopstate=b'All cases'
    data = dsi.sel(doop=doopstate)['cloud_cover_in_column'].sel(type=b'any')
    im = ax.imshow(data, origin='lower', extent=[-180,180,-90,90], transform=ccrs.PlateCarree(),cmap='viridis',interpolation='none',vmin=0,vmax=1)
    ax.coastlines()
    t0  = dsi.attrs['time_range'][:10]
    t1  = dsi.attrs['time_range'][26:36]
    title = f"{t0} to {t1} {instr[kind]} cloud cover \n doop={doopstate}, {dx}° grid ({dsi.attrs['num_granules']} granules, {f_miss*100:.1f}% missing)"
    ax.set_title(title)
    cb=fig.colorbar(im,fraction=0.023, pad=0.04)
    cb.set_label('cloud cover')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(outdir,fname+'.png'),transparent=False,facecolor='white',bbox_inches="tight")
    plt.close()
    
def global_attrs(dsi,kind,ds_monthly_sample):
        '''get global attributes'''
        
        descs = {'radarlidar': 'merged CloudSat and CALIPSO hydrometeor masks',
                 'radaronly':'CloudSat hydrometeor mask only',
                 'lidaronly':'CALIPSO hydrometeor mask only'}
        
        global_attrs = ds_monthly_sample.attrs

        #time created
        tstr = datetime.today().strftime('%Y-%m-%d %H:%M:%S MST')
        global_attrs['date_created'] = f'{tstr}'

        #created by script
        global_attrs['from_script'] = f'file created by script {os.path.basename(__file__)}'
        global_attrs['from_command'] = " ".join(sys.argv[:])

        #other
        #global_attrs['author'] = f'Leah Bertrand'
        global_attrs['num_granules'] = int(dsi.num_granules)
        global_attrs['max_num_granules'] = int(dsi.max_num_granules)
        global_attrs['keep_type'] = 'calculated only from granules for which both radar and lidar data are available'
        global_attrs['instrument_type'] = f"calculated from {descs[kind]}"

        #date range
        #first day of season
        ti = str(dsi.first_time.dt.strftime('%Y-%m-%d %H:%M:%S').data)+' UTC'
        tf = str(dsi.last_time.dt.strftime('%Y-%m-%d %H:%M:%S').data)+' UTC'
        global_attrs['time_range'] = f'{str(ti)} - {str(tf)}'
        return global_attrs
    
def run_to_disk(dsi,outdir,kind,fname,ds_monthly_sample):
    '''save lazy season dataset to disk'''
    #set attributes
    dsi.attrs = global_attrs(dsi,kind,ds_monthly_sample)
    #drop vars that got saved as attrs
    dsi = dsi.drop_vars(['num_granules','max_num_granules','first_time','last_time'])
    #load into (worker) memory
    dsi = dsi.compute()
    #save plot
    savefig(dsi,kind,fname,outdir)
    #save netcdf
    print(f'\t saving file {fname}.nc')
    dsi.to_netcdf(os.path.join(outdir,fname+'.nc'))
    #wipe
    del dsi

def main():
    ### CLI ###
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@',
                                     description='process a series of seasonal (tri-monthly) 3S-GEOPROF-COMB granules from a set of monthly granules. Read arguments from a file by specifying "@args.txt".'
                                    )
    parser.add_argument("monthly_dir",metavar="MONTHLY_DIR",help="folder containing a monthly product run. Will check PARENT_DIR/MONTHLY_DIR/out/ for .nc files")
    parser.add_argument("--localdir",help="local (fast) temporary storage directory for Dask LocalCluster worker files", default=os.path.expandvars('$SLURM_SCRATCH'))
    parser.add_argument("--nworkers",help="number of workers in Dask LocalCluster", default=16,type=int,dest='num_workers')
    parser.add_argument("--threads-per-worker",help="number of threads per worker in Dask LocalCluster",default=1,type=int,dest='threads_per_worker')
    parser.add_argument("-p",help="where to look for monthly folder and save seasonal output",
                        dest='parent_dir',default=os.path.expandvars('/projects/$USER/cloudsat_work'))
    
    args = parser.parse_args()
    args = vars(args)
    print("running with the following params: ")
    pprint(args)

    print('initializing cluster...',end='')
    ### INITIALIZE CLUSTER ###
    cluster = LocalCluster(n_workers=args['num_workers'],
                           local_directory=args['localdir'],
                           threads_per_worker=args['threads_per_worker'])
    client = Client(cluster)
    print('done')
    print_cluster_info(client)
    print(f'View the cluster at {client.dashboard_link}')

    flavors = {'radarlidar':'','radaronly':'-RO','lidaronly':'-LO'}
    folder = args['monthly_dir']
    print(f'processing seasons from {folder}')
    #get seasonal timeseries dataset
    ds_sea = get_seasonal_dst(args['parent_dir'],folder)

    #make a new folder to save the data in
    if 'olidar' in folder:
        kind = 'lidaronly'
    elif 'oradar' in folder:
        kind = 'radaronly'
    else:
        kind = 'radarlidar'
    dx = float(folder.split('-')[-1][1:])
    newfolder = re.sub('months','seasons',folder)
    outdir = os.path.join(args['parent_dir'],newfolder)
    outdir = os.path.join(outdir,'out')
    os.makedirs(outdir, exist_ok=True)
    print(f'made directory {outdir}')
    
    #get a sample monthly file for global metadata
    sample_fp = os.path.join(args['parent_dir'],args['monthly_dir'])
    sample_fp = os.path.join(sample_fp,'out')
    sample_fp = sorted(glob(os.path.join(sample_fp,'*.nc')))[0]
    ds_monthly_sample = xr.open_dataset(sample_fp)
    
    #put back bounds variables
    for name in ds_monthly_sample:
        if 'bounds' in name:
            ds_sea[name] = ds_monthly_sample[name]


    #loop over timesteps in run and save NC4s
    print('building calculation...')
    res = [ ]
    for time in ds_sea.time:
        dsi = ds_sea.sel(time=time)

        #missing data fraction
        f_miss = float(1-dsi.num_granules/dsi.max_num_granules)
        if f_miss > 0.5:
            print(f"\t skipping {dsi.time.dt.strftime('%Y-%m-%d').values} -- {100*f_miss:.0f}% data missing (max 50%)")
            continue

        #get filename
        fname_suffix = 'v8.4'
        timestr = f'{str(dsi.time.dt.year.values)}-{str(dsi.time.dt.season.values)}'
        fname = f"{timestr}_CSCAL_3S-GEOPROF-COMB{flavors[kind]}_{dx}x{dx}_{fname_suffix}"

        if os.path.exists(os.path.join(outdir,fname+'.nc')):
            print(f"\t skipping {dsi.time.dt.strftime('%Y-%m-%d').values} -- file already present")
        else:
            print(f"\t adding {dsi.time.dt.strftime('%Y-%m-%d').values}")
            
        #compute
        res.append(delayed(run_to_disk)(dsi,outdir,kind,fname,ds_monthly_sample))

    print('computing...')
    compute(*res)
    print('success')
    sys.exit(0)
        
if __name__ == "__main__":
    main()
