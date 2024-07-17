#!/usr/bin/env python
# coding: utf-8

# Compute one 3S-GEOPROF-COMB granule
# Run "python build_coverandheight_grid_v8.4.py -h" for usage
# Version 0.8.4
# Leah Bertrand, University of Colorado
# leah.bertrand@colorado.edu

# dependencies: dask, xarray, numpy, pandas, scipy, cartopy, matplotlib, pyhdf, bottleneck, netCDF4

import sys
import argparse
import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

from dask.distributed import Client, LocalCluster
import dask.bag as db

from cloudsat_util_10 import *
# used methods: last_day_of_month, get_granule_paths, listdir_fullpaths, intersection,
#               read_convert_merge_granule, print_dataquality, ds_ll2gbnum, proc_and_mask,
#               get_total_counts_nosum, get_cloud_counts_nosum, get_total_counts_nosum_lidaronly,
#               get_cloud_counts_nosum_lidaronly, get_total_counts_nosum_radaronly,
#               get_cloud_counts_nosum_radaronly, broadcast_cloud_level_thresholds_to_binary_masks,
#               apply_all_cloud_defs_variable_heights, do_gridding

def print_cluster_info(client):
    info=client.scheduler_info()
    n_workers=len(info['workers'])
    n_cores=sum(client.ncores().values())
    n_threads=sum(client.nthreads().values())
    memory=sum([w['memory_limit'] for w in info['workers'].values()])/1073741824 #bytes per Gibibyte
    print(f'cluster has {n_cores} cores with {n_workers} workers\
, {n_threads//n_workers} threads per worker, and {memory/n_workers:.2f} GiB memory/worker')

def main():
    ### OPTIONS ###
    standard_defs = ('any', 'thick', 'high',
             'middle','low', 'uniquehigh',
             'uniquemiddle', 'uniquelow')

    rename = {'height':{'total_counts':'total_counts_on_levels',
                        'cloud_counts':'cloud_counts_on_levels',
                        'cloud_cover':'cloud_fraction_on_levels',
                        'attenuated_lidar_counts':'attenuated_lidar_counts_on_levels',
                        'radar_surface_clutter_counts':'radar_surface_clutter_counts_on_levels'},
              'cover':{'total_counts':'total_counts_in_column',
                       'cloud_counts':'cloud_counts_in_column',
                       'cloud_cover':'cloud_cover_in_column',
                       'attenuated_lidar_counts':'attenuated_lidar_counts_in_column',
                       'radar_surface_clutter_counts':'radar_surface_clutter_counts_in_column'}}

    dtypes = { }
    for kind in ('height','cover'):
        dtypes[rename[kind]['cloud_counts']] = int
        dtypes[rename[kind]['total_counts']] = int
        dtypes[rename[kind]['cloud_cover']] = float

    #results of survey of all granules from 8 July 2006 onward (see notebook 9-2-5)
    height_midpoints = np.array([17987. , 17747. , 17507. , 17267. , 17027. , 16788. , 16548. ,
       16308. , 16068. , 15828. , 15588. , 15349. , 15109. , 14869. ,
       14629. , 14389. , 14149. , 13910. , 13670. , 13430. , 13190. ,
       12950. , 12710.5, 12471. , 12231. , 11991. , 11751. , 11511. ,
       11271.5, 11032. , 10792. , 10552. , 10312. , 10072. ,  9833. ,
        9593. ,  9353. ,  9113. ,  8873. ,  8633. ,  8394. ,  8154. ,
        7914. ,  7674. ,  7434. ,  7194. ,  6955. ,  6715. ,  6475. ,
        6235. ,  5995. ,  5755. ,  5516. ,  5276. ,  5036. ,  4796. ,
        4556. ,  4316. ,  4077. ,  3837. ,  3597. ,  3357. ,  3117. ,
        2877.5,  2638. ,  2398. ,  2158. ,  1918. ,  1678. ,  1438.5,
        1199. ,   959. ,   719. ,   479. ,   239. ,     0. ,  -240. ,
        -480. ])

    height_mins = np.array([17867., 17627., 17387., 17147., 16907., 16668., 16428., 16188.,
       15948., 15708., 15468., 15229., 14989., 14749., 14509., 14269.,
       14029., 13790., 13550., 13310., 13070., 12830., 12591., 12351.,
       12111., 11871., 11631., 11391., 11152., 10912., 10672., 10432.,
       10192.,  9952.,  9713.,  9473.,  9233.,  8993.,  8753.,  8513.,
        8274.,  8034.,  7794.,  7554.,  7314.,  7074.,  6835.,  6595.,
        6355.,  6115.,  5875.,  5635.,  5396.,  5156.,  4916.,  4676.,
        4436.,  4196.,  3957.,  3717.,  3477.,  3237.,  2997.,  2758.,
        2518.,  2278.,  2038.,  1798.,  1558.,  1319.,  1079.,   839.,
         599.,   359.,   119.,  -120.,  -360.,  -600.])

    height_maxes = np.array([18107., 17867., 17627., 17387., 17147., 16908., 16668., 16428.,
       16188., 15948., 15708., 15469., 15229., 14989., 14749., 14509.,
       14269., 14030., 13790., 13550., 13310., 13070., 12830., 12591.,
       12351., 12111., 11871., 11631., 11391., 11152., 10912., 10672.,
       10432., 10192.,  9953.,  9713.,  9473.,  9233.,  8993.,  8753.,
        8514.,  8274.,  8034.,  7794.,  7554.,  7314.,  7075.,  6835.,
        6595.,  6355.,  6115.,  5875.,  5636.,  5396.,  5156.,  4916.,
        4676.,  4436.,  4197.,  3957.,  3717.,  3477.,  3237.,  2997.,
        2758.,  2518.,  2278.,  2038.,  1798.,  1558.,  1319.,  1079.,
         839.,   599.,   359.,   120.,  -120.,  -360.])

    ### COMMAND-LINE INTERFACE ###
    class MyArgumentParser(argparse.ArgumentParser):
        def convert_arg_line_to_args(self, arg_line):
            return arg_line.split()
    def dir_path(path):
        if os.path.isdir(path):
            return Path(path)
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
    #parser with args that should always be present
    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument("-p",help="path to radar and lidar granule folders",dest='parent_dir',
                               default='.',type=dir_path) #ftp.cloudsat.cira.colostate.edu
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
    shared_parser.add_argument("-d",help="path to resources directory with ancillary data files", dest='resource_dir',default='resources',type=dir_path)
    shared_parser.add_argument("--na",action="store_true",help="don't mask bins where lidar is likely attenuated (untested)",dest='no_attenuate_lidar')
    shared_parser.add_argument("--nb",action="store_true",help="don't build intermediates netCDF4 files, read existing intermediates",dest='no_build_intermediates')
    shared_parser.add_argument("--nd", action="store_true",help="don't delete intermediate netCDF4 files on completion",dest='no_delete_intermediates')
    shared_parser.add_argument("--outdir",help="base path in which to save output .nc and .pngs", default='.',type=dir_path)
    shared_parser.add_argument("--nworkers",help="number of workers in Dask LocalCluster", default=32,type=int,dest='num_workers')
    shared_parser.add_argument("--threads-per-worker",help="number of threads per worker in Dask LocalCluster. >1 breaks as of netCDF4=1.7.1",default=1,type=int,dest='threads_per_worker')

    #top-level parser
    parser        = MyArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                            fromfile_prefix_chars='@',
                                            description='compute one output granule of the global-gridded 3S-GEOPROF-COMB product. Read arguments from a file by specifying "@args.txt".',
                                            epilog='example: %(prog)s sub --help')
    subparsers    = parser.add_subparsers(dest='file_mode',help="input granule file storage hierarchy")

    #add subparser for simple hierarchy, inherit shared arguments
    small_parser  = subparsers.add_parser('sub',parents=[shared_parser],epilog='example: %(prog)s -p path/to/granules -r 2B-GEOPROF -l 2B-GEOPROF-LIDAR -g 10',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                          help="process all granules in folders organized by product")

    #add subparser for complex hierarchy, inherit shared arguments
    full_parser   = subparsers.add_parser('full',parents=[shared_parser],epilog='example: %(prog)s --year=2010 --month=06 -g 10',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                          help="process a time range of granules from a copy of the CloudSat DPC archive organized by product/year/day")
    full_parser.add_argument("--year",help="YY or YYYY year to process",required=True)
    full_parser.add_argument("--month",help="MM month to process (incompatible with dayfi/dayla)")
    full_parser.add_argument("--dayfi",type=int,help="first Julian day to include (incompatible with month)")
    full_parser.add_argument("--dayla",type=int,help="last Julian day to include (incompatible with month). Note dayfi and dayla should be in the same month, or else cloud cover by type will fail.")
    full_parser.add_argument("--scratchdir",help="where to save intermediate netCDF4 files", default='.',type=dir_path)
    full_parser.add_argument("--localdir",help="local (fast) temporary storage directory for Dask LocalCluster worker files", default='.',type=dir_path)

    ### SET OPTIONS FROM CLI ARGS ###
    args = parser.parse_args()
    args = vars(args)

    print(f"xarray version: {xr.__version__}")
    print("running with the following params: ")
    pprint(args)

    print(f"local directory: {args['localdir']}")
    print('initializing cluster...',end='')

    ### INITIALIZE CLUSTER ###
    cluster = LocalCluster(n_workers=args['num_workers'], 
            threads_per_worker=args['threads_per_worker'],local_directory=args['localdir'])
    #    cluster = LocalCluster() #let dask infer defaults
    client = Client(cluster)
    print('done')
    print_cluster_info(client)
    print(f'View the cluster at {client.dashboard_link}')

    ### ENUMERATE INPUT FILE PATHS AND ###
    ### GET INTERMEDIATE FOLDER NAME ###
    print('processing options and enumerating filepaths...',end='')
    if args['file_mode'] == 'full':
        #check for proper input conditions
        if args['month'] and (args['dayfi'] or args['dayla']) in args:
            print('ERROR: cannot specify both a month and a day! Quitting...')
            sys.exit(1)
        if (args['dayfi'] or args['dayla']) and ((args['dayfi'] is None) or (args['dayla'] is None)):
            print('ERROR: both --dayfi and --dayla must be specified if not using --month! Quitting...')
            sys.exit(1)
        if not args['year']:
            print('ERROR: --year must be specified when using summit mode (-s)! Quitting...')
            sys.exit(1)

        #process time args
        if len(args['year'])==1:
            args['year']='200'+args['year'] #add century+decade
        elif len(args['year'])==2:
            args['year']='20'+args['year'] #add century
        if args['dayfi'] and args['dayla']:
            args['dayfi'],args['dayla']=int(args['dayfi']),int(args['dayla'])
            t0=datetime.strptime(f"{args['year']}{args['dayfi']}",'%Y%j')
            t1=datetime.strptime(f"{args['year']}{args['dayfi']}",'%Y%j')
            assert t0.month==t1.month, 'dayfi and dayla must be in the same month for 440/680 mb heights'
            t=datetime.strptime(f"{args['year']}{t0.month:02d}",'%Y%m') #first day of month
        if args['month']:
            #get first/last day of year for the month
            t=datetime.strptime(f"{args['year']}{args['month']}",'%Y%m')
            args['dayfi']=int(t.strftime('%j'))
            args['dayla']=int(last_day_of_month(t).strftime('%j'))

        #get correct folders to process + get items in folders
        radar_filepaths = get_granule_paths(args['dayfi'],args['dayla'],os.path.join(args['parent_dir'],args['radar_folder'],args['year']))
        lidar_filepaths = get_granule_paths(args['dayfi'],args['dayla'],os.path.join(args['parent_dir'],args['lidar_folder'],args['year']))

        #folder name for intermediates
        t0=datetime.strptime(f"{args['dayfi']:03d}{args['year']}",'%j%Y')
        t1=datetime.strptime(f"{args['dayla']:03d}{args['year']}",'%j%Y')
        #if dates equal whole month use e.g. 2013jan,
        #otherwise includes months+dates
        if (t0.day==1) and (t1.day==last_day_of_month(t0).day):
            timestr=t0.strftime('%Y-%m').lower()
        else:
            timestr=(t0.strftime('%Y-%m-%d')+'_'+t1.strftime('%m-%d')).lower()
        new_combined_folder = f"nc_{timestr}_combined"
        new_combined_path = os.path.join(args['scratchdir'],new_combined_folder)

    elif args['file_mode'] == 'sub':
        #items in folder
        radar_filepaths = tuple(listdir_fullpaths(os.path.join(args['parent_dir'],args['radar_folder'])))
        lidar_filepaths = tuple(listdir_fullpaths(os.path.join(args['parent_dir'],args['lidar_folder'])))

        #folder name for intermediates
        new_combined_folder = 'nc_'+args['radar_folder']+'_combined'
        new_combined_path = os.path.join(args['parent_dir'],new_combined_folder)

    print('done')
    ### MAKE FOLDER FOR SAVING INTERMEDIATES ###
    try:
        os.makedirs(new_combined_path, exist_ok=False)
        print(f'made directory {new_combined_path} to store intermediate files')
    except FileExistsError:
        print(f'intermediates directory {new_combined_path} already present!')
        if not args['no_build_intermediates']:
            print(f"But no_build_intermediates == {args['no_build_intermediates']}... Quitting")
            sys.exit(1)

    ### BUILD INTERMEDIATE NETCDF4 FILES FROM HDF4-EOS GRANULES ###
    keep_vars=['CPR_Cloud_mask','SurfaceHeightBin','DEM_elevation','Profile_time',
               'Data_quality','CloudFraction','Latitude','Longitude','Data_status'] 
    #only keep granules for which there is both lidar and radar data
    #always compute since it's needed for the quicklook printout()
    thin_radar_filepaths, thin_lidar_filepaths = intersection(radar_filepaths,lidar_filepaths,keep_granule=args['keep_granule'])

    if not args['no_build_intermediates']:
        print(f'writing {len(thin_radar_filepaths)} intermediate netcdf4 files at {new_combined_path}...')
        #use dask.bag to parallelize the iteration
        rlb = db.from_sequence(zip(thin_radar_filepaths,thin_lidar_filepaths))
        _   = rlb.map(read_convert_merge_granule,new_combined_path,keep_only=keep_vars,resources_dir=args['resource_dir']).compute()
        print('write complete. Restarting dask client...')
    #        client.restart()

    print('opening intermediates as dataset...',end='')
    ### LAZILY LOAD NC4S FROM DISK ###
    tcv = xr.open_mfdataset(new_combined_path+'/*.nc',concat_dim='start_time',combine='nested',
                             decode_cf=True, mask_and_scale=False, decode_times=True, 
                             decode_timedelta=True, use_cftime=False,
                             concat_characters=False, decode_coords=True, parallel=True)
    print('done')

    ### CHECK IF ANY PROFILES HAVE LONGITUDE==+180 ###
    lon      = tcv['Longitude'].compute()
    lon_mask = (lon==180.0)
    if lon_mask.any():
    # subtract a tiny value (equal to a max of 0.1 um) to fix the gridbox assignment
        print(f'correcting {np.sum(lon_mask.values)} profiles with Longitude==180')
        tcv = tcv.assign(Longitude=tcv.Longitude-(1e-12)*lon_mask.astype(int))

    ### ADD/HANDLE QC VARIABLES ###
    #add in sampling fields
    #days in period (input DayOfYear)
    tcv = tcv.assign(dofy=(tcv.start_time+tcv.Profile_time).dt.dayofyear)
    #unique granules (input start_time)
    tcv = tcv.assign(overpass=tcv.start_time)
    #local time bins
    shape, dims = tcv.Longitude.data.shape,tcv.Longitude.dims
    utc_times_flat = (tcv.start_time+tcv.Profile_time).compute().data.flatten()
    lon_flat = lon.data.flatten()
    tz_offsets_flat = pd.to_timedelta((lon_flat+15/2)//15,unit='h')
    local_times_flat = pd.to_datetime(utc_times_flat + tz_offsets_flat)
    local_hour_flat = np.asarray(local_times_flat.hour) #can't reshape pd.DateTimeIndex
    local_hour = local_hour_flat.reshape(shape)
    # next step: bool for 0<=t<6, 6<=t<12, 12<=t<18, 18<=t<24
    lt_funcs  = {'localhour22':lambda x: np.logical_or(x>=22,x<4),
                 'localhour04':lambda x: np.logical_and(x>=4,x<10),
                 'localhour10':lambda x: np.logical_and(x>=10,x<16),
                 'localhour16':lambda x: np.logical_and(x>=16,x<22)}
    new_vars = {name: (dims,func(local_hour)) for name,func in lt_funcs.items()}
    tcv = tcv.assign(new_vars)
    del shape,dims,utc_times_flat,lon_flat,tz_offsets_flat
    del local_times_flat,local_hour_flat,local_hour,new_vars

    #handle radar data quality
    print_dataquality(tcv) #print out diagnostic info
    tcv = tcv.where(tcv.Data_quality==0) #if any DQ flag is set, set all vars in the profile to NaN
    #note -- this might have to go before or after the sampling vars

    #restrict valid profiles to time range
    print()
    print('removing profiles not in month...')
    dstime=tcv.start_time+tcv.Profile_time
    time_mask = (dstime>=pd.to_datetime(t0)) & (dstime<pd.to_datetime(t1)+pd.Timedelta(1,unit='days'))
    tcv = tcv.where(time_mask,drop=False)

    #print out info on what was masked
    tfmt = lambda da: da.start_time.dt.strftime("%Y-%m-%d %H:%M:%S").values
    nmski = int((~time_mask.isel(start_time=0)).sum(dim='profile') - np.isnan(dstime.isel(start_time=0)).sum(dim='profile'))
    nmskf = int((~time_mask.isel(start_time=-1)).sum(dim='profile') - np.isnan(dstime.isel(start_time=-1)).sum(dim='profile'))
    print(f"masked {nmski} profs (~{nmski*0.16/60:.1f} min) in first granule with start_time {tfmt(time_mask.isel(start_time=0))}")
    print(f"masked {nmskf} profs (~{nmskf*0.16/60:.1f} min) in last granule with start_time {tfmt(time_mask.isel(start_time=-1))}")

    ### ADD ATTRIBUTES IF DUMMY GRANULES ARE PRESENT ###
    # v7.2.1: this fix should really be in read_convert_merge_granule's dummy features,
    # but I'm putting it here since I don't want to rewrite all the intermediates.
    if args['keep_granule'] == 'radar': #if dummy lidar granules are present,
        #then add back the correct attributes (since they may have been stripped)
        tcv['CloudFraction'].attrs = {'_FillValue':-9.0,
                                      'factor':1.0,
                                      'offset':0.0,
                                      'long_name':'Cloud Fraction',
                                      'valid_range':[0,100],
                                      'missing':-9,
                                      'missop':'=='}
    elif args['keep_granule'] == 'lidar': #if dummy radar granules are present,
        #then add back the correct attributes (since they may have been stripped)
        tcv['CPR_Cloud_mask'].attrs = {'_FillValue':-9.0,
                                      'factor':1.0,
                                      'offset':0.0,
                                      'long_name':'Cloud Fraction',
                                      'valid_range':[0,100],
                                      'missing':-9,
                                      'missop':'=='}

    ### READ IN 440 AND 680 MB HEIGHTS FOR CLOUD COVER CALCULATION ###
    pressure_heights = xr.open_dataset(args['resource_dir']/'height_bin_at_pressure_level.nc')
    pressure_heights = pressure_heights.height_index_at_pressure
    lat = tcv['Latitude'].compute()
    lat_quant = 2.5*np.round(lat/2.5)
    cloud_bounds_levels = pressure_heights.sel(time=t,lat=lat_quant.fillna(90))
    cloud_bounds = {'lo':cloud_bounds_levels.sel(level=680).drop_vars(['time','level']),
                    'hi':cloud_bounds_levels.sel(level=440).drop_vars(['time','level'])}

    ### ASSIGN GRIDBOX NUMBER TO EACH START_TIME,PROFILE COORD ###
    tcv_gb = ds_ll2gbnum(tcv,dx=args['grid_spacing'])
    tcv_gb = tcv_gb.drop_vars(['Latitude','Longitude'])

    ### DECODE/PROCESS RADAR AND LIDAR TO BINARY MASKS ###
    # make a template dataset containing the variables proc_and_mask will return
    template = tcv_gb.rename({'CPR_Cloud_mask':'cpr_binary','CloudFraction':'vfm_binary'})
    template = template.assign(radar_surface_clutter_counts=template.cpr_binary.astype(bool))
    if not args['no_attenuate_lidar']: 
        template = template.assign(attenuated_lidar_counts=template.vfm_binary.astype(bool))
    tcm = tcv_gb.map_blocks(proc_and_mask,template=template,kwargs={'lidar_thresh':args['lidar_thresh'],'radar_thresh':args['radar_thresh']})
    tcm = tcm.drop_vars(('DEM_elevation','Data_quality','SurfaceHeightBin'))
    # variables needed for final output
    all_mask      = ((tcv.Data_quality==0) & time_mask).compute()
    valid_granule = all_mask.any('profile')
    dstime_valid  = dstime[valid_granule]
    grab_pos = lambda ds,pos: ds.isel(start_time=pos).dropna('profile').isel(profile=pos)
    start_time_initial = grab_pos(dstime_valid.where(time_mask),0).compute()  #time of first profile
    start_time_final   = grab_pos(dstime_valid.where(time_mask),-1).compute() #time of last profile
    n_granules = tcm.sizes['start_time']

    ### MERGE MASKS TO TOTAL_COUNTS, CLOUD_COUNTS ###
    if args['include_only'] == 'both':
        tcm = get_total_counts_nosum(tcm)
        tcm = get_cloud_counts_nosum(tcm)
    elif args['include_only'] == 'lidar':
        tcm = get_total_counts_nosum_lidaronly(tcm)
        tcm = get_cloud_counts_nosum_lidaronly(tcm)
    elif args['include_only'] == 'radar':
        tcm = get_total_counts_nosum_radaronly(tcm)
        tcm = get_cloud_counts_nosum_radaronly(tcm)
    tcc = tcm.drop_vars(['vfm_binary','cpr_binary'],errors='ignore')
    tcc = tcc.persist()

    ### PROCESS PROFILES TO BE VERTICALLY-RESOLVED AND/OR COLUMN-INTEGRATED ###
    #remove keys that are not 'cloud_counts' or 'total_counts' from the inner dicts using dict comprehension (py>2.7)
    #do it since other vars like 'cloud_cover' are not present yet and would cause xr.rename to fail
    rename_counts = {k:{i:d[i] for i in d if i in ('cloud_counts',
                                                   'total_counts',
                                                   'attenuated_lidar_counts',
                                                   'radar_surface_clutter_counts')} for k,d in rename.items()}

    #exclude non-counts variables from this list (e.g. doesn't end in "_counts" (new v7.2)
    non_counts_names = list(filter(lambda x: x.split('_')[-1]!='counts',list(tcc.data_vars)))
    s_tcc = tcc.drop_vars(non_counts_names) #s for counts which get summed
    u_tcc = tcc[non_counts_names] #u for non-counts which get the number of unique values

    print("computing output fields...",end='')

    #compute+rename both and combine
    #feed one into the other for speedup
    cloud_bound_masks = broadcast_cloud_level_thresholds_to_binary_masks(cloud_bounds,s_tcc.cloud_counts)
    #this is where radar_surface_clutter_counts_in_column would get computed but doesn't
    stcm  = apply_all_cloud_defs_variable_heights(s_tcc,cloud_bound_masks)
    hstcm = s_tcc.rename(rename_counts['height']).sel(bin=slice(29,107))
    stcm  = stcm.rename({var:f'{var}_in_column' for var in stcm.data_vars}) #add _in_column to every var
    stcm  = xr.merge([hstcm,stcm])


    ### COMPUTE BEFORE GROUPY TO AVOID NUMBER OF TASKS OVERFLOW ###
    print('done. Computing mask/process/gridbox_num assign...',end='')
    stcm = xr.merge([stcm,u_tcc]) #put back non-counts variables
    stcm = stcm.compute() # ~7 GB
    #del tcv,tcm,tcc,s_tcc,u_tcc #uncomment for any overflow issues
    print('done')

    ### CREATE TOP-LEVEL GROUPS BY DOOP-FLAG ###
    # NB: groupby on full data spikes swap usage to ~45 GB during 'All cases'
    # NB: rechunking before groupby slows down operation 6x
    stacked_stcm = stcm.stack(stacked_start_time_profile=('start_time','profile'))
    doop_stcm = stacked_stcm.sel(stacked_start_time_profile=stacked_stcm.doop_flag==True)
    d = {b'DO-OP observable':doop_stcm,
         b'All cases':stacked_stcm}
    d_new = { }
    for key, dstcm in d.items():
        print(f'doing gridding for doop_flag=={key}')
        d_new[key] = do_gridding(dstcm,args['grid_spacing'],height_midpoints)

    ### JOIN RUNS INTO SINGLE DS WITH NEW DIM DOOP ###
    gridded = xr.concat([ds.assign_coords(doop=key) for key,ds in d_new.items()],dim='doop')

    ### PUT CLOUD COVER BY TYPE ALONG NEW DIMENSION ###
    # apply_all_cloud_defs returns 8 total_counts_in_column and cloud_counts_in_column by cover type.
    # reorganize these to be in two data variables but with a new dimension 'type'

    # encode coordinates as bytes for consistency with doop and to pass CF-checker
    da_cc = xr.concat([gridded['cloud_counts_'+alg+'_in_column'].assign_coords({'type':alg.encode('utf-8')}) for alg in standard_defs],dim='type')
    gridded = gridded.drop_vars(['cloud_counts_'+alg+'_in_column' for alg in standard_defs])
    gridded = gridded.assign(cloud_counts_in_column=da_cc)

    da_tc = xr.concat([gridded['total_counts_'+alg+'_in_column'].assign_coords({'type':alg.encode('utf-8')}) for alg in standard_defs],dim='type')
    gridded = gridded.drop_vars(['total_counts_'+alg+'_in_column' for alg in standard_defs])
    gridded = gridded.assign(total_counts_in_column=da_tc)

    ### CALCULATE CLOUD COVER ###
    for kind in ('height','cover'):
        cloud_counts = gridded[rename[kind]['cloud_counts']]
        total_counts = gridded[rename[kind]['total_counts']]
        gridded = gridded.assign({rename[kind]['cloud_cover']:cloud_counts/total_counts})

    # output is done now, so delete other variables
    del total_counts, cloud_counts, d_new, doop_stcm, stacked_stcm

    ### CALCULATE FILENAME ###
    try:
        timestr #timestr is undefined outside file_mode == 'full'
    except NameError:
        timestr = str(start_time_initial.dt.strftime('%Y-%m').data)

    #get flavor from args (e.g. -RO, -LO, -RO2)
    flavor = ''
    if args['keep_granule'] == 'both':
        if args['include_only'] == 'radar':
            flavor = '-RO'
        elif args['include_only'] == 'lidar':
            flavor = '-LO'
    elif args['keep_granule'] == 'radar':
        if args['include_only'] == 'radar':
            flavor = '-RO2'
    fname = f"{timestr}_CSCAL_3S-GEOPROF-COMB{flavor}_{args['grid_spacing']}x{args['grid_spacing']}_{args['fname_suffix']}"

    ### SAVE PLOT FOR TOTAL CLOUDFRACTION ###
    fig = plt.figure(num=None, figsize=(10, 5), dpi=150, edgecolor='k')
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    doopstate=b'All cases'
    data = gridded.sel(doop=doopstate)[rename['cover']['cloud_cover']].sel(type=b'any')
    im = ax.imshow(data, origin='lower', extent=[-180,180,-90,90], transform=ccrs.PlateCarree(),cmap='viridis',interpolation='none',vmin=0,vmax=1)
    ax.coastlines()

    t0  = str(start_time_initial.dt.strftime('%Y-%m-%d').data)
    t1  = str(start_time_final.start_time.dt.strftime('%Y-%m-%d').data)
    title = f"{t0} to {t1} Cloudsat+Calipso Cloud Cover, doop={doopstate}, {args['grid_spacing']}° grid ({n_granules} granules)"
    ax.set_title(title)
    cb=fig.colorbar(im,fraction=0.023, pad=0.04)
    cb.set_label('cloud cover')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(args['outdir'],fname+'.png'),transparent=False,facecolor='white',bbox_inches="tight")

    ### SET GLOBAL ATTRIBUTES ###
    global_attrs = { }

    #time created
    tstr = datetime.today().strftime('%Y-%m-%d %H:%M:%S MST')
    global_attrs['date_created'] = f'{tstr}'

    global_attrs['citation'] = 'Please cite Bertrand, L., Kay, J. E., Haynes, J., and de Boer, G.: A global gridded dataset for cloud vertical structure from combined CloudSat and CALIPSO observations, Earth Syst. Sci. Data, 16, 1301–1316, https://doi.org/10.5194/essd-16-1301-2024, 2024.'

    #created by script
    global_attrs['from_script'] = f'file created by script {os.path.basename(__file__)}'
    global_attrs['from_command'] = " ".join(sys.argv[:])

    #other
    global_attrs['num_granules'] = n_granules
    global_attrs['keep_type'] = 'calculated only from granules for which both radar and lidar data are available'
    global_attrs['resolution_lon'] = args['grid_spacing']
    global_attrs['resolution_lat'] = args['grid_spacing']

    #date range
    ti = str(start_time_initial.dt.strftime('%Y-%m-%d %H:%M:%S').data)+' UTC'
    tf = str(start_time_final.dt.strftime('%Y-%m-%d %H:%M:%S').data)+' UTC'
    global_attrs['time_range'] = f'{ti} - {tf}'

    #apply global attrs
    gridded.attrs = global_attrs

    ### SET COORDINATE AND DATA VAR ATTRIBUTES ###
    #specify all coordinate and data var attributes here in the form attribute:varname:val
    descriptions = {rename['height']['total_counts']: 'total number of valid radar-lidar samples (clear sky or cloudy) in lat/lon/height bin',
                    rename['height']['cloud_counts']: 'number of cloudy radar-lidar samples in lat/lon/height bin',
                    rename['height']['cloud_cover']: 'relative frequency of cloud in lat/lon/height bin',
                    rename['height']['attenuated_lidar_counts']: 'number of estimated attenuated lidar counts in lat/lon/height bin',
                    rename['height']['radar_surface_clutter_counts']: 'number of times surface clutter prevented radar measurement in lat/lon/height bin',
                    rename['cover']['total_counts']: 'total number of valid radar-lidar profiles (clear sky or cloudy) in lat/lon grid cell',
                    rename['cover']['cloud_counts']: 'number of cloudy profiles in lat/lon grid cell',
                    rename['cover']['cloud_cover']: 'relative frequency of cloudy columns in lat/lon grid cell',
                    rename['cover']['attenuated_lidar_counts']: 'profiles which attenuate at some level',
                    'type':'''Different cloudy criteria applied to profiles. High, middle, low are satisfied if any cloud is present in a pressure range. High is <440 hPa, low is >680 hPa, and middle is 440 to 680 hPa.
    Unique variants are satisfied if all cloud in the profile is within the presure range. Thick is satisfied if a cloud layer is >=4.8 km thick.
    Any is satisfied if any cloud is in the profile. Total counts report the number of profiles with valid observations in the pressure range.''',
                    'n_overpasses': 'number of unique visits to/overpasses of the grid cell during the aggregating period',
                    'n_days': 'number of unique days in month grid cell was observed',
                    'localhour_counts': 'number of profiles in 6-hour local time bins'}

    units = {rename['height']['total_counts']: 'count', #or units "count" or blank ""
             rename['height']['cloud_counts']: 'count',
             rename['height']['cloud_cover']: '%',
             rename['height']['attenuated_lidar_counts']: 'count',
             rename['height']['radar_surface_clutter_counts']: 'count',
             rename['cover']['total_counts']: 'count',
             rename['cover']['cloud_counts']: 'count',
             rename['cover']['cloud_cover']: '%',
             rename['cover']['attenuated_lidar_counts']: 'count',
             'lat': 'degrees_north',
             'lon': 'degrees_east',
             'height': 'm',
             'height_bounds': 'm',
             'n_overpasses':'count',
             'n_days':'count',
             'localhour_counts': 'count'}

    long_name = {rename['height']['total_counts']: 'Number of observations at altitude',
                 rename['height']['cloud_counts']: 'Number of observations of cloud at altitude',
                 rename['height']['cloud_cover']: 'Cloud counts at level divided by total counts at level x 100',
                 rename['height']['attenuated_lidar_counts']: 'Estimated count of attenuated lidar samples at level',
                 rename['height']['radar_surface_clutter_counts']: 'Count of no radar observation due to surface clutter at level',
                 rename['cover']['total_counts']: 'Number of observations of whole column',
                 rename['cover']['cloud_counts']: 'Number of times cloud of (type) present in column',
                 rename['cover']['cloud_cover']: 'Cloud counts in column of type divided by total counts of type in column x 100',
                 rename['cover']['attenuated_lidar_counts']: 'Count of lidar attenuation estimated anywhere in column',
                 'lat': 'latitude midpoint of grid cell',
                 'lon': 'longitude midpoint of grid cell',
                 'height': 'altitude midpoint meters above mean sea level',
                 'height_bounds': 'maximum and minimum height of bin range (above mean sea level)',
                 'doop':'Daylight-only operations status',
                 'type':'cloud cover definition',
                 'n_overpasses':'number of overpasses of grid cell',
                 'n_days':'number of unique days grid cell observed',
                 'localhour': 'local time bin midpoints',
                 'localhour_bounds': 'local time bin ranges',
                 'localhour_counts': 'number of profiles in local time bins'}

    dx = {'lat': args['grid_spacing'], 'lon': args['grid_spacing']}

    #put all attrs together with top level as attr name
    attrs_by_attrname = {'dx':dx,'description':descriptions,'units':units,'long_name':long_name} 

    #rename non-counts variables
    gridded = gridded.rename({'overpass':'n_overpasses','dofy':'n_days'})

    #save bounds with a "bounds" coordinate
    height_bounds = np.zeros([gridded.sizes['height'],2])
    height_bounds[:,0] = height_mins
    height_bounds[:,1] = height_maxes
    height_bounds = xr.DataArray(data=height_bounds,dims=['height','bound'])
    gridded = gridded.assign({'height_bounds':height_bounds})

    #reshape localhour into a single data variable
    lts   = {b'01:00': 'localhour22', b'07:00': 'localhour04', b'13:00': 'localhour10', b'19:00': 'localhour16'}
    da_lt = xr.concat([gridded[key].assign_coords(localhour=np.array(lt,dtype=bytes)) for lt,key in lts.items()],dim='localhour')
    lt_bounds = np.array([[b'22:00',b'03:59'],[b'04:00',b'09:59'],[b'10:00',b'15:59'],[b'16:00',b'21:59']])
    lt_bounds = xr.DataArray(data=lt_bounds,dims=['localhour','bound'],coords={'localhour':da_lt.localhour})
    gridded = gridded.drop_vars(lts.values())
    gridded = gridded.assign({'localhour_counts': da_lt, 'localhour_bounds': lt_bounds})

    #transpose from attribute:varname:val to varname:attribute:val
    full_vars = [*list(gridded.data_vars),*list(gridded.coords)] #flat list of vars and coords
    attrs_by_varname = {k:{} for k in full_vars} #build nested dict with right keys
    for attrname,d in attrs_by_attrname.items():
        for varname,val in d.items():
            attrs_by_varname[varname][attrname] = val

    #apply attrs to dataset
    for varname in full_vars:
        gridded[varname].attrs = attrs_by_varname[varname]

    #add a "bounds" attribute to relevant coordinates
    gridded['height'].attrs['bounds']    = 'height_bounds'
    gridded['localhour'].attrs['bounds'] = 'localhour_bounds'

    #re-order data variables in dataset so the listing makes more sense
    sensical_ordering = (rename['height']['cloud_counts'],
                         rename['height']['total_counts'],
                         rename['height']['cloud_cover'],
                         rename['cover']['cloud_counts'],
                         rename['cover']['total_counts'],
                         rename['cover']['cloud_cover'],
                         rename['height']['attenuated_lidar_counts'],
                         rename['cover']['attenuated_lidar_counts'],
                         rename['height']['radar_surface_clutter_counts'],
                         'n_overpasses',
                         'n_days',
                         'localhour_counts',
                         'localhour_bounds',
                         'height_bounds')
    reordered_vars = {name:gridded[name] for name in sensical_ordering}
    gridded = xr.Dataset(data_vars = reordered_vars, coords=gridded.coords, attrs=gridded.attrs)

    #add some minor extra attributes for CF-compliance
    gridded.attrs['Conventions'] = 'CF-1.8'
    gridded.height.attrs['positive'] = 'up'
    #convert cloud cover and cloud fraction to percent since 'ratio' isn't a valid unit
    gridded[rename['height']['cloud_cover']].data = 100*gridded[rename['height']['cloud_cover']].data
    gridded[rename['cover']['cloud_cover']].data = 100*gridded[rename['cover']['cloud_cover']].data

    #reverse height direction to be increasing instead of decreasing (CF compliance)
    gridded = gridded.isel(height=slice(None, None, -1))

    ### EXPORT GRIDDED DATASET TO DISK ###
    print(f'saving file {fname}.nc')
    gridded.to_netcdf(os.path.join(args['outdir'],fname+'.nc'))


    ### DONE WITH DASK CLUSTER ###
    client.shutdown()

    ### DELETE INTERMEDIATE FILES SAVED TO DISK ###
    if not args['no_delete_intermediates']:
        print('deleting intermediate files...')
        combined_granules = list(listdir_fullpaths(new_combined_path))
        if all([path[-3:] == '.nc' for path in combined_granules]):
            print('all non-hidden files in {} ending in .nc... \ndeleting {} files...'
                  .format(new_combined_path,len(combined_granules)),end='')
            for granule in combined_granules:
                os.remove(granule)
                #print(granule)
            print('done.')
            hidden_files = os.listdir(new_combined_path)
            hidden_filepaths = [os.path.join(new_combined_path,hidden_file) for hidden_file in hidden_files]
            try:
                for hidden_filepath in hidden_filepaths:
                    print('deleting hidden file {}'.format(hidden_filepath))
                    os.remove(hidden_filepath)
                os.rmdir(new_combined_path)
                print(f'deleted directory {new_combined_path}')
            except OSError:
                print(f'ERROR: OS Error when deleting hidden files! Directory {os.path.split(new_combined_path)[-1]} not deleted')
                sys.exit(1)
        else:
            print(f'not all files in {new_combined_path} have .nc file extensions. Are you sure you want to delete these files?')
    else:
        print('no_delete_intermediates set to True. Leaving intermediate files in place...')

    print('success')
    sys.exit(0)

if __name__ == "__main__":
    main()