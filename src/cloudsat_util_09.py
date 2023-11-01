#"""
#Utility library for working with CloudSat granules for
#project building a combined radar-lidar level 3 data product.
#Uses pyhdf, which is built to read these HDF4-EOS files.
#
#Version 0.7 made 07/07/22 for script 8-5_build_cloudcover_grid_v6.2.py
#migrates new method defs from v6.1 and v6.
# NOT UPDATED CHANGELOG
# -- migrated methods from build_parallel_grid_v4.py into cloudsat_util_06.py.
#    see build_parallel_grid_v4.py for new methods and changes. Major features unique to v4:
#        -- modified read_convert_merge step to handle missing radar/lidar granules
#        -- modified proc_and_mask step to mask regions where lidar is likely attenuated
#    major additions unique to v3:
#        -- added printout routine for diagnostic printout plots
# -- modified apply_reshape and reshape_irregular to handle variable grid spacing
# -- modified reshape_irregular to put lat/lon coords in bin center instead of bin left/bottom
# -- modified proc_and_mask to keep doop_flag variable
# -- modified read_convert_merge_granule to calculate doop_flag field
# -- modified apply_reshape and reshape_irregular to handle variable gridsize
# -- added read_spline and save_spline methods for reading in doop curve
# -- added get_doop_flag routine
# -- added json and scipy.interpolate to import for working with doop spline curve
#"""

from pyhdf.SD import SD, SDC  # HDF4-EOS library
from pyhdf import HDF, VS, V #VS, V are necessary!

import os
import time
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs         # import projections
import cartopy.feature as cf       # import features
from datetime import datetime,timedelta      # for formatting names during read/write/plot, timedelta v6.2 last_day_of_month
import json                        # for reading in doop-spline data
from scipy import interpolate      # for using doop-splines

import sys,getopt                  # for processing command-line arguments

#ONLY NECESSARY WHILE GET_DOOP_FLAG IS IN THIS SCRIPT
import json                        # for reading in doop-spline data
from scipy import interpolate      # for using doop-splines

### HELPER METHOD FOR read_GEOPROF ###
# NOTE: just use np.squeeze() instead
def thin(l):
    ''' Remove leading empty axes from an array. '''
    dims = np.array(l).shape
    for dim in dims:
        if dim == 1:
            l = l[0]
        else:
            break
    return l


def read_swath(h):
    ''' Build nested dict from CloudSat swath attributes group.
        Tested for 2B-GEOPROF R04 and R05.
        Arguments: initialized pyhdf.HDF object
        Returns: dict of attribute values nested by ownership
        (e.g. Radar_Reflectivity.units => d['Radar_Reflectivity']['units'])
    '''

    # initialize
    full_attrs = {}
    v = h.vgstart()
    vs = h.vstart()

    # scan 'Swath Attributes' vgroup
    vid = v.find('Swath Attributes')
    vg = v.attach(vid)
    members = vg.tagrefs()
    for tag, ref in members:
        if tag == HDF.HC.DFTAG_VH:
            vd = vs.attach(ref)
            nrecs, intmode, fields, size, name = vd.inquire()
            #print("  vdata:",name, "tag,ref:",tag, ref)
            #print("    fields:",fields)
            #print("    nrecs:",nrecs)

            # get name of attribute owner
            sep = name.find('.')
            if sep != -1:
                class_name = name[:sep]
                attr_name = name[sep + 1:]
            # EXCEPTION: fillValues have special naming
            elif name[:4] == '_FV_':
                class_name = name[4:]
                attr_name = '_FillValue'
            else:
                class_name = 'shared'
                attr_name = name

            # store attr in subdict
            try:
                sub_attrs = full_attrs[class_name]
            except KeyError:
                full_attrs[class_name] = {}
                sub_attrs = full_attrs[class_name]

            # remove redundant axes and store
            attr_value = thin(vd.read(nRec=nrecs))
            # EXCEPTION: convert 'units' ints back to char
            if attr_name == 'units' and isinstance(attr_value,int):
                attr_value = chr(attr_value)
            sub_attrs[attr_name] = attr_value

            vd.detach()
        else:
            print('WARNING: found member of "Swath Attributes" group not of type vdata!')
    v.end()
    vs.end()
    return full_attrs

def padnan(indata,nrows=3):
    # built for 2B-GEOPROF and 2B-GEOPROF-LIDAR
    if len(indata.shape) == 2:
        inrows,incols         = indata.shape
        data                  = np.zeros((nrows,incols))
        data[:,:]             = np.nan
        data[:inrows,:incols] = indata
    elif len(indata.shape) == 1:
        inrows        = indata.shape[0]
        data          = np.zeros((nrows))
        data[:]       = np.nan
        data[:inrows] = indata
    elif len(indata.shape) == 3:
        inrows,incols,inlayers          = indata.shape
        data                            = np.zeros((nrows,incols,inlayers))
        data[:,:,:]                     = np.nan
        data[:inrows,:incols,:inlayers] = indata
    else:
        print('padnan ERROR: unrecognize number of dimensions {}'.format(len(indata.shape)))
    return data


#modified for v7.2.1 (I think to handle a really short or long granule?)
def read_GEOPROF(FILE_NAME,nprofiles=37100,keep_only=None):
    # READS EOS-HDF4 TO PROPERLY FORMATTED XARRAY DATASET
    # TESTED FOR R04 AND R05 2B-GEOPROF AND 2B-GEOPROF-LIDAR
    
    # pseudocode:
    # build dict of all attributes in file, determine ownership (if any)
    # iterate over ScientificDatasets (SDs)
    #     set aside SDs marked as coordinates
    #     declare DataArray for SD and apply owned attrs
    # initialize xarray dataset from SD DataArrays 
    # iterate over Vdatas
    #     treat scalar Vdatas as dataset-level attributes
    #     treat vector Vdatas as dataset DataArrays
    #         declare DataArray for Vdata and apply owned attrs
    #         set aside Vdatas marked as coordinates
    # apply unowned attrs as DataSet-level attrs
    # apply coordinate SDs and Vdatas as DataSet coords
    # return DataSet
    orig_nprofiles = 37000

    coord_keys = ['Profile_time','Latitude','Longitude','Height']

    hdf = SD(FILE_NAME, SDC.READ)

    # initialize underlying hdf reader
    h = HDF.HDF(FILE_NAME)

    # retrieve attributes for all fields in the HDF file
    swath_attrs = read_swath(h)

    # check that data is in a tested format
    alg_name = swath_attrs['shared']['algorithm_name']
    if alg_name not in ('2B-GEOPROF', '2B-GEOPROF-LIDAR'):
        print(
            'WARNING: routine not prepared to read data from algorithm {:}!'.format(alg_name))

    # read ScientificDatasets to DataArrays & append to DataSet
    ds_dict = {}
    coords_dict = {}
    for key in hdf.datasets().keys():

        # get data
        dset = hdf.select(key)
        data = dset[:, :]

        # determine dimension names (for 2B-GEOPROF-LIDAR)
        # NOTE: this is not a great way to do it, since # of profiles is not
        # consistent
        name, rank, dims, type, nattrs = dset.info()
        #skip the read if keep_only given and name not in it
        if keep_only:
            if name not in keep_only:
                continue
        dim_names = []
        for dim in dims:
            if dim == 5:
                dim_names.append('layer')
            elif dim == 125:
                dim_names.append('bin')
            # I have noticed granules with 37081 and 37082 profiles
            elif dim >= 20000: #CHANGED FROM 37000 FOR V6.2!
                orig_nprofiles = dim
                dim_names.append('profile')
            else:
                print(
                    'ERROR: unrecognized dimension of {:d} in ScientificDataset {:}!'.format(
                        dim, name))

        # add extra np.nan columns for consistent length
        if nprofiles and dim_names[0] == 'profile':
            data = padnan(data,nrows=nprofiles)

        # make DataArray combining data with swath attributes
        da = xr.DataArray(
            data,
            name=key,
            dims=dim_names,
            attrs=swath_attrs[key])

        # add to dict of DataArrays
        if key in coord_keys:
            coords_dict[key] = da
        else:
            ds_dict[key] = da

    # manually add start time (modified 9 Jan 2023)
    try:
        coords_dict['start_time'] = xr.DataArray(datetime.strptime(swath_attrs['shared']['start_time'],'%Y%m%d%H%M%S'),
                                                 attrs={'long_name':'time of first profile in UTC'})
    except ValueError: #added v7.2.1
        #some granules (e.g. 31-dec-2008 14240) have seconds = 60 :(
        #try the start time in the filename in case of error
        print(f"WARNING: bad start time {swath_attrs['shared']['start_time']} in granule {os.path.split(FILE_NAME)[-1].split('_')[1]}")
        coords_dict['start_time'] = xr.DataArray(datetime.strptime(os.path.split(FILE_NAME)[-1].split('_')[0],'%Y%j%H%M%S'),
                                                 attrs={'long_name':'time of first profile in UTC'})

    # init DataSet with dict
    ds = xr.Dataset(ds_dict)

    # scan vgroups 'Geolocation Fields' and 'Data Fields' for
    # the fields not already covered by 'Swath Attributes'
    # (since iterating over all vdatas would produce duplicates)

    data_groups = ['Geolocation Fields', 'Data Fields']
    v = h.vgstart()
    vs = h.vstart()
    for group in data_groups:
        vid = v.find(group)
        vg = v.attach(vid)
        members = vg.tagrefs()
        for tag, ref in members:
            if tag == HDF.HC.DFTAG_VH:
                vd = vs.attach(ref)  # get VD object for ref number
                nrecs, intmode, fields, size, name = vd.inquire()  # get info
                if keep_only: #if keep_only is given,
                    if name not in keep_only: #and it doesn't specify this var,
                        continue #skip the member read
                vd.setfields(name)  # initialize VD object
                vdata = vd.read(nRec=nrecs)  # read data

                # read in vector vdatas
                if nrecs > 1:
                    vdata = np.array(vdata)

                    if vdata.shape[1] == 1:
                        # if data is N x 1, cast to 1D
                        vdata = vdata.T[0]

                        # add extra np.nan columns for consistent length
                        if nprofiles:
                            vdata = padnan(vdata,nrows=nprofiles)

                        # write DataArray to DataSet
                        # use attrs if present

                        # NOTE: assumes dim is profile for all vdatas

                        if name in coord_keys:
                            target = coords_dict
                        else:
                            target = ds

                        try:
                            target[name] = xr.DataArray(
                                dims=('profile'), data=vdata, attrs=swath_attrs[name])
                        except KeyError:
                            target[name] = (('profile'), vdata)
                    else:
                        print(
                            'WARNING: found 2-D vdata "{}" in group "{}"!'.format(group, name))
                        print('\t Handler expects 0- or 1-D vdata and 2-D SDs')

                # read scalar vdata
                else:
                    # cast 1x1 array to scalar
                    vdata = thin(vdata)

                    # store as a DataSet variable if it has attributes
                    if name in swath_attrs.keys():
                        ds[name] = xr.DataArray(
                            data=vdata, attrs=swath_attrs[name])
                    # otherwise keep as a DataSet-level attr
                    else:
                        ds.attrs[name] = vdata

                vd.detach()  # close vdata
            elif tag == HDF.HC.DFTAG_NDG:
                ___ = None
                # do nothing if SD, since they were already handled
            else:
                print(
                    'WARNING: found member of "{}" group not of type vdata or sd!'.format(group))
        vg.detach()  # close vgroup

    v.end()  # close vgroup interface
    vs.end()  # close vdata interface
    h.close()  # close hdf file interface

    # apply unaffiliated swath attrs as dataset attrs
    for key in swath_attrs['shared']:
        ds.attrs[key] = swath_attrs['shared'][key]
    #if nprofiles:
        #print(f'read_GEOPROF: added {nprofiles-orig_nprofiles} np.nan columns to dimension \'profile\' to change length from {orig_nprofiles} to {nprofiles}')
    return ds.assign_coords(coords=coords_dict)

### DECODE/MASK METHODS ###
def decode(in_da,xdim='profile',ydim='bin',clip=True):
    # decode according to https://xarray.pydata.org/en/stable/user-guide/io.html#writing-encoded-data
    # with added clipping to valid_range
    
    #turn to float (to enable nan)
    #print(in_da.attrs)
    da = in_da.astype(float,keep_attrs=True)
    #print(da.attrs)
    
    scale_factor = da.attrs['factor']
    offset = da.attrs['offset']
    nan = da.attrs['_FillValue']
    valid_min, valid_max = da.attrs['valid_range']
    
    #turn _FillValue to np.nan
    da = xr.where(da==nan,np.nan,da)
    
    # (data > valid_max) = np.nan 
    da = xr.where(da>valid_max,np.nan,da)
    
    # (data < valid_min) = np.nan
    da = xr.where(da<valid_min,np.nan,da)
    
    #apply factor/offset
    da.data = scale_factor*da.data + offset
    
    return da

#DEV NOTE: CHANGED V8.1
#v8.1: removed setting surface clutter to nan
#v7.2: added del statements
def predecode_cpr(in_da,SurfaceHeightBin,xdim='profile',ydim='bin',clip=True):
    # set surface clutter and subsurface bins to missing_value
    
    #print(in_da.attrs)
    #turn to float (to enable nan)
    da = in_da.astype(float,keep_attrs=True)
    
    scale_factor = da.attrs['factor']
    offset = da.attrs['offset']
    nan = da.attrs['_FillValue']
    valid_min, valid_max = da.attrs['valid_range']
    
    #v8.1 removed the setting of surface clutter to NaN
    
    # set all subsurface bins to nan
    bin_arr    = np.broadcast_to(np.arange(0,125),(1,37100,125))
    bin_da     = xr.DataArray(data=bin_arr,dims=('start_time','profile','bin'))
    surf_da, _ = xr.broadcast(SurfaceHeightBin,bin_da)
    #surf bin is 1-indexed not 0-indexed
    da.data = xr.where(bin_da>=surf_da,nan,da.data) 
    del nan, scale_factor, offset, valid_min, valid_max, bin_arr, bin_da, surf_da
    return da

#added v7.2
#modified (added try/except) for v7.2.1
def predecode_vfm(in_da,SurfaceHeightBin,xdim='profile',ydim='bin',clip=True):
    #just set subsurface bins to missing_value
    
    #print(in_da.attrs)
    #turn to float (to enable nan)
    da = in_da.astype(float,keep_attrs=True)
    try: #modified v7.2.1 since this fails for -k radar
    	nan = da.attrs['_FillValue']
    except KeyError:
        nan = -9 #CloudFraction fill value
    
    # set all subsurface bins to nan
    bin_arr    = np.broadcast_to(np.arange(0,125),(1,37100,125))
    bin_da     = xr.DataArray(data=bin_arr,dims=('start_time','profile','bin'))
    surf_da, _ = xr.broadcast(SurfaceHeightBin,bin_da)
    #surf bin is 1-indexed not 0-indexed
    da.data = xr.where(bin_da>=surf_da,nan,da.data)
    del nan, bin_arr, bin_da, surf_da
    return da

def mask(dx,radar_thresh=0,lidar_thresh=0.5):
    
    # Q.O.L. input flexiblity:
    if isinstance(dx,xr.Dataset) == True:
        alg_name = dx.attrs['algorithm_name']
        if alg_name == '2B-GEOPROF':
            inda = dx['CPR_Cloud_mask']
            kind = 'radar'
        elif alg_name == '2B-GEOPROF-LIDAR':
            inda = dx['CloudFraction']
            kind = 'lidar'
        else:
            print('ERROR: algorithm not 2B-GEOPROF or 2B-GEOPROF-LIDAR!')
            return None
    elif isinstance(dx,xr.DataArray):
        inda = dx
        if inda.name == 'CPR_Cloud_mask':
            kind = 'radar'
        elif inda.name == 'CloudFraction':
            kind = 'lidar'
        else:
            print('ERROR: dataArray not "CPR_Cloud_mask" or "CloudFraction"!')
            return None
    else:
        print('ERROR: input of type {} not of type xarray.DataArray or xarray.Dataset!'.format(type(dx)))
        return None
    
    # main code:
    if kind == 'radar':
        thresh = radar_thresh
    elif kind == 'lidar':
        thresh = lidar_thresh
   
    xdim = 'profile'
    da = inda.copy()
    
    # set all non-nan values below thresh to 0
    da = xr.where(np.logical_and(da < thresh,~np.isnan(da)),0,da)
    
    # set all non-nan values above 0 to 1
    da = xr.where(np.logical_and(da > 0,~np.isnan(da)),1,da)
    
    return da

def proc_mask(in_cpr,in_vfm,radar_thresh=10,lidar_thresh=0.1):
    cpr_decoded = decode(predecode_cpr(in_cpr['CPR_Cloud_mask']))
    vfm_decoded = decode(in_vfm['CloudFraction'])
    cpr_binary  = mask(cpr_decoded,radar_thresh=radar_thresh)
    vfm_binary  = mask(vfm_decoded,lidar_thresh=lidar_thresh)
    
    vfm_binary.name = 'binary_mask'
    cpr_binary.name = 'binary_mask'
    
    vfm_binary.attrs['thresh'] = lidar_thresh
    cpr_binary.attrs['thresh'] = radar_thresh
    
    return in_cpr.assign(binary_mask=cpr_binary), in_vfm.assign(binary_mask=vfm_binary)

#NOTE DEV: CHANGE V8.1
# added radar surface clutter
# added predecode_vfm
#added predecode vfm and attenuated mask
# changed to assign instead of rebuild dataset
def proc_and_mask(in_cv,radar_thresh=20,lidar_thresh=1,attenuate_lidar=True, count_clutter=True):

    #decode and mask to binary for vfm
    vfm_decoded = decode(predecode_vfm(in_cv['CloudFraction'],in_cv['SurfaceHeightBin']))
    vfm_binary  = mask(vfm_decoded,lidar_thresh=lidar_thresh)

    #decode and mask to binary for cpr
    # not the same because we save surface clutter before we get rid of it
    cpr_decoded = decode(predecode_cpr(in_cv['CPR_Cloud_mask'],in_cv['SurfaceHeightBin'])) #mask subsurface bins, decode to float adding NaNs
    cluttered_bins = cpr_decoded == 5 #save the surface-and-above bins with surface clutter
    cpr_decoded = cpr_decoded.where(cpr_decoded!=5) #set surface clutter to NaN
    cpr_binary  = mask(cpr_decoded,radar_thresh=radar_thresh) #then apply the threshold

    #set regions where lidar is attenuated to nan
    if attenuate_lidar == True:
        combined_vis = calculate_combined_vis(cpr_binary,vfm_binary)
        vfm_binary, attenuated_bins = mask_attenuated_lidar_nods(vfm_binary,combined_vis)

    #combined to dataset
    vfm_binary.name = 'vfm_binary'
    cpr_binary.name = 'cpr_binary'

    vfm_binary.attrs['thresh'] = lidar_thresh
    cpr_binary.attrs['thresh'] = radar_thresh

    new_vars = {'vfm_binary':vfm_binary,
                'cpr_binary':cpr_binary}

    if attenuate_lidar: 
        new_vars['attenuated_lidar_counts'] = attenuated_bins
    if count_clutter:
        new_vars['radar_surface_clutter_counts'] = cluttered_bins

    in_cv = in_cv.drop_vars(('CPR_Cloud_mask','CloudFraction')).assign(new_vars)
    
    del vfm_binary, cpr_binary, cpr_decoded, vfm_decoded, new_vars, cluttered_bins
    return in_cv

def calculate_combined_vis(cpr_binary,vfm_binary):
    vfm_fillna = xr.where(xr.ufuncs.isnan(vfm_binary) & ~xr.ufuncs.isnan(cpr_binary),0,vfm_binary)
    cpr_fillna = xr.where(xr.ufuncs.isnan(cpr_binary) & ~xr.ufuncs.isnan(vfm_binary),0,cpr_binary)
    combined_vis = 2*vfm_fillna+cpr_fillna
    return combined_vis

#v7.2: modified to also the pruned mask (ie samples w/ attenuation inferred)
#added del statements
def mask_attenuated_lidar_nods(vfm_binary,combined_vis):
    """Generous masking of regions where lidar is inferred
    to be attenuated.
    
    Masks lidar below the lowest both-to-radar transitions
    which do not contain any further lidar cloud.
    
    Parameters:
    cm (xarray.Dataset): Dataset containing radar, lidar, and combined binary masks
    
    Returns:
    vfm_binary_mask (xarray.DataArray): DataArray of np.nan-masked lidar binary cloud mask
    """
    #calculate bin-to-bin differences by profile:
    diff_small = combined_vis.diff('bin') #missing top row
    #zero-pad top bin-row of diff so sizes match
    diff_data = np.zeros(combined_vis.shape)
    diff_data[:,:,1:] = diff_small.data
    
    #get profile-normalized cumulative sum of both-to-radar transitions:
    #set all both-to-radar pixels equal to 1
    radar_and_diff = xr.where((diff_data==-2) & (combined_vis==1),1,0)
    #get the number of previous both-to-radar transitions
    csum = radar_and_diff.cumsum(dim='bin')
    #normalize so that the lowest both-to-radar equals 1 down to surface
    csum_norm = csum/(csum.max(dim='bin',skipna=True)) #ignore in-column missing values
    #set nan (no both-to-radar) columns equal to 0
    csum_norm_fillna = csum_norm.fillna(0)
    
    #exclude all profiles which have lidar cloud below the last both-to-radar:
    #find lidar cloud pixels under the lowest-both-to-radar transition
    prune_da = xr.where((csum_norm_fillna == 1) & (vfm_binary == 1),1,0)
    #reduce to dims of [start_time,profile]
    prune_profs = prune_da.any(dim='bin')
    #broadcast back to [start_time,profile,bin] (entire profile = 1)
    prune_profs_da, _ = xr.broadcast(prune_profs,prune_da)
    #set all profiles with lidar cloud below last both-to-radar = 0
    csum_pruned = xr.where(prune_profs_da==1,0,csum_norm_fillna)
    
    #apply mask to vfm_binary:
    vfm_binary_masked = xr.where(csum_pruned==1,np.nan,vfm_binary)
    
    del vfm_binary, diff_small, diff_data, radar_and_diff, 
    del csum, csum_norm, csum_norm_fillna, prune_da, prune_profs,
    del prune_profs_da, _
    return vfm_binary_masked, (csum_pruned==1).astype(bool)

### PLOT ROUTINES ###
def contour(ds,key='CPR_Cloud_mask',ax=None,cb=True,coords='bin',n=10,cmap='viridis'):
    data = ds[key].squeeze().data

    if coords == 'height':
        longname_t = ds['Profile_time'].attrs['long_name']
        units_t    = ds['Profile_time'].attrs['units']
        units_h    = ds['Height'].attrs['units']
    try:
        units      = ds[key].attrs['units']
    except KeyError:
        units = key
    try:
        long_name  = ds[key].attrs['long_name']
    except KeyError:
        long_name = key

    dataf = np.array(data.astype(float))

    # Contour the data.
    if ax == None:
        fig,ax=plt.subplots(1,1,dpi=200,figsize=(10,2))
    if coords=='height':
        time = ds['Profile_time'].data
        height = ds['Height'].data
        #fill any padded nan columns
        t = np.nan_to_num(time,nan=-100)
        H = np.nan_to_num(height,nan=-1e5)
        H = H.T
        T, _ = np.meshgrid(t,H[:,0])
        im = ax.contourf(T,H,data.T,n,cmap=cmap)
        ax.set_xlim((0,t.max()))
        ax.set_ylim((0,H.max()))
        ax.set_xlabel(longname_t)
        ax.set_ylabel('Height ({})'.format(units_h))
    if coords=='bin':
        nprofs,nbins = ds.dims['profile'],ds.dims['bin']
        im = ax.contourf(data.T,n,origin='upper',extent=(0,nprofs,nbins,0),cmap=cmap)
        ax.set_ylim((nbins-1,0))
        ax.set_xlim((0,nprofs-1))
        ax.set_xlabel('profile')
        ax.set_ylabel('vertical height bin')

    fmt ='%Y%m%d%H%M%S'
    dt = datetime.strptime(ds.attrs['start_time'],fmt)
    start_time = dt.strftime('%Y-%m-%d %H:%M')
    basename = '{} UTC {}-{}'.format(start_time,ds.attrs['algorithm_name'],ds.attrs['algorithm_version'][:-1])
    ax.set_title(basename)

    if cb:
        cb = plt.colorbar(im,ax=ax)
        cb.set_label(units)
        return ax,cb
    
    return ax

#modified to fix subplot labels and use outdir for save path
def printout(n,thin_radar_filepaths,tcv,tcm,tcc,fname,outdir):
    fig, axes = plt.subplots(3,2,figsize=(10,8),dpi=150,sharex=True,sharey=True)
    granule = fpath2granule(thin_radar_filepaths[n])

    contour(tcv.isel(start_time=n),key='CPR_Cloud_mask',ax=axes[0][0])
    contour(tcv.isel(start_time=n),key='CloudFraction',ax=axes[0][1])
    contour(tcm.isel(start_time=n),key='cpr_binary',ax=axes[1][0])
    contour(tcm.isel(start_time=n),key='vfm_binary',ax=axes[1][1])
    contour(tcc.isel(start_time=n),key='cloud_counts',ax=axes[2][0])
    contour(tcc.isel(start_time=n),key='total_counts',ax=axes[2][1])

    #titles = [['cpr raw','vfm raw'],['cpr decoded+binary','vfm decoded+binary'],['combined_vis','vfm atten-proc'],['total counts','cloud counts']]
    titles = [['cpr raw','vfm raw'],['cpr decoded+binary','vfm decoded+binary'],['cloud counts','total counts']]
    for i,row in enumerate(axes):
        for j,ax in enumerate(row):
            ax.set_title(titles[i][j])
    fig.suptitle(f"granule {granule} for file {fname}.nc")
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    fname=f"granule{granule}_{fname}.png"
    plt.savefig(os.path.join(outdir,fname),transparent=False,facecolor='white')

#NOTE DEV: CHANGED FOR V8.1
#ADDED V8.1
def print_dataquality(tcv):
    '''takes a Dataset with field 'Data_quality' from 2B-GEOPROF and prints
    the frequency of nonzero (i.e. bad) data quality profiles and what
    fraction of the bad profiles each flag makes up (e.g. which flags dominate the bad data)'''
    
    #load data quality values
    dq = tcv.Data_quality.to_numpy().flatten()

    #get counts of ints for which data_quality is non-nan
    #7-bit int, so possible values are 0 to 255
    counts, bins = np.histogram(dq,bins=np.arange(0,256))

    #get list of arrays showing which ints have which bits true
    #array i in list shows whether array index int j has bit i true or false
    bit_groups = [(np.arange(0,255) & 2**n).astype(bool) for n in range(8)]

    #from 1B-CPR PDICD explaining Data_quality bit flags
    bit_meanings = {0: 'RayStatus_validity not normal',
                    1: 'GPS data not valid',
                    2: 'Temperatures not valid',
                    3: 'Radar telemetry data quality not normal',
                    4: 'Peak power not normal',
                    5: 'CPR calibration maneuver',
                    6: 'Missing frame',
                    7: 'Data advisory, check website'}

    print('PRINTING DATA QUALITY INFORMATION')
    counts_nonzero = np.sum(counts[1:])
    print(f'out of {len(dq)} profiles, {counts_nonzero} are bad, which is {counts_nonzero/len(dq)*100:.02f}%')
    print('the breakdown of bad profiles by bit is')
    for bit,mask in enumerate(bit_groups):
        counts_bit = counts[mask].sum()
        print(f'bit {bit}: {counts_bit} is {counts_bit/counts_nonzero*100:.02f}% ({bit_meanings[bit]})')

### BUILD INTERMEDIATE NETCDF4 ROUTINES (I/O) ###

def compare(a,b):
    comp = (a == b) | (np.isnan(a) & np.isnan(b))
    return comp.all()

# must skip hidden files for uniform filenames
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
def listdir_fullpaths(path):
    fnames = list(listdir_nohidden(path))
    fnames = sorted(fnames,key=lambda s: s.split('_')[1]) #sort by granule number
    fpaths = [os.path.join(path,fname) for fname in fnames]
    return fpaths

def read_convert_merge_granule(filepaths,new_folder_path,keep_only=None):
    #unpack
    radar_filepath, lidar_filepath = filepaths
    
    #TODO: MAKE 'DUMMY' METHOD TO RETURN RIGHT SHAPE ALL -9s
    #read HDF4-EOS granule into xarray.Dataset
    if ((radar_filepath!=None) and (lidar_filepath!=None)):
        cpr = read_GEOPROF(radar_filepath,nprofiles=37100,keep_only=keep_only)
        vfm = read_GEOPROF(lidar_filepath,nprofiles=37100,keep_only=keep_only)
    if ((radar_filepath==None) and (lidar_filepath!=None)):
        vfm = read_GEOPROF(lidar_filepath,nprofiles=37100,keep_only=keep_only)
        cpr = dummy(vfm) 
    if ((radar_filepath!=None) and (lidar_filepath==None)):
        cpr = read_GEOPROF(radar_filepath,nprofiles=37100,keep_only=keep_only)
        vfm = dummy(cpr)
    
    #compare files
    if ((radar_filepath!=None) and (lidar_filepath!=None)):
        if cpr.attrs['granule_number'] != vfm.attrs['granule_number']:
            print('ERROR: non-matching granule numbers! Quitting')
            return
    
    #build filename similiar to the original granule filename
    time = str(cpr.coords['start_time'].dt.strftime('%Y%m%d%H%M%S').data)
    granule = int(cpr.attrs['granule_number'])
    alg_name = str(cpr.attrs['algorithm_name'])+'-COMBINED'
    alg_ver = str(cpr.attrs['algorithm_version'])
    fname = '{}_{}_CS_{}-{}0.nc'.format(time,granule,alg_name,alg_ver)
    
    #merge datasets
    ds = xr.merge([cpr,vfm],compat='no_conflicts')
    
    #assign do-op flag
    ds = ds.assign(doop_flag=get_doop_flag(cpr))
    ds = ds.drop_vars(['Data_status']) #only used for doop_flag calc

    #save
    ds.attrs['coordinates']='Height start_time Profile_time Latitude Longitude'
    ds.attrs['description']='Merged 2B-GEOPROF and 2B-GEOPROF-LIDAR granule'
    ds.expand_dims(start_time=[ds.start_time.dt])
    ds.to_netcdf(path=os.path.join(new_folder_path,fname),encoding=None,mode='w')

def fpath2granule(path):
    #NOTE: do I actually need this special behavior?
    if path == None:
        return None
    ## extract granule number from granule filepath
    return os.path.split(path)[1].split('_')[1]

def granule2fpath(granule,granules,fpaths):
    if granule == None:
        return None
    else:
        i = (np.asarray(granules).astype(int) == int(granule)).nonzero()[0][0] #0th element of 0th dimension
        return fpaths[i]
    
def matchfill(a,b):
    ''' Make lists a,b the same length by 
    (A) adding 'None's to b if len(a)>len(b)
    (B) deleting elems from b if len(a)<len(b)
    '''
    return [x if (x in b) else None for x in a]

def list_in(list1,list2):
    # check if a smaller list1 is contained within list2
    return len(set(list1).intersection(list2)) == len(list1)

def intersection(radar_filepaths,lidar_filepaths,quiet=False,keep_granule='both'):
    ### REMOVES ALL GRANULES FOR WHICH BOTH RADAR AND LIDAR ARE NOT PRESENT ###
    
    #build list of granule numbers
    radar_granules = [fpath2granule(path) for path in radar_filepaths]
    lidar_granules = [fpath2granule(path) for path in lidar_filepaths]
    
    #find intersection
    if keep_granule == 'both':
        radar_keep_granules  = list(set(lidar_granules).intersection(radar_granules))
        lidar_keep_granules  = radar_keep_granules
    elif keep_granule == 'radar':
        radar_keep_granules  = radar_granules
        lidar_keep_granules  = matchfill(radar_granules,lidar_granules)
    elif keep_granule == 'lidar':
        radar_keep_granules  = matchfill(lidar_granules,radar_granules)
        lidar_keep_granules  = lidar_granules
    else:
        print(f"ERROR: keep_granule value {keep_granule} not recognized! Only 'both','radar','lidar' recognized! Quitting...")
        sys.exit(1)
    
    #retrieve filepaths corresponding to intersecting granule numbers
    thinned_radar_filepaths = [granule2fpath(granule,radar_granules,radar_filepaths) for granule in radar_keep_granules]
    thinned_lidar_filepaths = [granule2fpath(granule,lidar_granules,lidar_filepaths) for granule in lidar_keep_granules]
    
    #summarize
    if quiet == False:
        print(f'keep_granule == {keep_granule}, so ',end='')
        if keep_granule == 'both':
            print('exclude all granules for which both radar and lidar are not present.')
            print(f'excluded {len(lidar_granules)-len(lidar_keep_granules)}/{len(lidar_granules)} lidar-only granules ',end='') 
            print(f'and {len(radar_granules)-len(radar_keep_granules)}/{len(radar_granules)} radar-only granules, ')
        elif keep_granule == 'radar':
            print('include dummy lidar granules when radar is present but lidar is not.')
            print(f'padded lidar_granules with {len(lidar_keep_granules)-len(lidar_granules)} dummy granules')
        elif keep_granule == 'lidar':
            print('include dummy radar granules when lidar is present but radar is not.')
            print(f'padded radar_granules with {len(radar_keep_granules)-len(radar_granules)} dummy granules')

        n_radarskips = len(np.where(np.diff(np.array(radar_granules).astype(int))!=1)[0])
        n_lidarskips = len(np.where(np.diff(np.array(lidar_granules).astype(int))!=1)[0])
        print(f'with {n_radarskips} skips in radar granule counting and {n_lidarskips} skips in lidar granule counting')
    
    return thinned_radar_filepaths, thinned_lidar_filepaths

def dummy(other_ds,
          vars_dims = {'CPR_Cloud_mask':['profile','bin'],
             'CloudFraction':['profile','bin'],
             'DEM_elevation':['profile'],
             'Data_quality':['profile'],
             'SurfaceHeightBin':['profile']
            },
          fill_values = {'CPR_Cloud_mask':-9,
             'DEM_elevation':-9,
             'Data_quality':-9,
             'SurfaceHeightBin':0,
             'CloudFraction':-9
            }
         ):
    '''Build a complementary dataset to fill in missing variables.

    Parameters:
    other_ds -- dataset with missing variables to build complement of
    vars_dims -- variables to fill in with dimension labels specified
    fill_values -- fill values for missing variables (must be int)
    '''
    dim_sizes = other_ds.dims
    var_names = set(vars_dims.keys())-set(other_ds.keys())
    vars_dims = dict([(key,vars_dims[key]) for key in var_names])
    data_vars = {}
    for var,dims in vars_dims.items():
        data = np.zeros([dim_sizes[dim] for dim in dims]).astype(int)
        data = data+fill_values[var]
        data_vars[var] = (dims,data)
    return xr.Dataset(data_vars=data_vars)

#NEW METHOD v6 (helper for handling of summit mode time range)
def last_day_of_month(any_day):
    # FROM SO "How to get the last day of the month?"
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, 
    # or said programattically said, the previous day of the first of next month
    return next_month - timedelta(days=next_month.day)

#NEW METHOD v6 (helper for listing files over multiple subfolders)
def flatten_list(t):
    #from S/O "How to make a flat list out of a list of lists?"
    return [item for sublist in t for item in sublist]

#NEW METHOD v6
#DEV NOTE: CHANGED THIS FOR V8.2
#changed v8.2 to get the last granule from the day before the start date
def get_granule_paths(dayfi,dayla,year_path):
    #for cloudsat ftp server filestructure
    #get all granules in day/year range when granules are in folders organized by day
    
    #get directories in time range
    all_daydir_names = list(listdir_nohidden(year_path)) #list all day folders under year
    all_daydir_paths = [os.path.join(year_path,fname) for fname in all_daydir_names] #get full paths
    all_days = np.array([int(i) for i in all_daydir_names]) #convert to int array
    daymap  = (dayfi<=all_days) & (all_days<=dayla) #boolean map of days to keep
    keep_daydir_paths = np.asarray(all_daydir_paths)[daymap] #list of folders to use
    keep_daydir_paths = [str(s) for s in keep_daydir_paths] #use list of strs instead of numpy

    #get filepaths in list of directories
    granule_paths = [[os.path.join(path,name) for name in os.listdir(path)] for path in keep_daydir_paths] #nested list
    granule_paths = flatten_list(granule_paths) #flattened list
    granule_paths = sorted(granule_paths,key=lambda s: os.path.split(s)[1].split('_')[1]) #sort by granule number
    #NB 5/21: I think 'intersection' might not actually require filepath lists sorted by granule number,
    # but I added it here anyway since listdir_fullpaths sorts by granule
    
    #NEW V8.2
    #since granules don't begin exactly at the start of the day, include the last granule from the previous day
    path,year = os.path.split(year_path) #get year back from year_path
    dt_dayfi = pd.to_datetime(f'{year}-{dayfi}',format='%Y-%j') #convert year+dayfi to datetime
    dt_prev = dt_dayfi-pd.Timedelta(1,unit='days') #get the previous day
    daypath_prev = os.path.join(path,str(dt_prev.year),f'{dt_prev.dayofyear:03d}') #get filepath to previous day
    if os.path.exists(daypath_prev): #if there's data for the previous day,
        granulepath_prev = sorted(listdir_fullpaths(daypath_prev))[-1] #get the last granule of that previous day
        granule_paths.insert(0,granulepath_prev) #add it to the start of the list of granules to use
    
    return granule_paths

### COUNT CLOUD ROUTINES ###
def count_clouds(ds, sumdim='profile'):
    # currently unused as of 7-1 and the parallelization attemps
    cloud_counts_rl = ((ds['vfm_binary'] > 0).astype(bool) | (ds['cpr_binary'] > 0).astype(bool)).sum(dim=sumdim)
    total_counts_rl = (ds['vfm_binary'].notnull() | ds['cpr_binary'].notnull()).sum(dim=sumdim)
    return (cloud_counts_rl,total_counts_rl)

def get_cloud_counts_nosum(ds, sumdim='profile'):
    da = (ds['vfm_binary'] > 0).astype(bool) | (ds['cpr_binary'] > 0).astype(bool)    
    return ds.assign(cloud_counts=da)

def get_total_counts_nosum(ds, sumdim='profile'):
    da =  (ds['vfm_binary'].notnull() | ds['cpr_binary'].notnull())
    return ds.assign(total_counts=da)

def get_cloud_counts_nosum_radaronly(ds, sumdim='profile'):
    da = (ds['cpr_binary'] > 0).astype(bool)    
    return ds.assign(cloud_counts=da)

def get_total_counts_nosum_radaronly(ds, sumdim='profile'):
    da =  ds['cpr_binary'].notnull()
    return ds.assign(total_counts=da)

def get_cloud_counts_nosum_lidaronly(ds, sumdim='profile'):
    da = (ds['vfm_binary'] > 0).astype(bool)    
    return ds.assign(cloud_counts=da)

def get_total_counts_nosum_lidaronly(ds, sumdim='profile'):
    da =  ds['vfm_binary'].notnull()
    return ds.assign(total_counts=da)

#NOTE V7.2: I might have changed this to handle the new qc vars, but 
# it also might be exactly the same at cloudat_util_07
def apply_cloud_def(ds,alg='min1',bounds={'lo':93,'hi':75}):
    #93: 2600 masl +/- 100 m
    #75: 6950 masl +/- 100 m
    
    # takes counts dataset and applies a cloud definition to classify
    # whether a profile is clear or cloudy.
    
    #takes counts dataset with dims {'start_time','profile','bin'},
    # applied a definition of a cloudy profile given by 'alg',
    # and returns a dataset with dims {'start_time','profile'}
    
    #ds: binary cloud_counts,total_counts with dims {'start_time','profile','bin'}
    #alg: definition of a cloudy profile
    #bounds: bin thresholds for high/middle/low clouds
    
    #helper function for at least 2 contiguous bins
    min2 = lambda cm: cm.shift({'bin':1},fill_value=False) & cm
    #definition of thick clouds
    if alg=='thick': alg='min20' #20x240m=4.8km
    if alg=='any': alg='min2' #480 m thick
    
    #apply different definitions of a cloudy profile
    
    #minimum N contiguous cloudy bins in a profile
    #just process cloud counts
    if alg[:3]=='min':
        cloud_counts = ds.cloud_counts
        N=int(alg[3:])
        if N==1:
            cloud_counts = cloud_counts
        elif N==2:
            cloud_counts = min2(cloud_counts)
        elif N>2:
            #bfill method shown to be quickest in 8-3-3 def minN_bfill2
            #requires the package 'bottleneck' to be installed
            fnan = xr.where(cloud_counts==True,np.nan,0) #replace True with NaN
            fill = fnan.bfill(dim='bin',limit=N-1) #fill up to N-1 nans with 0
            cloud_counts = xr.where(np.isnan(fill),True,False) #replace NaN with True
    
    #NB: low cloud cover fraction is a matter of some subtlety, since
    # there are fewer low cloud counts than middle or high. So the issue
    # is whether or not total_counts should be the same or different for 
    # different vertical categories. And when calculating the unique
    # cloud cover fractions, which version of the total counts to use
    # at all. I think it would be best for total_counts to vary by type
    # and just include some duplicates e.g. for any/thick. For the unique
    # clouds, I think the correct definitions are for e.g. unique low
    #   * cloud_counts: low and not (high or mid)
    #   * total_counts: low and high and mid
    # but then this means profiles with thick clouds would be excluded
    # from the fraction if the low bins were not observed.
    
    #non-unique vertical definitions
    #process cloud_counts and total_counts
    elif alg=='high':
        ds = ds.isel({'bin':slice(0,bounds['hi'])})
        cloud_counts = min2(ds.cloud_counts)
    elif alg=='middle':
        ds = ds.isel({'bin':slice(bounds['hi'],bounds['lo'])})
        cloud_counts = min2(ds.cloud_counts)
    elif alg=='low':
        ds = ds.isel({'bin':slice(bounds['lo'],None)})
        cloud_counts = min2(ds.cloud_counts)
        
    #unique vertical definitions
    # (note we must return within the case since
    #  we must fully calculate all three options)
    elif alg in ('uniquehigh','uniquemiddle','uniquelow'):
        curr_key = alg[6:]
        
        #process high, middle, and low
        levels = {
            'high':ds.isel({'bin':slice(0,bounds['hi'])}),
            'middle':ds.isel({'bin':slice(bounds['hi'],bounds['lo'])}),
            'low':ds.isel({'bin':slice(bounds['lo'],None)})
        }
        for key,ds_i in levels.items():
            ds_i['cloud_counts'] = min2(ds_i['cloud_counts'])
            ds_i = ds_i.any(dim='bin')
            levels[key] = ds_i
        
        #pick out current val and remove it from dict
        current = levels.pop(curr_key)
        other1, other2 = levels.values()
        #compare current key against other two keys
        cloud_counts = current.cloud_counts & ~(other1.cloud_counts | other2.cloud_counts) #unique X if no Y or Z
        total_counts = current.total_counts & other1.total_counts & other2.total_counts #obs X possible if obs Y and obs Z
        
        #drop and re-assign our cloud/total counts with no bin dim
        ds = ds.drop_dims('bin')
        ds = ds.assign({'cloud_counts':cloud_counts,'total_counts':total_counts})
        return ds
    
    else:
        print(f'cloud definition {alg} not recognized! Quitting...')
        sys.exit(1)
    
    
    #shared processing for processing whole profiles
    ds['cloud_counts'] = cloud_counts
    
    #ADDED V7.2
#     for name in ds:
#         if 'bin' in ds[name].dims:
#             ds[name] = ds[name].any(dim='bin')
#     print('new util oct19')
    return ds.any(dim='bin')

#NOTE DEV: CHANGED FOR V8.1

#NOTE V7.2: I might have changed this to handle the new qc vars, but 
# it also might be exactly the same at cloudat_util_07

# v8.2: changed height thresholds to be on the ISA + better height ranges from the survey
# removed min2 for level types
# v8.1: revise unique counts methodology:
#    -- it's a 'unique high' Y/N count if there's high cloud in the relevant region and there ISN'T cloud at the other levels.
#       this makes no stipulation about the surface pressure in the atmospheric column. There is a 'unique mid' cloud if all the cloud in the column
#       is at the stated level. so the condition then is simply that the level of interest be defined.
#    -- so the revision is: total count = current level counts (then this also shouldn't need to do `cloud_counts = cloud_counts & total_counts`)
def apply_all_cloud_defs(ds,bounds={'lo':91,'hi':77}):
    #680 mb: 3238.1 masl
    #440 mb: 6503 masl
    #3238.1 m: bin 91 (2997 to 3237 m) (yes I know it's one lower it's right on the boundary)
    #6503 m: bin 77 (6355 to 6595 m)
    
    # takes counts dataset and applies a cloud definition to classify
    # whether a profile is clear or cloudy.
    
    #takes counts dataset with dims {'start_time','profile','bin'},
    # applied a definition of a cloudy profile given by 'alg',
    # and returns a dataset with dims {'start_time','profile'}
    
    #ds: binary cloud_counts,total_counts with dims {'start_time','profile','bin'}
    #alg: definition of a cloudy profile
    #bounds: bin thresholds for high/middle/low clouds
    
    #helper function for at least 2 contiguous bins
    min2 = lambda cm: cm.shift({'bin':1},fill_value=False) & cm

    #store each counts of type in a dict
    cloud_counts_by_type = { }
    total_counts_by_type = { }
    
    
    #apply different definitions of a cloudy profile
    #minimum N contiguous cloudy bins in a profile
    #just process cloud counts
    for alg in ('any','thick'):
        #definition of thick clouds
        if alg=='thick': alg_loc='min20' #20x240m=4.8km
        if alg=='any': alg_loc='min1' #240 m
        cloud_counts = ds.cloud_counts
        N=int(alg_loc[3:])
        if N==1:
            cloud_counts_by_type[alg] = cloud_counts
        elif N==2:
            cloud_counts_by_type[alg] = min2(cloud_counts)
        elif N>2:
            #bfill method shown to be quickest in 8-3-3 def minN_bfill2
            #requires the package 'bottleneck' to be installed
            fnan = xr.where(cloud_counts==True,np.nan,0) #replace True with NaN
            fill = fnan.bfill(dim='bin',limit=N-1) #fill up to N-1 nans with 0
            cloud_counts_by_type[alg] = xr.where(np.isnan(fill),True,False) #replace NaN with True
        
        total_counts_by_type[alg] = ds.total_counts #for any, thick, say if >=1 bin in profile is defined then it's a valid profile
        
    
    #NB: low cloud cover fraction is a matter of some subtlety, since
    # there are fewer low cloud counts than middle or high. So the issue
    # is whether or not total_counts should be the same or different for 
    # different vertical categories. And when calculating the unique
    # cloud cover fractions, which version of the total counts to use
    # at all. I think it would be best for total_counts to vary by type
    # and just include some duplicates e.g. for any/thick. For the unique
    # clouds, I think the correct definitions are for e.g. unique low
    #   * cloud_counts: low and not (high or mid)
    #   * total_counts: low and high and mid
    # but then this means profiles with thick clouds would be excluded
    # from the fraction if the low bins were not observed.
    
    #non-unique vertical definitions
    #process cloud_counts and total_counts
    for alg,bin_slice in {'high':slice(0,bounds['hi']),
                      'middle':slice(bounds['hi'],bounds['lo']),
                      'low':slice(bounds['lo'],None)}.items():
        #process cloud_counts
        da = ds.cloud_counts
        cloud_counts_by_type[alg] = da.isel({'bin':bin_slice})

        #process total_counts
        da = ds.total_counts
        total_counts_by_type[alg] = da.isel({'bin':bin_slice})
        
    
    #check if any bin in profile satisfies
    #this is needed for the unique calculations
    cloud_counts_by_type = {alg:cc.any(dim='bin') for alg,cc in cloud_counts_by_type.items()}
    total_counts_by_type = {alg:cc.any(dim='bin') for alg,cc in total_counts_by_type.items()}
    
    #unique vertical definitions
    # (note we must return within the case since
    #  we must fully calculate all three options)
    for alg in ('uniquehigh','uniquemiddle','uniquelow'):        
        #get names of current and other two levels
        levels = ['high','middle','low']
        current_level = alg[6:] #e.g. high, middle, low
        other1_level, other2_level = list(set(levels)-set([current_level])) #remove current level
        
        #retrieve cloud counts variables (T/F by profile)
        current = cloud_counts_by_type[current_level]
        other1  = cloud_counts_by_type[other1_level]
        other2  = cloud_counts_by_type[other2_level]
        
        #compare current key against other two keys
        cloud_counts = current & ~(other1 | other2) #unique X if no Y or Z
        
        #retrieve total counts variables
        current = total_counts_by_type[current_level]
        other1  = total_counts_by_type[other1_level]
        other2  = total_counts_by_type[other2_level]
        
        #compare current key against other two keys
        #changed for v8.1
        total_counts = current #obs X possible if the level is defined (same as non-unique total counts)
        
        #put in dict
        cloud_counts_by_type[alg] = cloud_counts
        total_counts_by_type[alg] = total_counts
        
        
    #assign everything as a new data variable back to dict
    #vars of format cloud_counts_high, cloud_counts_any, etc...
    cloud_counts_by_type = {f'cloud_counts_{alg}':ds_i for alg,ds_i in cloud_counts_by_type.items()}
    ds = ds.assign(cloud_counts_by_type)
    total_counts_by_type = {f'total_counts_{alg}':ds_i for alg,ds_i in total_counts_by_type.items()}
    ds = ds.assign(total_counts_by_type)
    
    #attenuated_lidar_counts anywhere in column (still useful)
    #v8.1: put this in a try/except so it doesn't break when attenuate_lidar==False
    try:
        ds['attenuated_lidar_counts'] = ds['attenuated_lidar_counts'].any(dim='bin')
    except KeyError:
        print('apply_cloud_defs sees no attenuated_lidar_counts field')
    
    #remove all unused (profile,bin) variables
    ds = ds.drop_dims('bin')
    
    #remove

    return ds

def apply_all_cloud_defs_variable_heights(ds,masks):
    '''Calculate cloud counts and total counts by type, with variable height thesholds'''
    #680 mb: 3238.1 masl
    #440 mb: 6503 masl
    #3238.1 m: bin 91 (2997 to 3237 m) (yes I know it's one lower it's right on the boundary)
    #6503 m: bin 77 (6355 to 6595 m)
    
    # takes counts dataset and applies a cloud definition to classify
    # whether a profile is clear or cloudy.
    
    #takes counts dataset with dims {'start_time','profile','bin'},
    # applied a definition of a cloudy profile given by 'alg',
    # and returns a dataset with dims {'start_time','profile'}
    
    #ds: binary cloud_counts,total_counts with dims {'start_time','profile','bin'}
    #alg: definition of a cloudy profile
    #bounds: bin thresholds for high/middle/low clouds
    
    #helper function for at least 2 contiguous bins
    min2 = lambda cm: cm.shift({'bin':1},fill_value=False) & cm

    #store each counts of type in a dict
    cloud_counts_by_type = { }
    total_counts_by_type = { }
    
    
    #apply different definitions of a cloudy profile
    #minimum N contiguous cloudy bins in a profile
    #just process cloud counts
    for alg in ('any','thick'):
        #definition of thick clouds
        if alg=='thick': alg_loc='min20' #20x240m=4.8km
        if alg=='any': alg_loc='min1' #240 m
        cloud_counts = ds.cloud_counts
        N=int(alg_loc[3:])
        if N==1:
            cloud_counts_by_type[alg] = cloud_counts
        elif N==2:
            cloud_counts_by_type[alg] = min2(cloud_counts)
        elif N>2:
            #bfill method shown to be quickest in 8-3-3 def minN_bfill2
            #requires the package 'bottleneck' to be installed
            fnan = xr.where(cloud_counts==True,np.nan,0) #replace True with NaN
            fill = fnan.bfill(dim='bin',limit=N-1) #fill up to N-1 nans with 0
            cloud_counts_by_type[alg] = xr.where(np.isnan(fill),True,False) #replace NaN with True
        
        total_counts_by_type[alg] = ds.total_counts #for any, thick, say if >=1 bin in profile is defined then it's a valid profile
        
    
    #NB: low cloud cover fraction is a matter of some subtlety, since
    # there are fewer low cloud counts than middle or high. So the issue
    # is whether or not total_counts should be the same or different for 
    # different vertical categories. And when calculating the unique
    # cloud cover fractions, which version of the total counts to use
    # at all. I think it would be best for total_counts to vary by type
    # and just include some duplicates e.g. for any/thick. For the unique
    # clouds, I think the correct definitions are for e.g. unique low
    #   * cloud_counts: low and not (high or mid)
    #   * total_counts: low and high and mid
    # but then this means profiles with thick clouds would be excluded
    # from the fraction if the low bins were not observed.
    
    #non-unique vertical definitions
    #process cloud_counts and total_counts
    for alg,level_mask in masks.items():
        #process cloud_counts
        da = ds.cloud_counts
        cloud_counts_by_type[alg] = da & level_mask

        #process total_counts
        da = ds.total_counts
        total_counts_by_type[alg] = da & level_mask
        
    
    #check if any bin in profile satisfies
    #this is needed for the unique calculations
    cloud_counts_by_type = {alg:cc.any(dim='bin') for alg,cc in cloud_counts_by_type.items()}
    total_counts_by_type = {alg:cc.any(dim='bin') for alg,cc in total_counts_by_type.items()}
    
    #unique vertical definitions
    # (note we must return within the case since
    #  we must fully calculate all three options)
    for alg in ('uniquehigh','uniquemiddle','uniquelow'):        
        #get names of current and other two levels
        levels = ['high','middle','low']
        current_level = alg[6:] #e.g. high, middle, low
        other1_level, other2_level = list(set(levels)-set([current_level])) #remove current level
        
        #retrieve cloud counts variables (T/F by profile)
        current = cloud_counts_by_type[current_level]
        other1  = cloud_counts_by_type[other1_level]
        other2  = cloud_counts_by_type[other2_level]
        
        #compare current key against other two keys
        cloud_counts = current & ~(other1 | other2) #unique X if no Y or Z
        
        #retrieve total counts variables
        current = total_counts_by_type[current_level]
        other1  = total_counts_by_type[other1_level]
        other2  = total_counts_by_type[other2_level]
        
        #compare current key against other two keys
        #changed for v8.1
        total_counts = current #obs X possible if the level is defined (same as non-unique total counts)
        
        #put in dict
        cloud_counts_by_type[alg] = cloud_counts
        total_counts_by_type[alg] = total_counts
        
        
    #assign everything as a new data variable back to dict
    #vars of format cloud_counts_high, cloud_counts_any, etc...
    cloud_counts_by_type = {f'cloud_counts_{alg}':ds_i for alg,ds_i in cloud_counts_by_type.items()}
    ds = ds.assign(cloud_counts_by_type)
    total_counts_by_type = {f'total_counts_{alg}':ds_i for alg,ds_i in total_counts_by_type.items()}
    ds = ds.assign(total_counts_by_type)
    
    #attenuated_lidar_counts anywhere in column (still useful)
    #v8.1: put this in a try/except so it doesn't break when attenuate_lidar==False
    try:
        ds['attenuated_lidar_counts'] = ds['attenuated_lidar_counts'].any(dim='bin')
    except KeyError:
        print('apply_cloud_defs sees no attenuated_lidar_counts field')
    
    #remove all unused (profile,bin) variables
    ds = ds.drop_dims('bin')
    
    #remove

    return ds

def broadcast_cloud_level_thresholds_to_binary_masks(cloud_bounds,da_counts):
    '''Turn heights of ISCCP pressure thresholds into boolean masks defined at each bin'''

    #make a DataArray shaped like cloud counts where each value is the bin number
    mask = np.zeros_like(da,dtype=np.int16)    #shape of start_time, profile, bin
    mask[:,:] = np.arange(125)                 #number of bins
    da_mask = xr.DataArray(data=mask,          #DataArray correctly labels dimensions,
                           dims=da.dims,       #so broadcasting will work out right.
                           coords=da.coords) 
    
    #make a boolean mask for each (start_time, profile, bin) which states whether
    #the bin is above or below the threshold specified at each (start_time,profile)
    #compare (start_time, profile) BIN_THRESH to (start_time, profile, bin) BIN_NUM 
    da_mask_lo = da_mask>=cloud_bounds['lo'] #>= 680 mb
    da_mask_hi = da_mask<cloud_bounds['hi']  #<  440 mb
    da_mask_mid = ~da_mask_lo & ~da_mask_hi  #>= 440 mb and < 680 mb
    
    #pack into dictionary
    return {'low':da_mask_lo,'middle':da_mask_mid,'high':da_mask_hi}

### BUILD GRID ROUTINES ###
#NEW METHOD v6
def print_cluster_info(client):
    info=client.scheduler_info()
    n_workers=len(info['workers'])
    n_cores=sum(client.ncores().values())
    n_threads=sum(client.nthreads().values())
    memory=sum([w['memory_limit'] for w in info['workers'].values()])/1073741824 #bytes per Gibibyte
    print(f'cluster has {n_cores} cores with {n_workers} workers\
, {n_threads//n_workers} threads per worker, and {memory/n_workers:.2f} GiB memory/worker')
    
#7.1: added 'heights' and 'dtype=' to call signature to pass to apply_reshape
#7.2 added 'uniq' kwarg to separate aggregation procedures between:
#       -- getting the number of unique values in a group (grid cell), and
#       -- summing all values in a group (grid cell)
#7.2: added more print statements, del statements
def do_gridding(dstcm,grid_spacing,heights,dtype=int):
    ### DROP DOOP_FLAG VARIABLE ###
    dstcm = dstcm.drop_vars('doop_flag')
    ### SEPARATE UNIQUE VS SUM VARS ###
    uniq=['overpass','dofy']
    sums=set(dstcm.data_vars)-set(uniq)
    u_dstcm = dstcm[uniq]
    s_dstcm = dstcm[sums]
    ### REDUCE DIMS FROM 'START_TIME,PROFILE' to 'GRIDBOX_NUM' ###
    #u_dstcm = u_dstcm.compute() #added v7.2-nbmemtest1
    print('groupby unique...',end='')
    gbs = u_dstcm.groupby('Gridbox_num')
    out = gbs.reduce(lambda arr,axis=None,dim=None: len(np.unique(arr)))
    del u_dstcm, gbs
    print('done.')
    print('groupby sum...',end='')
    print(f'dataset to group size: {s_dstcm.nbytes/1e6:.2f} MB')
    #if s_dstcm.nbytes/1e6>2500:
    gbs = s_dstcm.groupby('Gridbox_num')
    out2 = gbs.sum(dim='stacked_start_time_profile')
    del s_dstcm, gbs
    print('done.')
    # reform
    print('merge out,out2...',end='')
    out2 = xr.merge([out,out2])
    del out
    print('done.')

    ### RESHAPE FROM 1D GRIDBOX_NUM TO 2D LON/LAT ###
    print('computing output...',end='')
    out2 = out2.compute()
    print('done')
    print('reshaping output...',end='')
    gridded = apply_reshape(out2,heights,dx=grid_spacing,dtype=dtype)
    del out2
    print('done')
    return gridded

def ds_ll2gbnum(ds,dx=2):
    #lon/lat to gridbox number for dataset.map
    gbattrs = {'description':'gridbox number counting from -180,-90 by longitude (col) then latitude (row)','boxsize':dx}
    #lat/lon format
    xmin,xmax = -180,180
    ymin,ymax = -90,90
    #calc dims of bin grid
    ncols = (xmax-xmin)//dx
    nrows = (ymax-ymin)//dx
    #convert from bin edges to indices
    #(origin lower left)
    i = (ds.Latitude-ymin)//dx #row
    j = (ds.Longitude-xmin)//dx #col
    #encode i,j as a scalar
    k = i*ncols+j
    #k = k.astype(int)
    gb = xr.DataArray(k,attrs=gbattrs,dims=('start_time','profile'))
    return ds.assign_coords(Gridbox_num=gb)

def gbnum2ij(k,dx=2):
    ncols = 360//dx
    j = int(k%ncols)
    i = int(k//ncols)
    return i,j

#v7.2 added more print and del statements
#v7.1 accept shape+dtype dict as an argument
def reshape_irregular(da,desired_shape,dx=2,dtype=int):
    print('reshape irregular: in...',end='')
    #create blank arrays of the proper sizes
    #i,j,k == latitude, longitude, bin (origin lower)
    reshaped = np.zeros(desired_shape,dtype=dtype) 
    #reshaped[:,:] = np.nan #NOTE: changed for v5.1!!

    for i in range(da.shape[0]):
        #print('{:.2f}% '.format(100*i/cc.shape[0]),end='')
        gridbox = da.isel(Gridbox_num=i)
        ij = gbnum2ij(gridbox.Gridbox_num,dx=dx)
        reshaped[ij] = gridbox.data
    del gridbox
    print('out.')
    return reshaped

#v7.2 added more print and del statements
#v7.1 handle more shapes+dtypes
#relies on parent scope vars rmcp_heights
def apply_reshape(ds,heights,dx=2,dtype=int):
    print('apply reshape: in...',end='')
    ds_dict = {}
    xmin, xmax = -180, 180
    ymin, ymax = -90, 90
    lon1d = np.arange(xmin, xmax, dx).astype(float) 
    lat1d = np.arange(ymin, ymax, dx).astype(float)
    lat1d, lon1d = lat1d+dx/2, lon1d+dx/2 #NOTE: changed for v5.1!!

    for key in ds.keys():
        #handle both lat/lon/height and lat/lon vars
        if len(ds[key].dims) == 2: #vertically-resolved key
            kind = 'height'
            coords = {'lat':lat1d,'lon':lon1d,'height':heights}
            desired_shape = (len(lat1d),len(lon1d),len(heights))
        elif len(ds[key].dims) == 1: #column-integrated key
            kind = 'cover'
            coords = {'lat':lat1d,'lon':lon1d}
            desired_shape = (len(lat1d),len(lon1d))
    
        data = reshape_irregular(ds[key],desired_shape,dx=dx,dtype=dtype)
        ds_dict[key] = xr.DataArray(data,name=key,
                                               dims=tuple(coords.keys()),
                                               #attrs={'desc':descriptions[key]},
                                               coords=coords
                                              )
    del coords, data
    print('out.')
    return xr.Dataset(ds_dict)

### DOOP-PROCESSING ROUTINES ###
def save_spline(tck_fi,tck_la,filename='doop_spldict.json'):
    '''converts two scipy.interpolate tck tuples 
    (from splprep) to a nested dict and saves json to disk.
    For saving fit doop first profile, doop last profile curves '''
    spldict = {'first_doop_profile':
        {'knot_pos':list(tck_fi[0]),'bspl_coeff':list(tck_fi[1]),'deg':tck_fi[2]},
         'last_doop_profile':
         {'knot_pos':list(tck_la[0]),'bspl_coeff':list(tck_la[1]),'deg':tck_la[2]}
        }
    with open(filename, 'w') as f:
        f.write(json.dumps(spldict)) 
    print(f'saved file {filename}!')
    return

def read_spline(filename='doop_spldict.json',keys=('knot_pos','bspl_coeff','deg')):
    '''read json dict of t,c,k values back to tck tuples
    for use in scipy.interpolate.splXXX routines.
    Note: routine assumes dict has two splines:
    'first_doop_profile' and 'last_doop_profile'
    '''
    #read json dict of structure {spline1:{t:,c:,k:},spline2:{t:,c:,k:}
    #convert to {spline1:tck,spline2:tck} tuple of arrays
    with open(filename) as f:
        rspl = json.loads(f.read())
    #convert the 2nd level of keys (knotpos, coeff, deg) to tuple of arrays (tck)
    rspl = dict([(d,[np.asarray(rspl[d][key]) for key in keys]) for d in rspl])
    #note that splder CHANGES spline degree if k is of type np.array and not int
    #so we have to manually change that
    for d in rspl:
        rspl[d][2] = int(rspl[d][2])
    return rspl['first_doop_profile'],rspl['last_doop_profile']

#overrides cloudsat_util_06.py (which has defs for build_cloudcover_grid_v5.py)
def get_doop_flag(ds,filename='doop_spldict.json',anomaly_start=np.datetime64('2011-04-17')):
    #read splines for first,last doop profile
    tck_fi,tck_la = read_spline(filename=filename)
    
    #get day of year float start_time coordinate
    dofy = ds.start_time.dt.strftime('%j').data.astype(float)
    h = ds.start_time.dt.strftime('%H').data.astype(int)
    m = ds.start_time.dt.strftime('%M').data.astype(int)
    s = ds.start_time.dt.strftime('%S').data.astype(int)
    f = (60*(60*h+m)+s)/(24*60*60) # fraction of the day (0-1)
    t = dofy+f #float time in day of year
    
    #get ascending/descending info from first/last curve
    ##retrieve precise minima for ascending branch
    critpts = interpolate.sproot(interpolate.splder(tck_fi))
    asc_start = critpts[np.argmin(np.abs(critpts-62))]
    asc_end   = critpts[np.argmin(np.abs(critpts-280))]
    asc_fi = (t>=asc_start) & (t<=asc_end)
    asc_la = False
    
    ##bit 14 for asc/desc, cit. 1B-GEOPROF PDICD
    #asc_flag = (ds.Data_status.data >> 14 & 1)
    #fill nans to 0, cast to int to use bit-shifting
    #(neccesary if using read_GEOPROF nprofiles kwarg
    asc_flag = (ds.Data_status.fillna(0).data.astype(int) >> 14 & 1)
    
    #get first/last latitude
    lat_fi = interpolate.splev(t,tck_fi)
    lat_la = interpolate.splev(t,tck_la)
    
    #get index of first/last do-op profile
    ##exclude the ascending/descending branch from the nearest-neighbor
    ##by setting it equal to -999. Then use argmin to find the closest
    ##index in the desired branch to the chosen latitude
    ##fillna latitude is required since argmin chooses nan over not-nan
    idx_fi = np.argmin(np.abs(xr.where(asc_flag==asc_fi,ds.Latitude.fillna(-999).values,-999)-lat_fi))
    idx_la = np.argmin(np.abs(xr.where(asc_flag==asc_la,ds.Latitude.fillna(-999).values,-999)-lat_la))
    
    #build into dataArray
    ii = np.arange(ds.dims['profile'])
    
    if ds.start_time.data>=anomaly_start: #NOTE: changed for v5.1!!
        #after battery anomaly doop = 2
        doop_flag = np.zeros((ds.dims['profile']))+1 #NOTE: changed for v5.1!!
    else:
        #before battery anomaly doop = 0 or 1
        doop_flag = (idx_fi<=ii) & (ii<=idx_la)
    da = xr.DataArray(doop_flag,name='doop_flag',dims=('profile'))
    return da


#### VERSION VERSION VERSION ####
#### VERSION VERSION VERSION ####
#### VERSION VERSION VERSION ####
#### VERSION VERSION VERSION ####
#### VERSION VERSION VERSION ####
#### VERSION VERSION VERSION ####
#### VERSION VERSION VERSION ####






