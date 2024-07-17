# 3S-GEOPROF-COMB production source code
Codebase used to produce the 3S-GEOPROF-COMB data product on the RMACC Alpine computing cluster. 

Of general interest may be the `read_GEOPROF` method in `cloudsat_util_10.py`, which reads a 2B-GEOPROF or 2B-GEOPROF-LIDAR granule (in HDF4-EOS format) into an xarray Dataset for easy integration with other Python libraries.

## Getting started

### Setup
1. Download repository
```bash
git clone https://github.com/bertrandclim/3S-GEOPROF-COMB
cd ./3S-GEOPROF-COMB/production
```
2. Create environment
```bash
conda env create -f environment.yml
conda activate cscalprod
```
3. Print out script help
```bash
python build_coverandheight_grid_v8.4.py -h
```
4. Download some 2B-GEOPROF and 2B-GEOPROF-LIDAR granules from the [CloudSat Data Processing Center](https://www.cloudsat.cira.colostate.edu/).
5. Run script to compute statistics of simultaneous 2B-GEOPROF and 2B-GEOPROF-LIDAR granules on a 10°x10° grid
```bash
`python build_coverandheight_grid_v8.4.py sub -g 10 -r /path/to/2B-GEOPROF -l /path/to/2B-GEOPROF-LIDAR` -nworkers 8
```
> If running on a local machine, I would not reccomend processing more than several hundred granules at a time, though much less than this will do for simple testing.

Many other settings for the product are possible, as indicated by `python build_coverandheight_grid_v8.4.py full -h` and `python build_coverandheight_grid_v8.4.py sub -h`.

## Contents:
* `build_coverandheight_grid_v8.4.py`: script for computing data product output for one month and one set of options
* `build_seasonal_from_monthly_v8.4.py`: script for computing seasonal output from a series of monthly outputs
* `cloudsat_util_10.py`: library of methods useful for working with the data
* __scripts_template__: submission scripts used for production of data product
  * `do_monthall.sh`: bash script to compute all months with options (e.g. resolution) specified by `do_month.sbatch`
  * `do_month.sbatch`: SLURM submission script to run `build_coverandheight_grid_v8.4.py` once
  * `monthYearSuffixNoHead-to2020.csv`: specifies which months to compute
* __resources__: reference data files used to create product
  * `height_bin_at_pressure_level.nc`: specifies zonal-mean 440 and 680 mb heights from NCEP reanalysis
  * `doop_spldict.json`: specifies DO-Op mode collection patterns