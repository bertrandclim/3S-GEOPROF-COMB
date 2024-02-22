# 3S-GEOPROF-COMB production source code
Codebase used to produce the 3S-GEOPROF-COMB data product on the RMACC Summit and Alpine computing clusters. 

__Note__: this code is __not__ distributed as a package or library, but is simply shared for archival purposes. Please contact me (leah.bertrand AT colorado.edu) if you are interested in adapting this code to your needs. 

Of general interest may be the `read_GEOPROF` method in `cloudsat_util_09.py`, which reads a 2B-GEOPROF or 2B-GEOPROF-LIDAR granule (in HDF4-EOS format) into an Xarray Dataset for easy integration with other Python libraries.

## Getting started

### Setup
1. Download repository
```bash
git clone https://github.com/bertrandclim/3S-GEOPROF-COMB
```
2. Create environment
```bash
conda env create -f ./3S-GEOPROF-COMB/src/environment.yml
```
3. Print out script help
```bash
cd ./3S-GEOPROF-COMB/src
python 9-7_build_coverandheight_grid_v8.3.py -h
```
4. Download 2B-GEOPROF and 2B-GEOPROF-LIDAR granules from the [CloudSat Data Processing Center](https://www.cloudsat.cira.colostate.edu/).
 
> If running on a local machine, I would not reccomend processing more than several hundred granules at a time, though much less than this will do for simple testing.

5. Run script to compute statistics of simultaneous 2B-GEOPROF and 2B-GEOPROF-LIDAR granules on a 10°x10° grid
```bash
`python 9-7_build_coverandheight_grid_v8.3.py -g 10 -r /path/to/2B-GEOPROF/ -l /path/to/2B-GEOPROF-LIDAR`
```
Many other settings for the product are possible. Options used for the most recent round of production (i.e. recently tested options) include `-g`,`-o`,`-k`,`-f`, `-s`, `--nb`, `--nd`, `--month`, and `--year`. 

## Contents:
* `9-7_build_coverandheight_grid_v8.3.py`: script for computing data product output for one month and one set of options
* `cloudsat_util_09.py`: library of methods useful for working with the data
* __scripts_template__: submission scripts used for production of data product
  * `do_monthall.sh`: bash script to compute all months with a given set of options
  * `do_month.sbatch`: submission script to run `9-7_build_coverandheight_grid_v8.3.py` once
  * `monthYearSuffixNoHead-to2020.csv`: specifies which months to compute
* __resources__: reference data files used to create product
  * `9-5_height_bin_at_pressure_level.nc`: specifies zonal-mean 440 and 680 mb heights from reanalysis
  * `doop_spldict.json`: specifies DO-Op mode collection patterns
