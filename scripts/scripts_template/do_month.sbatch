#!/bin/bash

#SBATCH --account=your_summit_allocation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --time=12:00:00
#SBATCH --mem=240G

#slurm batch script to compute one netCDF file of 3S-GEOPROF-COMB
echo SCRIPT RUNNING
echo SHELL $SHELL
echo "y=$y"
echo "m=$m"
echo "f=$f"
echo "c=$c"
module purge
module load anaconda
conda activate cscalprod

cd $c 
echo 'python -u ../build_coverandheight_grid_v8.4.py full --month=$m --year=$y -g 10 -f $f --localdir=${SLURM_SCRATCH} --outdir=./out --nd @params.txt' >> m${y}${m}.out 2>&1
python -u ../build_coverandheight_grid_v8.4.py full --month=$m --year=$y -g 10 -f $f --localdir=${SLURM_SCRATCH} --outdir=./out --nd @params.txt >> m${y}${m}.out 2>&1
