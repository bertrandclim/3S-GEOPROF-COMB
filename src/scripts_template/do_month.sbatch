#!/bin/bash

#SBATCH --account=ucb313_asc1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --time=01:00:00
#SBATCH --output=wk00.out
#SBATCH --job-name=wk00
#SBATCH --mem=240G

#increased resources to 64 cores @ 3.75G ram/core 
#    extra increase from 128 to see if job failures come from multiple LocalClusters running in tandem on the same node
#    did this since I get oom-killer job failures at 64G on alpine when doing groupby-sum all (but oddly not on summit iirc)
#removed SLURM_SCRATCH since this variable is buggy on alpine
#    i.e. jobs will fail because they get permission denied when trying to access their local $SLURM_SCRATCH
#added partition flag to make it explicit that it's supposed to go to alpine
echo SCRIPT RUNNING
echo SHELL $SHELL
echo "y=$y"
echo "m=$m"
echo "f=$f"
echo "c=$c"
module purge
module load anaconda
#conda init bash
conda activate sci2 

cd $c 
echo 'python -u ../9-7_build_coverandheight_grid_v8.3.py -s --month=$m --year=$y -f $f -g 2.5 --def=all --localdir=${SLURM_SCRATCH} --outdir=$(pwd)/out/ --nd --nb' >> m${y}${m}.out 2>&1
python -u ../9-7_build_coverandheight_grid_v8.3.py -s --month=$m --year=$y -f $f -g 2.5 --def=all --localdir=${SLURM_SCRATCH} --outdir=$(pwd)/out/ --nd --nb >> m${y}${m}.out 2>&1
