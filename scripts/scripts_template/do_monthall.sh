#! /bin/bash

# iterate over full data record and submit a job for each 3S-GEOPROF-COMB
# output file with a given set of params (e.g., grid size, intruments, etc.)
currdir=$(pwd)
while read p; do
	month=$(echo $p|awk -F "," '{print $1}')
	year=$(echo $p|awk -F "," '{print $2}')
	suffix=$(echo $p|awk -F "," '{print $3}')
	echo "sbatch --export=m=$month,y=$year,f=$suffix,c=$currdir --job-name=$year --output=${currdir}/m${year}${month}.out ./do_month.sbatch"
 	sbatch --export=m=$month,y=$year,f=$suffix,c=$currdir --job-name=$year --output=${currdir}/m${year}${month}.out ./do_month.sbatch
done <monthYearSuffixNoHead-to2020.csv
