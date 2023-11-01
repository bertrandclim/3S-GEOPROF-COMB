#! /bin/bash
module load slurm/alpine #submits to alpine instead of summit
currdir=$(pwd)
while read p; do
	month=$(echo $p|awk -F "," '{print $1}')
	year=$(echo $p|awk -F "," '{print $2}')
	suffix=$(echo $p|awk -F "," '{print $3}')
	echo "sbatch --export=m=$month,y=$year,f=$suffix,c=$currdir --job-name=$year --output=${currdir}/m${year}${month}.out ./do_month.sbatch"
 	sbatch --export=m=$month,y=$year,f=$suffix,c=$currdir --job-name=$year --output=${currdir}/m${year}${month}.out ./do_month.sbatch
done <monthYearSuffixNoHead-to2020.csv

#for k in {1..12}
#do
#	k=$(printf "%02d" ${k})
#	echo "sbatch --export=k=$k,f=m$k --job-name=m$k --output=/projects/wibe4964/cloudsat_work/month2010_v6.1/m${k}.out ./do_month2010.sbatch"
#	sbatch --export=k=$k,f=m$k --job-name=m$k --output=/projects/wibe4964/cloudsat_work/month2010_v6.1/m${k}.out ./do_month2010.sbatch
#	#sbatch --export=i=$i,j=$j,f=wk$k --job-name=wk$k --output=/projects/wibe4964/week2010/wk${k}.out ./do_week2010.sbatch
#done
