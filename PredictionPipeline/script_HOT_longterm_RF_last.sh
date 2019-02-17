#!/bin/bash -l
# created: Mar 26, 2018 1:28 PM
# author: hanav
#SBATCH -J testjob
#SBATCH --constraint="snb|hsw"
#SBATCH -o msgs/hot_longterm_last_%j.txt
#SBATCH -e msgs/hot_longterm_last_errors_%j.txt
#SBATCH -p serial
#SBATCH -n 6
#SBATCH -t 3:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=END
#SBATCH --mail-user=hanav@uef.fi

# commands to manage the batch script
#   submission command
#     sbatch [script-file]
#   status commandw
#     squeue -u hanav
#   termination command
#     scancel [jobid]

# For more information
#   man sbatch
#   more examples in Taito guide in
#   http://research.csc.fi/taito-user-guide

# run my scripts
# -i is input file with features
# -o is output directory (for now)

module load  biopython-env

python $USERAPPL/project_hoy/python/3_ml_classify_loocv_longterm_RF.py -i $WRKDIR/project_hoy/features_longterm_last5min_stats_omitShort_ValenceArousal_npNan.csv -o $WRKDIR/project_hoy/results -m "HOT valence and arousal: last 5 minutes - RandomGridSearch" -d "HOT_LAST_SMOTE"

# This script will print some usage statistics to the
# end of file: std.out
# Use that to improve your resource request estimate
# on later jobs.
# used_slurm_resources.bash