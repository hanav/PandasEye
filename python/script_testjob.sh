#!/bin/bash -l
# created: Mar 26, 2018 1:28 PM
# author: hanav
#SBATCH -J testjob
#SBATCH --constraint="snb|hsw"
#SBATCH -o output_%j.tx
#SBATCH -e errors_%j.txt
#SBATCH -p test
#SBATCH -n 3
#SBATCH -t 00:30:00
#SBATCH --mem-per-cpu=64000
#SBATCH --mail-type=END
#SBATCH --mail-user=hanav@uef.fi

# commands to manage the batch script
#   submission command
#     sbatch [script-file]
#   status command
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

python $USERAPPL/project_hoy/python/taito_SVM.py -i $WRKDIR/project_hoy/all.csv -o $WRKDIR/project_hoy/results -m "Test,select features, upsample"

# This script will print some usage statistics to the
# end of file: std.out
# Use that to improve your resource request estimate
# on later jobs.
# used_slurm_resources.bash

# Upload scripts:
# scp -rq /Users/icce/Dropbox/_thesis_framework/_scripts_hoy/r_icmi/python hanav@taito.csc.fi:/homeappl/home/hanav/appl_taito/project_hoy/

# Upload data:
# scp -rq /Users/icce/Dropbox/_thesis_framework/_dataset_8Puzzles/processed/2015_umap_shifted_2/qe_-5_0/all.csv hanav@taito.csc.fi:/wrk/hanav/project_hoy/