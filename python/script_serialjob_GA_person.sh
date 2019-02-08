#!/bin/bash -l
# created: Mar 26, 2018 1:28 PM
# author: hanav
#SBATCH -J testjob
#SBATCH --constraint="snb|hsw"
#SBATCH -o msgs/person_ga_output_%j.txt
#SBATCH -e msgs/person_ga_errors_%j.txt
#SBATCH -p serial
#SBATCH -n 3
#SBATCH -t 06:30:00
#SBATCH --mem-per-cpu=500
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

module load  biopython-env

python $USERAPPL/project_hoy/python/taito_SVM_person.py -i $WRKDIR/project_hoy/output_GA_4_1_90features.csv -o $WRKDIR/project_hoy/results -m "GA-4-1-PersonCV: Person CrossVal, Person split, Person upsample" -d "GA-4-1"

# This script will print some usage statistics to the
# end of file: std.out
# Use that to improve your resource request estimate
# on later jobs.
# used_slurm_resources.bash

# Upload scripts:
# scp -rq /Users/icce/Dropbox/_thesis_framework/_scripts_hoy/r_icmi/python hanav@taito.csc.fi:/homeappl/home/hanav/appl_taito/project_hoy/

# Upload data:
# scp -rq /Users/icce/Dropbox/_thesis_framework/_dataset_8Puzzles/processed/2015_umap_shifted_2/qe_-5_0/all.csv hanav@taito.csc.fi:/wrk/hanav/project_hoy/