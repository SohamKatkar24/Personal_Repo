#!/bin/bash
#SBATCH --job-name=climatePrediction
#SBATCH --partition=high_mem
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --constraint=hpcf2018
#SBATCH --mem=64000
#SBATCH --qos=normal
#SBATCH --output=climate-pred-%j-%u-slurm.out
#SBATCH --error=climate-pred-%j-%u-slurm.out
#SBATCH --time=4:00:00

# Load required modules (if needed)
# module purge
# module load java spark python/3.8

# Run the Spark job via Taki wrapper script
echo "Starting Climate Prediction job on Taki"
/umbc/rs/is789sp25/common/instruction/runSparkOnTaki.sh \
  /umbc/rs/is789sp25/users/skatkar1/Project/climate_analysis_final.py

# End of script
