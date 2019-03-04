#!/bin/bash
#
#SBATCH --partition=1080ti-long    # Partition to submit to <m40-short|m40-long|teslax-short|teslax-long>
#SBATCH --job-name=pica-net
#SBATCH --gres=gpu:1
#SBATCH -o run_logs/pica_net_%j.txt            # output file
#SBATCH -e run_logs/pica_net_%j.err            # File to which STDERR will be written
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00          # D-HH:MM:SS
#SBATCH --mem=32768
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dchakraborty@cs.umass.edu

set -x
set -e

#module unload cuda80/toolkit/8.0.61 cudnn/6.0 cudnn/7.0-cuda_9.0 
#module load cudnn/7.0


python3 image_test.py \
 --model_dir /mnt/nfs/scratch1/dchakraborty/pica/models/state_dict/03030334/56epo_399881step.ckpt \
 --dataset /mnt/nfs/scratch1/dchakraborty/KAIST_SALIENCY_SUBSET/images \
 --batch_size 4 \
 --save_dir /mnt/nfs/scratch1/dchakraborty/pica/results/

sleep 1
exit
