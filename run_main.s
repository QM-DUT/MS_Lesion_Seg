#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=HRNet
#SBATCH --mail-type=END
#SBATCH --mail-user=qc690@nyu.edu
#SBATCH --output=slurm_%j.out



module purge
module load cuda/10.0.130   
cd /home/qc690/Video/MS_Lesion_Seg
/home/qc690/anaconda3/bin/python train_HR_net3D_patches.py>log_HRNet_3D.txt


