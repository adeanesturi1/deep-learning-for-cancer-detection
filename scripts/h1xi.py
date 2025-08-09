#!/bin/bash
#
#SBATCH --job-name=nnunet_multi
#SBATCH --partition=gpu                
#SBATCH --gres=gpu:1                   
#SBATCH --cpus-per-task=8              
#SBATCH --mem=64G                      
#SBATCH --time=2-00:00:00              
#SBATCH --output=/sharedscratch/an252/cancerdetectiondataset/logs/nnunet_multi_%j.out
#SBATCH --error=/sharedscratch/an252/cancerdetectiondataset/logs/nnunet_multi_%j.err

# load your environment
module load anaconda3                 
source activate brats-env             

# point nnU-Net to your data
export nnUNet_raw_data_base=/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw
export nnUNet_preprocessed=/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed
export RESULTS_FOLDER=/sharedscratch/an252/cancerdetectiondataset/nnUNet_results

# run training on fold 3 with your new multitask trainer
nnUNetv2_train 002 3d_fullres 3 nnUNetTrainer_MultiTaskSimple
