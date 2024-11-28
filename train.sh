
#!/bin/bash

#SBATCH —-account=rrg-tyrell-ab

#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --gpus=gpu:1                     # Number of GPUs
#SBATCH --mem-per-cpu=10G                    # Memory needed per node
#SBATCH --time=0:15:00                  # Time limit hrs:min:sec


module load scipy-stack/2024b python/3.11.5 opencv/4.10.0
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio

source corhip/bin/activate
cd $HOME/cor-hip


python main.py



