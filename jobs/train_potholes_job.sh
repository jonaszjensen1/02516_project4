#!/bin/sh

### --- 1. EDIT THESE LINES FOR EACH NEW EXPERIMENT ---
### Set the folder name manually here. Variables ($EXP_NAME) do NOT work in #BSUB lines.
#BSUB -J Pothole_CNN_Baseline
#BSUB -o project4/experiments/cnn_baseline_v1/output_%J.out
#BSUB -e project4/experiments/cnn_baseline_v1/error_%J.err

### --- BSUB Standard Options ---
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=10GB]"
#BSUB -W 04:00

### --- Script Logic ---
# Define the same name here so Python knows where to save the model
EXP_NAME="cnn_baseline_v1"
OUTPUT_DIR="project4/experiments/$EXP_NAME"

echo "Running experiment: $EXP_NAME"
echo "Job ID: $LSB_JOBID"

# Load environment
module load python3/3.13.5
module load cuda/11.6
source dl_env/bin/activate

# Add src to PYTHONPATH so imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)/project4/src

# Run Training
# Note: We pass the created OUTPUT_DIR to python so it saves model/plots there
python project4/src/train_potholes.py \
    --data_path project4/src/pothole_proposals_train.pt \
    --output_dir $OUTPUT_DIR \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001

echo "Job finished."