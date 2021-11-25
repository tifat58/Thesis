# Thesis

# create an new virtual environment and install packages using requirements.txt
pip install -r requirements.txt

update ./tools/train.sh file to define data directory
# run command:
CUDA_VISIBLE_DEVICES=0 bash ./tools/train.sh
