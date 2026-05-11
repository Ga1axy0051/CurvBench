```bash
# 1. Install dependencies
conda create -n hat --file requirements.txt
conda activate hat

# 2. Run all 14 datasets
bash hat.sh

# 3. Check outputs
ls log/
ls runs/