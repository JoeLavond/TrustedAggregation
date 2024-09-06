set -e
source activate tag;

python main_accuracy_single.py --show=0 --alpha=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --alpha=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --alpha=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --neuro=1 --alpha=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --neuro=1 --alpha=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --neuro=1 --alpha=1 --alpha_val=1;
