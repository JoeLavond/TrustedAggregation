#set -e
source activate joeenv;

python main_accuracy_single.py --show=0 --alpha=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --alpha=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --alpha=1;
python main_accuracy_single.py --show=0 --alpha=1 --neuro=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --alpha=1 --neuro=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --alpha=1 --neuro=1;

python main_accuracy_single.py --show=0 --alpha=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --alpha=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --alpha=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --alpha=1 --alpha_val=1 --d_scale=1.5;
python main_accuracy_single.py --show=0 --alpha=1 --neuro=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --alpha=1 --neuro=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --alpha=1 --neuro=1 --alpha_val=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --alpha=1 --alpha_val=1 --d_scale=1.25;

