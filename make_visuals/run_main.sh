set -e
source activate joeenv;

python main_accuracy.py --show=0;
python main_accuracy.py --show=0 --dba=1 --n_malicious=2;
python main_accuracy.py --show=0 --dba=1 --n_malicious=4;
python main_accuracy.py --show=0 --neuro=1;
python main_accuracy.py --show=0 --dba=1 --n_malicious=2 --neuro=1;
python main_accuracy.py --show=0 --dba=1 --n_malicious=4 --neuro=1;
