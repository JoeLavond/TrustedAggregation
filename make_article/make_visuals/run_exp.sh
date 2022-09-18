set -e
source activate joeenv;

python exp_accuracy_v2.py --show=0;
python exp_accuracy_v2.py --show=0 --dba=1 --n_malicious=2;
python exp_accuracy_v2.py --show=0 --dba=1 --n_malicious=4;

python exp_accuracy_v2.py --show=0 --neuro=1;
python exp_accuracy_v2.py --show=0 --dba=1 --n_malicious=2 --neuro=1;
python exp_accuracy_v2.py --show=0 --dba=1 --n_malicious=4 --neuro=1;
