set -e
source activate joeenv;

python t_main_accuracy_clean.py --show=0;
python t_main_accuracy_clean.py --show=0 --dba=1 --n_malicious=2;
python t_main_accuracy_clean.py --show=0 --dba=1 --n_malicious=4;
python t_main_accuracy_clean.py --show=0 --neuro=1;
python t_main_accuracy_clean.py --show=0 --dba=1 --n_malicious=2 --neuro=1;
python t_main_accuracy_clean.py --show=0 --dba=1 --n_malicious=4 --neuro=1;

python t_main_accuracy_pois.py --show=0;
python t_main_accuracy_pois.py --show=0 --dba=1 --n_malicious=2;
python t_main_accuracy_pois.py --show=0 --dba=1 --n_malicious=4;
python t_main_accuracy_pois.py --show=0 --neuro=1;
python t_main_accuracy_pois.py --show=0 --dba=1 --n_malicious=2 --neuro=1;
python t_main_accuracy_pois.py --show=0 --dba=1 --n_malicious=4 --neuro=1;
