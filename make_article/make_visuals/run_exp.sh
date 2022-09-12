source activate joeenv;

python exp_accuracy_clean.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=0 --n_malicious=1 --data='stl' --neuro=0;
python exp_accuracy_pois.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=0 --n_malicious=1 --data='stl' --neuro=0;
python exp_threshold.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=0 --n_malicious=1 --data='stl' --neuro=0;
echo 'done'

python exp_accuracy_clean.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=2 --data='stl' --neuro=0;
python exp_accuracy_pois.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=2 --data='stl' --neuro=0;
python exp_threshold.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=2 --data='stl' --neuro=0;
echo 'done'

python exp_accuracy_clean.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=4 --data='stl' --neuro=0;
python exp_accuracy_pois.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=4 --data='stl' --neuro=0;
python exp_threshold.py --show=0 --n_classes=10 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=4 --data='stl' --neuro=0;
echo 'done'

