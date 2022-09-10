source activate joeenv;

python exp_accuracy_clean.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=0 --n_malicious=1 --data='cifar' --neuro=0;
python exp_accuracy_pois.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=0 --n_malicious=1 --data='cifar' --neuro=0;
python exp_threshold.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=0 --n_malicious=1 --data='cifar' --neuro=0;
echo 'done'

python exp_accuracy_clean.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=2 --data='cifar' --neuro=0;
python exp_accuracy_pois.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=2 --data='cifar' --neuro=0;
python exp_threshold.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=2 --data='cifar' --neuro=0;
echo 'done'

python exp_accuracy_clean.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=4 --data='cifar' --neuro=0;
python exp_accuracy_pois.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=4 --data='cifar' --neuro=0;
python exp_threshold.py --show=1 --n_classes=100 --n_rounds=50 --alpha=10000 --alpha_val=10000 --dba=1 --n_malicious=4 --data='cifar' --neuro=0;
echo 'done'

