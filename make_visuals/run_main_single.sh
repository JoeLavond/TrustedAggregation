set -e
source activate tag;

python main_accuracy_single.py --show=0 --n_classes=100 --d_scale=2.0 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --n_classes=100 --d_scale=2.0 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --n_classes=100 --d_scale=2.0 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --neuro=1 --n_classes=100 --d_scale=2.0 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --neuro=1 --n_classes=100 --d_scale=2.0 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --neuro=1 --n_classes=100 --d_scale=2.0 --alpha=1 --only=1;

python main_accuracy_single.py --show=0 --n_classes=100 --d_scale=1.5 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --n_classes=100 --d_scale=1.5 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --n_classes=100 --d_scale=1.5 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --neuro=1 --n_classes=100 --d_scale=1.5 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --neuro=1 --n_classes=100 --d_scale=1.5 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --neuro=1 --n_classes=100 --d_scale=1.5 --alpha=1 --only=1;

python main_accuracy_single.py --show=0 --n_classes=100 --d_scale=1.25 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --n_classes=100 --d_scale=1.25 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --n_classes=100 --d_scale=1.25 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --neuro=1 --n_classes=100 --d_scale=1.25 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --neuro=1 --n_classes=100 --d_scale=1.25 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --neuro=1 --n_classes=100 --d_scale=1.25 --alpha=1 --only=1;

python main_accuracy_single.py --show=0 --n_classes=100 --d_scale=1.1 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --n_classes=100 --d_scale=1.1 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --n_classes=100 --d_scale=1.1 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --neuro=1 --n_classes=100 --d_scale=1.1 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --neuro=1 --n_classes=100 --d_scale=1.1 --alpha=1 --only=1;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --neuro=1 --n_classes=100 --d_scale=1.1 --alpha=1 --only=1;

