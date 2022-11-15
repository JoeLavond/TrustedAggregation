set -e
source activate joeenv;

# python main_accuracy_single.py --show=0 --resnet=0 --beta=0.1;
# python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --resnet=0 --beta=0.1;
# python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --resnet=0 --beta=0.1;
python main_accuracy_single.py --show=0 --resnet=0 --beta=0.1 --neuro=1 --d_scale=2.5;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=2 --resnet=0 --beta=0.1 --neuro=1 --d_scale=2.5;
python main_accuracy_single.py --show=0 --dba=1 --n_malicious=4 --resnet=0 --beta=0.1 --neuro=1 --d_scale=2.5;
