source activate tag;

python ext_adapt.py --show=0 --mu=0.01 --dba=0 --n_malicious=1 --neuro=0 &
python ext_adapt.py --show=0 --mu=0.01 --dba=1 --n_malicious=2 --neuro=0 &
python ext_adapt.py --show=0 --mu=0.01 --dba=1 --n_malicious=4 --neuro=0;

python ext_adapt.py --show=0 --mu=0.01 --dba=0 --n_malicious=1 --neuro=1 &
python ext_adapt.py --show=0 --mu=0.01 --dba=1 --n_malicious=2 --neuro=1 &
python ext_adapt.py --show=0 --mu=0.01 --dba=1 --n_malicious=4 --neuro=1;
echo 'mu 0.01 done';


python ext_adapt.py --show=0 --mu=0.005 --dba=0 --n_malicious=1 --neuro=0 &
python ext_adapt.py --show=0 --mu=0.005 --dba=1 --n_malicious=2 --neuro=0 &
python ext_adapt.py --show=0 --mu=0.005 --dba=1 --n_malicious=4 --neuro=0;

python ext_adapt.py --show=0 --mu=0.005 --dba=0 --n_malicious=1 --neuro=1 &
python ext_adapt.py --show=0 --mu=0.005 --dba=1 --n_malicious=2 --neuro=1 &
python ext_adapt.py --show=0 --mu=0.005 --dba=1 --n_malicious=4 --neuro=1;
echo 'mu 0.005 done';


python ext_adapt.py --show=0 --mu=0.001 --dba=0 --n_malicious=1 --neuro=0 &
python ext_adapt.py --show=0 --mu=0.001 --dba=1 --n_malicious=2 --neuro=0 &
python ext_adapt.py --show=0 --mu=0.001 --dba=1 --n_malicious=4 --neuro=0;

python ext_adapt.py --show=0 --mu=0.001 --dba=0 --n_malicious=1 --neuro=1 &
python ext_adapt.py --show=0 --mu=0.001 --dba=1 --n_malicious=2 --neuro=1 &
python ext_adapt.py --show=0 --mu=0.001 --dba=1 --n_malicious=4 --neuro=1;
echo 'mu 0.001 done';
