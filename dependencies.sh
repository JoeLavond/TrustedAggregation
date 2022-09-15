# instructions
# ./dependencies.sh will run dependencies.sh in a sub-shell
# . ./dependencies.sh will run dependencies.sh in current shell

# setup
export http_proxy=http://152.2.41.28:3128
export HTTP_PROXY=http://152.2.41.28:3128
export https_proxy=http://152.2.41.28:3128
export HTTPS_PROXY=http://152.2.41.28:3128

# create and activate conda environment
#yes | conda create --name joeenv
#source activate joeenv

# package instalations
#yes | conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
#yes | conda install -c conda-forge scikit-learn
#yes | conda install -c conda-forge matplotlib
#yes | conda install -c anaconda seaborn
#yes | conda install -c conda-forge statsmodels
#yes | pip install torchsummary
