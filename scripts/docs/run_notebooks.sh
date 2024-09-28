set -e

source .venv/bin/activate
jupyter_workdir="jupyter_temp"
mkdir -p $jupyter_workdir
cd $jupyter_workdir

for notebook in $(ls ../docs/*.ipynb)
do
    echo $notebook
    jupyter execute $notebook
done

cd ..
rm -r $jupyter_workdir
