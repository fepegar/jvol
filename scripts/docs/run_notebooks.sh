set -e

# source .venv/bin/activate
papermill_workdir="papermill_temp"
mkdir -p $papermill_workdir
cd $papermill_workdir

for notebook in $(ls ../docs/*.ipynb)
do
    echo $notebook
    papermill $notebook $notebook -k python3
done

cd ..
rm -r $papermill_workdir
