set -e

papermill_workdir="papermill_temp"
mkdir $papermill_workdir
cd $papermill_workdir

for notebook in $(ls ../docs/*.ipynb)
do
    echo $notebook
    uv run papermill $notebook $notebook
done

cd ..
rm -r $papermill_workdir
