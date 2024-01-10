
# How to set python environment ?

## From a IFPEN linux box 

I did this on irlin386204

> All the installation on topaze will be perfomed on SCRATCH because of the quota on the other area.
> I tried on WORK and STORE and it exceeded the quota twice

> Take care not to install on HOME too, because this area is not visible when submitting a job.

### get and copy a miniconda distribution

retrieve the version of miniconda you wish. In this case, i use python 3.10

> Note that Python 3.10 and pytorch 2.0.1 generates many incompatibilities that were difficult to solve.


```basd
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh
scp Miniconda3-py310_23.11.0-2-Linux-x86_64.sh lecomtej@topaze.ccc.cea.fr:/ccc/work/cont013/ifpr11/lecomtej/
```

### Create a directory with python dependencies

On a laptop that is connected to internet, first get all wheel python package that you need.

```bash
mkdir $dependencies
pip download -r ../requirements.txt -d $dependencies
```

This will download from pip index all wheel packages required by your environment.
Then copy them on the *WORK* zone on topaze

```
scp $dependecies/*.whl lecomtej@topaze.ccc.cea.fr:/ccc/work/cont013/ifpr11/lecomtej/dependencies/
```

If needed or missing some other wheel package may be downloaded in `$dependencies` directory to ensure a whole distribution of python

for example:
```
wget https://files.pythonhosted.org/packages/c7/c3/55076fc728723ef927521abaa1955213d094933dc36d4a2008d5101e1af5/wheel-0.42.0-py3-none-any.whl
```

## On topaze

Create a directory for you where to store data and virtual envirnoment.

```bash
cd $CCCSCRATCHDIR/
mkdir lecomtje
cd lecomtje
mkdir data venvs
```

Then install miniconda. 

```bash/ccc/scratch/cont013/ifpr11/lecomtej/lecomtje
$CCCWORKDIR/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -p /ccc/scratch/cont013/ifpr11/lecomtej/lecomtje/miniconda3
```

Then create the virtual environment.

```bash
$CCCSCRACTHDIR/miniconda3/bin/python -m venv drp3d
source $CCCSCRACTHDIR/drp3d/bin/activate
pip install --no-index --find-links=file:///ccc/work/cont013/ifpr11/lecomtej/dependencies torch
```

# lancer le job


