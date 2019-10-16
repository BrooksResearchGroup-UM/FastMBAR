# Command used to build and upload the package onto both PyPI and Anaconda Cloud

## PyPI
```
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
```

## Anaconda Cloud
```
cd conda
conda build .
anaconda upload /home/xqding/apps/miniconda3/conda-bld/linux-64/fastmbar-0.0.5-py37_0.tar.bz2
conda convert --platform all /home/xqding/apps/miniconda3/conda-bld/linux-64/fastmbar-0.0.4-py37_0.tar.bz2 -o conda_build/
```
