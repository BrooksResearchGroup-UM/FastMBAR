# Command used to build and upload the package onto both PyPI and Anaconda Cloud

## PyPI
```
python3 setup.py sdist bdist_wheel
twine upload dist/*
```

## Anaconda Cloud
```
conda-build conda -c pytorch
anaconda upload path_to_fastmbar-0.0.2-py35h95ea65b_0.tar.bz2
conda convert --platform linux-64 path_to_fastmbar-0.0.2-py35h95ea65b_0.tar.bz2 -o 
```