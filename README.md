# Efficient Long-Range Convolutions for Point Clouds

This is a lightweight implementation of the long-range convolutional layers in the paper (add Arxiv later)

The code is written in python 3.6 and Tensorflow 2.x (so far 2.1 seems to be the one with fewer issues)

The dependencies are

- numpy

- tensorflow 2.x

- numba

- h5py 


This repo follows the simple folder structure: 

-```long_range_convolutions/``` : contains the source files

-```examples/``` : contains a few examples using the modules 

-```run/```: contains the examples encoded in a Json file

you will need to create another folder data/ where all the data will be stored, i.e., you need to type
```mkdir data```

The codes are built so they require a json file for execution. 
Several examples are provided in the the ```run/``` folder. In addition, I added a few bash files to be used  with slurm 
