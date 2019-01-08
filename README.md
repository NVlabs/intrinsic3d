# Intrinsic3D
## High-Quality 3D Reconstruction by Joint Appearance and Geometry Optimization with Spatially-Varying Lighting (ICCV 2017)

[![Intrinsic3D](https://vision.in.tum.de/_media/data/datasets/intrinsic3d/maier2017intrinsic3d_teaser.jpg?w=700&tok=b8e6f7)](https://vision.in.tum.de/_media/spezial/bib/maier2017intrinsic3d.pdf)

### License ###
Copyright (c) 2019, NVIDIA Corp. and Technical University of Munich All Rights Reserved.
The Intrinsic3D source code is available under the [BSD license](http://opensource.org/licenses/BSD-3-Clause), please see the [LICENSE](LICENSE) file for details.
All data in the [Intrinsic3D Dataset](https://vision.in.tum.de/data/datasets/intrinsic3d) is licensed under a [Creative Commons 4.0 Attribution License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), unless stated otherwise.

### Project page ###
 * [NVIDIA Research project page](https://research.nvidia.com/publication/2017-10_Intrinsic3D%3A-High-Quality-3D)
 * [Intrinsic3D Dataset page at TUM](https://vision.in.tum.de/data/datasets/intrinsic3d)

### Paper ###
 * [ICCV (PDF)](https://vision.in.tum.de/_media/spezial/bib/maier2017intrinsic3d.pdf)
 * [ArXiv (PDF)](https://arxiv.org/pdf/1708.01670.pdf)
 
### Project members ###
* [Robert Maier](https://vision.in.tum.de/members/maierr), Technische Universität München
* [Kihwan Kim](https://research.nvidia.com/person/kihwan-kim), NVIDIA
* [Daniel Cremers](https://vision.in.tum.de/members/cremers), Technische Universität München
* [Jan Kautz](https://research.nvidia.com/person/jan-kautz), NVIDIA
* [Matthias Nießner](http://www.niessnerlab.org/members/matthias_niessner/profile.html), Technische Universität München

## Summary
**Intrinsic3D** is a method to obtain high-quality 3D reconstructions from low-cost RGB-D sensors.
The algorithm recovers fine-scale geometric details and sharp surface textures by simultaneously optimizing for reconstructed geometry, surface albedos, camera poses and scene lighting.

This work is based on our publication  
* [Intrinsic3D: High-Quality 3D Reconstruction by Joint Appearance and Geometry Optimization with Spatially-Varying Lighting](https://vision.in.tum.de/_media/spezial/bib/maier2017intrinsic3d.pdf) 
International Conference on Computer Vision (ICCV) 2017.

If you find our source code or [dataset](https://vision.in.tum.de/data/datasets/intrinsic3d) useful in your research, please cite our work as follows:
```
@inproceedings{maier2017intrinsic3d,
   title = {{Intrinsic3D}: High-Quality {3D} Reconstruction by Joint Appearance and Geometry Optimization with Spatially-Varying Lighting},
   author = {Maier, Robert and Kim, Kihwan and Cremers, Daniel and Kautz, Jan and Nie{\ss}ner, Matthias},
   booktitle = {International Conference on Computer Vision (ICCV)},
   year = {2017}
}
```

## Installation
As the code was mostly developed and tested on Ubuntu Linux (16.10 and 18.10), we only provide the build instructions for Ubuntu in the following.
However, the code should also work on Windows with Visual Studio 2013.

Please first clone the source code:
```
git clone https://github.com/NVlabs/intrinsic3d.git
```

### Dependencies
Building Intrinsic3D requires
[CMake](https://cmake.org/download/),
[Eigen](http://eigen.tuxfamily.org/),
[OpenCV](http://opencv.org/downloads.html),
[Boost](http://www.boost.org/users/download/) and
[Ceres Solver](http://ceres-solver.org/) (with [CXSparse](https://github.com/TheFrenchLeaf/CXSparse) on Windows)
as third-party libraries.
The following command installs the dependencies from the default Ubuntu repositories:
```
sudo apt install cmake libeigen3-dev libboost-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libopencv-dev
```

Please install [Ceres Solver](http://ceres-solver.org/installation.html) using the following commands:
```
# create third_party subdirectory (in Intrinsic3D root folder)
mkdir third_party && cd third_party

# install Ceres Solver dependencies
sudo apt install libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev

# download and extract Ceres Solver source code package
wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz
tar xvzf ceres-solver-1.14.0.tar.gz
cd ceres-solver-1.14.0

# compile and install Ceres Solver
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$EXT -DCMAKE_INSTALL_PREFIX=$PWD/../../ -DCXX11=ON -DSUITESPARSE=ON -DCXSPARSE=ON -DEIGENSPARSE=ON -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
make -j6
make install

# go back into Intrinsic3D source code root
cd ../../../
```

### Build
To compile Intrinsic3D, use the standard CMake approach:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCeres_DIR=$PWD/../third_party/lib/cmake/Ceres
make -j6
```

## Data
Download one of the RGB-D sequences from the [Intrinsic3D Dataset](https://vision.in.tum.de/data/datasets/intrinsic3d). If you want to use one of your own RGB-D sequences, you can convert it into the dataset format specified [here](https://vision.in.tum.de/data/datasets/intrinsic3d#format).
In the following, we will reconstruct and refine the [Lion sequence](https://vision.in.tum.de/_media/data/datasets/intrinsic3d/lion-rgbd.zip) using our approach.
```
# creating working folder
cd data
mkdir lion && cd lion

# download, unzip and rename lion dataset
wget https://vision.in.tum.de/_media/data/datasets/intrinsic3d/lion-rgbd.zip
unzip lion-rgbd.zip
mv lion-rgbd rgbd
```


## Usage
To run the Intrinsic3D applications, we continue in the current data folder ```data/lion/``` and copy the default YAML configuration files into it:
```
# copy defautl configuration files into current data folder
cp ../*.yml .
```
The dataset configuration file ```sensor.yml``` is loaded in all applications and specifies the input RGB-D dataset.
Please note that the working folder of each application will be set to the folder containing the ```sensor.yml``` passed as argument.
The outputs will be generated in the subfolders ```fusion/``` and  ```intrinsic3d/``` respectively.

### Keyframe Selection
We first run the keyframe selection to discard blurry frames based on a single-frame blurriness measure:
```
../../build/bin/AppKeyframes -s sensor.yml -k keyframes.yml
```
This will generate the keyframes file ```fusion/keyframes.txt```.
The window size for keyframe selection can be adjusted through the parameter ```window_size``` in the ```keyframes.yml``` configuration file.

### SDF Fusion
Performs a fusion of the RGB-D input frames in a voxel-hashed signed distance field:
```
../../build/bin/AppFusion -s sensor.yml -f fusion.yml
```
Input: 
* Config file for RGB-D data.
* Config file for SDF fusion.

Output:
* SDF volume dump.
* Output mesh.

Since the framework is very computationally demanding, it is recommended to crop the 3D reconstruction already during the SDF fusion process.
You can therefore specify the 3D clip coordinates (in absolute coordinates of the first camera frame) by setting the ```crop_*``` parameters in the ```fusion.yml``` configuration file. To disable clipping, you can set all ```crop_*``` parameters to 0.0.
To accelerate finding a suitable clip volume, increase the ```voxel_size``` value and integrate keyframes only (by specifying a filename in the parameter ```keyframes```).


### AppIntrinsic3D

Joint optimization of scene colors, geometry, camera poses and camera intrinsics.

Usage:
```
../../build/bin/AppIntrinsic3D -s sensor.yml -i intrinsic3d.yml
```

## Contact
If you have any questions, please contact [Robert Maier &lt;robert.maier@tum.de>](mailto:robert.maier@tum.de) and [Kihwan Kim &lt;kihwank@nvidia.com>](mailto:kihwank@nvidia.com).
