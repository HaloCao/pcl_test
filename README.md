# PCL registration test

## 1. PCL installation(compiling from source):

Dependencies:
Boost, Eigen3, FLANN, VTK.

Install dependencies:
```
sudo apt install libeigen3-dev libboost-dev libflann-dev libvtk5-dev libvtk5-qt4-dev libvtk-java python-vtk
```

Compiling:
```
cd ~
git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
sudo make install
```

## 2. Download this repository:
```
cd ~
git clone https://github.com/JinyaoZhu/pcl_test.git
```

## 3. Install yaml-cpp for configuration:
```
sudo apt install libyaml-cpp-dev 
```

## 4. Compile test files:
build:
```
cd ~/pcl_test
mkdir build
cd build
cmake ..
make
```
run test:
```
./scp
```
