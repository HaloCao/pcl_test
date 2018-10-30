# PCL registration test

## PCL installation(compiling from source):

Dependencies:
Boost, Eigen3, FLANN, VTK.

Install dependencies:
```
sudo apt install libeigen3-dev libboost-dev libflann-dev libvtk5-dev libvtk5-qt4-dev libvtk-java python-vtk libopenni-dev
```

Compiling:
```
git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
sudo make install
```

## Compile test files:
In root directory **pcl_test/**:

build:
```
mkdir build
cd build
cmake ..
make
```
run test:
```
./scp ../data/bun01.pcd ../data/bun4.pcd
```
