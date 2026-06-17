#install ROCm first!
sudo apt-get update
sudo apt-get install -y \
    cmake \
    git \
    build-essential \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libfreetype6-dev \
    libgoogle-perftools-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libqt5widgets5 qtbase5-dev \
    libglew-dev \
    libcgns-dev \
    libqt6svg6-dev \
    qt6-base-dev

export LD_LIBRARY_PATH=/var/home/bazzite/Data/dotLocal/lib/

git clone https://github.com/colmap/colmap.git
cd colmap
# Select GIT tag 4.0.3
mkdir build && cd build
rm -rf CMakeCache.txt CMakeFiles/
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHIPIFY=ON \
    -DCUDA_ENABLED=OFF \
    -DROCBLAS_ENABLED=ON \
    -DCMAKE_PREFIX_PATH=/opt/rocm

make -j$(nproc)


