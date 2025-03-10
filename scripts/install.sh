
# tar -xf deps/fastdds tar gz and move to /usr/local/fast-rtps-1.5.0-1

sudo apt-get -y install libtinyxml2-dev

echo "/usr/local/fast-rtps-1.5.0-1/lib" | sudo tee /etc/ld.so.conf.d/fastdds.conf && sudo ldconfig
