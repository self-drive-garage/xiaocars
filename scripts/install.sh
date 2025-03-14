
# tar -xf deps/fastdds tar gz and move to /usr/local/fast-rtps-1.5.0-1

sudo apt-get -y install libtinyxml2-dev #libncurses5-dev

# ON RASPBERRY PIs
# sudo apt install pigpio libpigpio-dev pigpio-tools


SUBSYSTEM=="gpio", KERNEL=="gpiochip*", MODE="0660", GROUP="gpio"


 1164  sudo usermod -a -G gpio $USER
 1165  sudo nano /etc/udev/rules.d/99-gpio.rules

 SUBSYSTEM=="gpio", KERNEL=="gpiochip*", MODE="0660", GROUP="gpio"


 1166  sudo udevadm control --reload-rules && sudo udevadm trigger
 1167  ll /dev/gpiochip0
 1168  grep -r gpio /etc/udev/rules.d/
 1169  grep -r gpio /usr/lib/udev/rules.d/  # Also check this location
 1170  sudo groupadd gpio  # Only if the group doesn't exist
 1171  sudo usermod -a -G gpio $USER  # Or the group found in step 1
 1172  ls -l /dev/gpiochip*
 1173  sudo udevadm control --reload-rules
 1174  sudo udevadm trigger
 1175  ls -l /dev/gpiochip*
 1176  bazel-bin/examples/gpiod/gpio_test
 1177  groups
 1178  ~/cursor-0.45.14-build-250219jnihavxsz-x86_64.AppImage
 1179  groups
 1180  history

echo "/usr/local/fast-rtps-1.5.0-1/lib" | sudo tee /etc/ld.so.conf.d/fastdds.conf && sudo ldconfig
