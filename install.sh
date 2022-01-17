#!/bin/bash

dir=$(readlink -f .)
ln -s $dir /usr/bin/sagetrader
cp ibeam.service /etc/systemd/system
chmod 740 run_ibeam.sh
chmod 600 config.sh
systemctl start ibeam
