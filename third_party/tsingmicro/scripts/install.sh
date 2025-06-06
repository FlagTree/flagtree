#!/bin/bash

apt install git
apt install lld

pip uninstall triton

pip install gitpython
pip install torch==2.7.0 torchvision
