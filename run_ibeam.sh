#!/bin/bash

source /usr/bin/sagetrader/config.sh

docker pull voyz/ibeam
docker stop ibeam
docker rm ibeam
docker run --rm --env-file /usr/bin/sagetrader/config.sh -p 5000:5000 --name ibeam voyz/ibeam

