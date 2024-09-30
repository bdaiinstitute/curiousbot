#!/bin/bash

docker rm -f cbot_container
docker run -d -it --network host -v /home/yixuan/curiousbot:/curiousbot --name cbot_container curiousbot_docker
