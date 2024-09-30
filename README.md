# curiousbot

### Build docker image
`docker build -t curiousbot_docker .`

### Docker usage
`bash ./scripts/start_docker.sh`
`bash ./scripts/start_shell.sh`

### Inside docker
`ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 send_buffer_limit:=200000000 num_threads:=4`

