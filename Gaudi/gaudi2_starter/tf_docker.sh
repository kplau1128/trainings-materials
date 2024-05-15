#!/bin/bash

docker_version_default="1.14.0"
tf_version_default="2.15.0"
build_version_default="493"

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "Usage: $0 [docker_version] [tf_version]"
  echo "  docker_version (optional): Docker version (default: $docker_version_default)"
  echo "  tf_version (optional): TensorFlow version (default: $tf_version_default)"
  echo "  build_version (optional): Build version (default: $build_version_default)"
  exit 0
fi

docker_version=${1:-$docker_version_default}
tf_version=${2:-$tf_version_default}
build_version=${3:-$build_version_default}

docker ps | grep -wq tf_gaudi2 2>/dev/null
if [ $? == 0 ]; then
  docker stop tf_gaudi2
  docker rm  tf_gaudi2
  sleep 2
fi

home_dir="$HOME"

docker run -it  --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechani --rm --cap-add=sys_nice --net=host --ipc=host \
 -e HTTP_PROXY=http://proxy-chain.intel.com:911 -e HTTPS_PROXY=http://proxy-chain.intel.com:912 -v ${home_dir}/gaudi2_starter/:/root/  --privileged \
 -v /data/:/data2/ -v /datasets/data:/data --name tf_gaudi2 --workdir=/root/ \
 vault.habana.ai/gaudi-docker/${docker_version}/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-${tf_version}:${docker_version}-${build_version}

