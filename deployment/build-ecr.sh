#!/bin/bash

# Check to see if the required parameters have been provided:
if [ -z "$1" ] || [ -z "$2" ] ||  [ -z "$3" ]; then
    echo "Please provide the region_name, account_id and image name to build the ecr image."
    echo "For example: ./build-ecr.sh <region_name> <aws_account_id> <your_image_name>"
    exit 1
fi

# Get reference for all important folders
TEMPLATE_DIR="$PWD"
SOURCE_DIR="${TEMPLATE_DIR}/../source"

echo "------------------------------------------------------------------------------"
echo "[Init] Get Env"
echo "------------------------------------------------------------------------------"
REGION=$1
ACCOUNT_ID=$2
IMAGE_NAME=$3

if [[ $1 == cn-* ]];
then
  DOMAIN=$2.dkr.ecr.$1.amazonaws.com.cn
else
  DOMAIN=$2.dkr.ecr.$1.amazonaws.com
fi

echo ECR_DOMAIN ${DOMAIN}

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${DOMAIN}


echo "---------------------------------------------------------------------------------------"
echo "[Build] Build Pedestrian Properties Recognition Image (GPU Version)        "
echo "---------------------------------------------------------------------------------------"
cd ${SOURCE_DIR}
docker build -t ${IMAGE_NAME}:latest -f containers/pedestrian-properties-recognition/Dockerfile containers/pedestrian-properties-recognition/
docker tag ${IMAGE_NAME}:latest ${DOMAIN}/${IMAGE_NAME}:latest

echo "---------------------------------------------------------------------------------------"
echo "[Push] Push Pedestrian Properties Recognition Image (GPU Version)          "
echo "---------------------------------------------------------------------------------------"
cd ${SOURCE_DIR}
aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION} >/dev/null 2>&1
docker push ${DOMAIN}/${IMAGE_NAME}:latest
