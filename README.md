# Building An Object Detection Application with NVIDIA NGC on GPU-Powered AWS Instances
We will walk you through running an object detection model with NVIDIA Metropolis - an application framework that simplifies the development, deployment and scale of AI-enabled video analytics applications from edge to cloud.
## Requirements:
A server with 1 or more (preferably 8) NVIDIA A100's, either on cloud (AWS p4dn.24xlarge) or on-prem.
GPUs sliced with 2g.10gb MIG profile.
NVIDIA Driver 460+
## Pull NVIDIA Container from AWS marketplace:
1. choose the deepstream container from aws marketplace: https://aws.amazon.com/marketplace/featured-seller/nvidia-ngc
2. Retrieve the login command to authenticate your Docker client to your registry:
```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 709825985650.dkr.ecr.us-east-1.amazonaws.com
```
4. pull the container
```
docker pull 709825985650.dkr.ecr.us-east-1.amazonaws.com/nvidia/containers/nvidia/deepstream:5.1-21.02-triton
```
6. run the container
```
docker run --gpus all --rm -it -p 8888:888 709825985650.dkr.ecr.us-east-1.amazonaws.com/nvidia/containers/nvidia/deepstream:5.1-21.02-triton
```
## Download the trafficcamnet model from NGC
Inside the container run the next commands to download the model from ngc.
```
cd /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models

## download trafficcamnet
mkdir -p ../../models/tlt_pretrained_models/trafficcamnet && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/files/resnet18_trafficcamnet_pruned.etlt \
    -O ../../models/tlt_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned.etlt && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/files/trafficnet_int8.txt \
    -O ../../models/tlt_pretrained_models/trafficcamnet/trafficnet_int8.txt
```

## Generate TensorRT Engine:
NVIDIA TensorRT is a model optimization engine and is designed to work in a complementary fashion with training frameworks such as TensorFlow, PyTorch, and MXNet. It focuses specifically on running an already-trained network quickly and efficiently on NVIDIA hardware and you can find detailed documentation at this address: https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html .

Run the next commands to move related config files to the config folder and make the TensorRT engine:

```
cd ~/..

# move config files and run deepstream app for generating tensorrt engines
git clone https://github.com/AshishSardana/ds_triton.git

cp /ds_triton/engine_bs/config_infer_primary_trafficcamnet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/config_infer_primary_trafficcamnet.txt

cp /ds_triton/engine_bs/deepstream_app_source1_trafficcamnet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/deepstream_app_source1_trafficcamnet.txt


deepstream-app -c /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/deepstream_app_source1_trafficcamnet.txt

```

## Prepare the dataset

```
# prepare video samples
apt-get update && apt-get install -y ffmpeg

cd /opt/nvidia/deepstream/deepstream-5.1/samples/

./prepare_classification_test_video.sh

# move the engine, config.pbtxt and label files for trafficcamnet
cd /opt/nvidia/deepstream/deepstream-5.1/samples/

mkdir -p trtis_model_repo/trafficcamnet/1
cp models/tlt_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned.etlt_b50_gpu0_int8.engine trtis_model_repo/trafficcamnet/1/resnet18_trafficcamnet_pruned.etlt_b50_gpu0_int8.engine

cp /ds_triton/trafficcamnet_config.pbtxt /opt/nvidia/deepstream/deepstream-5.1/samples/trtis_model_repo/trafficcamnet/config.pbtxt

cp /ds_triton/labels_trafficcamnet.txt /opt/nvidia/deepstream/deepstream-5.1/samples/trtis_model_repo/trafficcamnet/labels.txt

# move the model and app config file of the use-case
cp /ds_triton/config/config_infer_primary_trafficcamnet_triton.txt /opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis/config_infer_primary_trafficcamnet_triton.txt

```

## Deploy the model

```
deepstream-app -c /opt/nvidia/deepstream/deepstream-5.1/samples/configs/tlt_pretrained_models/deepstream_app_source1_trafficcamnet.txt
```

The result can be seen in the root directory as a mp4 video with the name final4.mp4.
