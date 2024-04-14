# Use CUDA base image
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04 AS builder

# Install OpenVINO dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    lsb-release wget ca-certificates && \
    wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021 && \
    apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021 && \
    echo "deb https://apt.repos.intel.com/openvino/2021 all main" | tee /etc/apt/sources.list.d/intel-openvino-2021.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    intel-openvino-runtime-ubuntu18-2021.4.582

# Install additional dependencies
RUN apt-get install -y --no-install-recommends \
    python3-pip \
    # Add any other dependencies needed for your model training and inference
    && rm -rf /var/lib/apt/lists/*

# Set up OpenVINO environment variables
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino
ENV LD_LIBRARY_PATH=/opt/intel/openvino/deployment_tools/inference_engine/external/hddl/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/gna/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_lnx/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/cldnn/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/omp/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/va/lib:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64:/opt/intel/openvino/deployment_tools/ngraph/lib:/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites

# Set up Python environment
RUN pip3 install --upgrade pip setuptools wheel

# Copy model training and inference scripts
WORKDIR /app
COPY . /app

# Set up entrypoint for running model training or inference
ENTRYPOINT ["python3", "train.py"]
ENTRYPOINT ["python3", "detect.py"]
