# Use the official CUDA base image with TensorFlow support
FROM nvidia/cuda:12.3.0-base-ubuntu22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install cuDNN 8.9
# Install cuDNN 8.9 via the NVIDIA installation script
RUN apt-get update && apt-get install -y \
    libcudnn8 \
    libcudnn8-dev \
    && apt-get clean
    
# Install Python 3, pip, and other necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean

# Install TensorFlow, Keras, Jupyter, and other libraries
RUN pip3 install --upgrade pip && \
    pip3 install tensorflow==2.17.1 keras==3.5.0 jupyter matplotlib numpy pandas

# Set the working directory inside the container
WORKDIR /workspace

# Expose the Jupyter port (default: 8888)
EXPOSE 8888

# Start Jupyter Notebook when the container runs
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
