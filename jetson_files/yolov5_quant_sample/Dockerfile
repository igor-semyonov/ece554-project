# Start FROM Nvidia Tensorrt image
#FROM nvcr.io/nvidia/tensorrt:21.09-py3

#This worked, but cuda pytorch is not compoatible 
#From nvcr.io/nvidia/l4t-tensorrt:r8.0.1-runtime

#nvidia-container-cli: requirement error: unsatisfied condition: cuda >= 11.4
#From nvcr.io/nvidia/l4t-tensorrt:r8.4.1-runtime

#same issue with cuda pytorch
#From nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime


#Test
From nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3


# Install linux packages
RUN apt update && apt install -y libgl1-mesa-dev

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache -r requirements.txt && pip install --no-cache jupyter
#RUN pip install --no-cache torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install pytorch-quantization toolkit
#RUN pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install --no-cache-dir --index-url https://pypi.nvidia.com --index-url https://pypi.org/simple pytorch-quantization==2.1.3

# Create working directory
RUN mkdir -p /root/space/projects
WORKDIR /root/space/projects