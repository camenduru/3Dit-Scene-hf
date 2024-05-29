# Reference:
# https://github.com/cvpaperchallenge/Ascender
# https://github.com/nerfstudio-project/nerfstudio

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG USER_NAME=dreamer
ARG GROUP_NAME=dreamers
ARG UID=1000
ARG GID=1000

# Set compute capability for nerfacc and tiny-cuda-nn
# See https://developer.nvidia.com/cuda-gpus and limit number to speed-up build
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV TCNN_CUDA_ARCHITECTURES=90;89;86;80;75;70;61;60
# Speed-up build for RTX 30xx
# ENV TORCH_CUDA_ARCH_LIST="8.6"
# ENV TCNN_CUDA_ARCHITECTURES=86
# Speed-up build for RTX 40xx
# ENV TORCH_CUDA_ARCH_LIST="8.9"
# ENV TCNN_CUDA_ARCHITECTURES=89

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/home/${USER_NAME}/.local/bin:${PATH}:/home/${USER_NAME}/tinycudann/bin
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

# apt install by root user
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    wget \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Change user to non-root user
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}
USER ${USER_NAME}

RUN pip install --upgrade pip setuptools==69.5.1 ninja
RUN pip install xformers==0.0.22 torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html           

RUN pip install nerfacc==0.5.2 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html

COPY requirements.txt /tmp

RUN cd /tmp && pip install -r requirements.txt
RUN pip install -U fastapi pydantic

RUN ls .


RUN rm -rf /home/${USER_NAME}/threestudio
RUN rm -rf /home/${USER_NAME}/.cache

RUN ls .

ADD "https://api.github.com/repos/zqh0253/3DitScene/commits?per_page=1" latest_commit
RUN git clone https://github.com/zqh0253/3DitScene.git /home/${USER_NAME}/threestudio --recursive

WORKDIR /home/${USER_NAME}/threestudio

RUN wget --quiet https://www.dropbox.com/scl/fi/ubyeroi2b85y78mbyy9or/1.zip?rlkey=b1sfgbmlmx3jz4onwk9qjj2rs -O tmp.zip
RUN wget --quiet https://www.dropbox.com/scl/fi/2s4b848d4qqrz87bbfc2z/cache.zip?rlkey=f7tyf4952ey253xlzvb1lwnmc -O tmp.zip
RUN unzip tmp.zip

RUN pip install ./submodules/segment-anything-langsplat
RUN pip install ./submodules/MobileSAM-lang

RUN wget --quiet https://www.dropbox.com/scl/fi/rhl1r9qww9fq6jtjmh43x/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl?rlkey=xp02kfjvyk9urnacybp4ll108 -O diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
RUN pip install diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl

RUN pip install ./submodules/simple-knn

# RUN cd diff-gaussian-rasterization && git show-ref --heads

RUN mkdir ckpts 
RUN wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./ckpts/sam_vit_h_4b8939.pth
RUN cp /home/${USER_NAME}/threestudio/submodules/MobileSAM-lang/weights/mobile_sam.pt ./ckpts/


# RUN git checkout 23b2d71
CMD ["python", "gradio_app_single_process.py", "--listen", "--hf-space"]
