FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
ENV TZ=Asia/Tokyo
RUN apt-get -y update && \
    apt-get install sudo && \
    mkdir -p /etc/sudoers.d/ && \
    touch /etc/sudoers.d/ubuntu && \
    groupadd ubuntu -g 1000 && \
    useradd ubuntu -m -u 1000 -g 1000 && \
    echo "%ubuntu ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/ubuntu

USER ubuntu
ENV HOME /home/ubuntu
WORKDIR $HOME

RUN sudo apt-get update && \
    sudo apt-get upgrade -y && \
    sudo apt-get install -y \
    zip \
    unzip \
    git \
    make \
    vim \
    wget \
    curl \
    screen \
    llvm \
    xz-utils \
    libxml2-dev \
    libxmlsec1-dev \
    build-essential \
    libncursesw5-dev \
    libgdbm-dev \
    libc6-dev \
    zlib1g-dev \
    libsqlite3-dev \
    tk-dev \
    libssl-dev \
    openssl \
    libbz2-dev \
    libreadline-dev \
    libffi-dev \
    liblzma-dev

ENV PYTHON_VERSION 3.9.13
ENV PYENV_ROOT $HOME/.pyenv
ENV PYTHON_ROOT $PYENV_ROOT/versions/$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PATH $PYENV_ROOT/bin/:$PATH

RUN git clone https://github.com/yyuu/pyenv.git $PYENV_ROOT && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.bash_profile && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.bash_profile && \
    echo 'eval "$(pyenv init --path)"' >> $HOME/.bash_profile
    
RUN ["/bin/bash","-c","source $HOME/.bash_profile && pyenv install $PYTHON_VERSION"] && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

RUN python && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install jupyter ipython notebook jupyterlab && \
    jupyter notebook --generate-config && \
    echo "c = get_config()" >> $HOME/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> $HOME/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = ''" >> $HOME/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> $HOME/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> $HOME/.jupyter/jupyter_notebook_config.py

RUN python && \
    pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install flax numpy pandas matplotlib seaborn Pillow opencv-python albumentations && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
