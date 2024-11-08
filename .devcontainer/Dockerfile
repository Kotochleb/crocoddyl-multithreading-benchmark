FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ENV RCUTILS_COLORIZED_OUTPUT=1
ENV CMAKE_BUILD_TYPE=RelWithDebInfo

# Install required packages
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    --mount=type=cache,sharing=locked,target=/root/.cache \
    apt-get update \
    && apt-get upgrade -y \
    && apt-get install -qqy --no-install-recommends \
        # general dependencies
        bash-completion \
        build-essential \
        clang-format \
        cmake \
        curl \
        gdb \
        git \
        gpg \
        htop \
        nano \
        sudo \
        wget \
        vim \
        libfontconfig1 \
        libxrender1 \
        libxi6 \
        libxtst6 \
        valgrind \
        dbus-x11 \
        kcachegrind \
        graphviz \
        massif-visualizer \
        coinor-libipopt-dev \
        cmake \
        build-essential \
        ca-certificates \
        cmake \
        coinor-libipopt-dev \
        git \
        libassimp-dev \
        libboost-all-dev \
        libeigen3-dev \
        liboctomap-dev \
        libqhull-dev \
        libtinyxml-dev \
        liburdfdom-dev \
        python-is-python3 \
        python3-dev \
        python3-numpy \
        python3-scipy \
        python-is-python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        libpoco-dev \
        libeigen3-dev \
        libnss3-dev \
        libgdk-pixbuf2.0-dev \
        libgtk-3-dev \
        libxss-dev \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        # python dependencies
        python-is-python3 \
        python3-flake8 \
        python3-pip \
        python3-setuptools \
    && python -m pip install -U \
        argcomplete \
        pre-commit

# Install Intel VTune
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    --mount=type=cache,sharing=locked,target=/root/.cache \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y intel-oneapi-vtune

# Create the user
RUN groupadd --gid 1000 gepetto \
    && useradd --uid 1000 --gid 1000 -m gepetto \
    && echo gepetto ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/gepetto \
    && chmod 0440 /etc/sudoers.d/gepetto

# Keep bash history when docker is rebuilt
RUN mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R gepetto /commandhistory \
    && echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" >> "/home/gepetto/.bashrc"

USER gepetto

WORKDIR /home/gepetto/dependencies
COPY install_dependency.sh /home/gepetto/dependencies/install_dependency.sh
RUN bash install_dependency.sh stack-of-tasks/eigenpy/v3.9.0/4
RUN bash install_dependency.sh humanoid-path-planner/hpp-fcl/devel/16
RUN bash install_dependency.sh stack-of-tasks/pinocchio/v3.2.0/6
RUN bash install_dependency.sh gepetto/example-robot-data/v4.1.0/16
RUN bash install_dependency.sh loco-3d/crocoddyl/v2.1.0/6

WORKDIR /home/gepetto/benchmark

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:${PYTHONPATH}

RUN echo "source /opt/intel/oneapi/vtune/latest/env/vars.sh" >> /home/gepetto/.bashrc \
    && echo "export DBUS_SESSION_BUS_ADDRESS=\`dbus-daemon --fork --config-file=/usr/share/dbus-1/session.conf --print-address\`" >> /home/gepetto/.bashrc

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
