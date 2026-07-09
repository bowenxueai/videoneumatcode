FROM nvcr.io/nvidia/pytorch:25.09-py3

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# After installing Miniconda
ENV PATH="/opt/miniconda/bin:${PATH}"
ENV PIP_NO_CACHE_DIR=1

# Accept Anaconda repo TOS non-interactively (required for defaults channels)
RUN /opt/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY environment.yml /environment.yml
RUN /opt/miniconda/bin/conda env create -f /environment.yml && \
    /opt/miniconda/bin/conda clean -afy && \
    rm -rf /root/.cache/pip

RUN /opt/miniconda/envs/neugen/bin/pip install opencv-python-headless --no-cache-dir

# Drop the init/source lines (non-interactive); use conda run instead
SHELL ["conda", "run", "-n", "neugen", "/bin/bash", "-c"]
# Optional: Ensure the correct environment path is added
ENV PATH="/opt/miniconda/envs/neugen/bin:$PATH"

# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "neugen"]
# CMD ["bash"]
