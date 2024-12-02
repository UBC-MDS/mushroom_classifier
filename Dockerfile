FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# add conda-lock
COPY conda-linux-64.lock /tmp/conda-linux-64.lock

USER root

# for Quarto PDF rendering
RUN sudo apt update \
    && sudo apt install -y lmodern
  

USER $NB_UID

RUN mamba update --quiet --file /tmp/conda-linux-64.lock \
    && mamba clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"
