#docker build -t ksp_notebook .
#docker run -p 8888:8888 -v "$PWD":/home/jovyan/work ksp_notebook
FROM jupyter/datascience-notebook


RUN  conda config --add channels defaults \
    && \
    conda config --add channels bioconda \
    && \
    conda config --add channels conda-forge

USER $NB_UID
RUN conda install --quiet --yes \
            'python-igraph=0.8*' \
            && \
            conda clean --all -f -y && \
            fix-permissions $CONDA_DIR && \
            fix-permissions /home/$NB_USER
