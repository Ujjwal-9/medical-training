FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
# COPY test-model.h5 .
# COPY train-small.csv .
# COPY util.py .
# COPY gradcam.py .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "test", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure numpy is installed:"
RUN python -c "import numpy"

# The code to run when container is started:
# COPY app.py .
# ENTRYPOINT ["conda", "run", "-n", "test", "python", "gradcam.py", "test.png", "images/"]

