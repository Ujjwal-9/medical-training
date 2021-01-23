FROM python:3.6.7

WORKDIR /app

# Create the environment:
# RUN apk update
# RUN apk add make automake gcc g++ subversion python3-dev

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY test-model.h5 .
COPY train-small.csv .
COPY gradcam.py .
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "test", "/bin/bash", "-c"]

# Make sure the environment is activated:
#RUN echo "Make sure numpy is installed:"
#RUN python -c "import numpy"

# The code to run when container is started:
# COPY app.py .
# ENTRYPOINT ["conda", "run", "-n", "test", "python", "gradcam.py", "test.png", "images/"]

