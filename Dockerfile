# Use the latest Python 3.12 image
FROM python:3.12-slim

# Set working directory

# Install dependencies and download Miniforge (ARM-compatible)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh && \
    bash Miniforge3-Linux-$(uname -m).sh -b -p /opt/conda && \
    rm Miniforge3-Linux-$(uname -m).sh

# Add conda to path
ENV PATH="/opt/conda/bin:$PATH"


WORKDIR /app

# Copy local environment requirements
COPY environment.yml /app

# Copy requirements file
COPY requirements.txt /app
COPY .env /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy environment.yml and install dependencies
COPY environment.yml /app
RUN conda env create -f /app/environment.yml


# Set Streamlit as the entry point
COPY . /app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

