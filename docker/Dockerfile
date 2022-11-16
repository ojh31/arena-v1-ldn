FROM continuumio/miniconda3

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Activate the environment, and make sure it's activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

COPY hello.py /app/hello.py
ENV NREPEATS=7

ENV PATH /opt/conda/envs/myenv/bin:$PATH
CMD ["bash", "-c", "source activate myenv && python app/hello.py"]