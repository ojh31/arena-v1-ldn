FROM continuumio/miniconda3

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

COPY hello.py /app/hello.py
ENV NREPEATS=7

ENV PATH /opt/conda/envs/myenv/bin:$PATH
CMD ["bash", "-c", "source activate myenv && python app/hello.py"]