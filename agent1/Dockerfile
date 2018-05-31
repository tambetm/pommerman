FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt-get update && apt-get install -y git curl bzip2

# Install miniconda to /miniconda
RUN curl -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

#RUN conda install keras-gpu
RUN pip install tensorflow-gpu keras
RUN pip install git+https://github.com/MultiAgentLearning/playground.git

ADD . /agent

#RUN python -c "from keras.models import load_model; load_model("modelAlpha_No_Discount_tuned_MCTS_300K_balanced_93.h5")"

EXPOSE 10080

ENV NAME Agent

WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
