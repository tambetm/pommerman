FROM python:3.6

RUN pip install tensorflow keras
RUN pip install git+https://github.com/MultiAgentLearning/playground.git

ADD . /agent

EXPOSE 10080

ENV NAME CrappyAgent1

WORKDIR /agent

# Precompiles the model for faster startup.
RUN python -c "from keras.models import load_model; load_model('modelAlpha_No_Discount_tuned_MCTS_300K_balanced_93.h5')"

ENTRYPOINT ["python"]
CMD ["run.py"]
