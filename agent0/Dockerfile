FROM python:3.6

ADD . /agent

RUN pip install git+https://github.com/MultiAgentLearning/playground.git

EXPOSE 10080

ENV NAME FlegmaticAgent

# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
