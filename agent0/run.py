from pommerman.runner import DockerAgentRunner


class MyAgent(DockerAgentRunner):
    def act(self, observation, action_space):
        return 0


if __name__ == "__main__":
    agent = MyAgent()
    agent.run()
