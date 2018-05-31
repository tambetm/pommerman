from pommerman.runner import DockerAgentRunner
from keras.models import load_model
import numpy as np


def featurize(obs):
    # TODO: history of n moves?
    board = np.array(obs['board'])

    # convert board items into bitmaps
    maps = [board == i for i in range(10)]
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    # duplicate ammo, blast_strength and can_kick over entire map
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[obs['position']] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'])
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    return np.stack(maps, axis=2)


class KerasAgent(DockerAgentRunner):
    def __init__(self, model_file):
        super().__init__()
        self.model = load_model(model_file)

    def act(self, obs, action_space):
        feats = featurize(obs)
        probs, values = self.model.predict(feats[np.newaxis])
        action = np.argmax(probs[0])
        return int(action)


if __name__ == "__main__":
    agent = KerasAgent("modelAlpha_No_Discount_tuned_MCTS_300K_balanced_93.h5")
    agent.run()
