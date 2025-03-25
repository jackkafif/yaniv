from game.combined_model import train_agent, CombinedModel
from game.state import GameState


def train():
    game_env = GameState()
    agent = CombinedModel()
    agent2 = CombinedModel()
    train_agent(agent, agent2, game_env, num_episodes=1000)


if __name__ == "__main__":
    train()
