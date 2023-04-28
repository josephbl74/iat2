from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from getQPolitics import Analyst

def main():

    game = SpaceInvaders(display=False)
    #controller = KeyboardController()
    # controller = RandomAgent(game.na)
    controller = QAgent(game, alpha=0.8, gamma = 0.2)
    controller.learn(env=game, nbEpisodes=50, maxSteps=10000)
    controller.load("qtable_1_0.75_25_10000.npy")
    
    newGame = SpaceInvaders(display=True)
    state = newGame.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = newGame.step(action)
        if(is_done):
            break
    print("Game over !")
    print("Score : ", newGame.score_val)
    # analyst = Analyst("qtable_1_0.75_25_10000.npy")
    # analyst.display()

if __name__ == '__main__' :
    main()
