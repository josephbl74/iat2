from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from getQPolitics import Analyst
import matplotlib.pyplot as plt
import numpy as np

def main():

    game = SpaceInvaders(display=False)
    #controller = KeyboardController()
    # controller = RandomAgent(game.na)
    controller = QAgent(game, alpha=0.8, gamma = 0.2)
    # def __init__(self, game, gamma, alpha):
    # controller = RandomAgent(8, 12, 2, game.na, game) 
    
    controller.learn(episodes=50, iterations=10000)
    controller.load("qtable_1_0.75_25_10000.npy")
    # controller.load("Q/0.2-0.8-50-5000.npy")
    
    analyst=Analyst()
    timeCounter=0
    
    game = SpaceInvaders(display=True)
    state = game.reset()

    while True:
        action = controller.select_action(state)
        state, reward, finished = game.step(action)
        
        # timeCounter +=1
        # analyst.updateTandR(timeCounter, reward)
        
        
        
        if(finished):
            break
    print("GAME OVER > ", game.score_val)
    
    
    
    #----POLITICS ANALYSIS----
    
    """ 1. displaying Q """
    
    # ''' v1 >>> does not work no more '''
    # analyst = Analyst("qtable_1_0.75_25_10000.npy")
    # analyst.displayQ()
    
    # ''' v2 '''
    # analyst.load("qtable_1_0.75_25_10000.npy")
    # analyst.displayQ()
    
    """ displaying R(t) """
    
    # analyst.displayR()
    
    """ displaying S(t) """
    
    # analyst.displayQS(controller.Time, controller.State)
    
    
    
    
    
    
    # fake plot
    
    data = [[   3.,    3.,    3.,    3.,    3.,    3.,    3.,    3.,    3.,    3.],
    [  49.,   48.,   48.,   48.,   48. ,  48.,   48.,   48.,   48.,   48.],
    [   9.,   18.,   28.,   38.,   48.,   57.,   66.,   75.,   85.,   95.],
    ]

    data = np.array(data) 

    plt.plot(np.arange(1, 11), data.transpose())
    plt.show()
    

if __name__ == '__main__' :
    main()
