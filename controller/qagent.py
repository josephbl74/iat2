import numpy as np
from game.SpaceInvaders import SpaceInvaders
from getQPolitics import Analyst

class EpsilonProfile:
    def __init__(self, initial=1., final=0., dec_episode=1., dec_step=0.):
        self.initial = initial          # initial epsilon in epsilon-greedy
        self.final = final              # final epsilon in epsilon-greedy
        self.dec_episode = dec_episode  # amount of decrement of epsilon in each episode is dec_episode / (number of episodes - 1)
        self.dec_step = dec_step        # amount of decrement of epsilon in each step

class QAgent():
    # def __init__(self, game: SpaceInvaders, gamma: float=1, alpha: float = 0.2):
    # def __init__(self, game: SpaceInvaders, gamma: float=0.2, alpha: float = 0.2):
    def __init__(self, game, gamma, alpha):
        self.Q = np.zeros([40, 40, 40, game.na])
        self.gamma = gamma
        self.alpha = alpha
        self.game = game
        self.na = game.na
        
        """politics ana;ysis parameters"""
        self.learningCounter=0
        self.Time=[0]
        self.State=[[0,0,0,0]]
        
        self.eProfile = EpsilonProfile(1, 0.1)
        self.epsilon = self.eProfile.initial

    def learn(self, episodes, iterations):
        nbSteps = np.zeros(episodes) + iterations
        
        for e in range(episodes):
            state = self.game.reset()
            
            for i in range(iterations):
                action = self.select_action(state)
                next_state, reward, res = self.game.step(action)
                
                # print(f"Episode {e} - Step {i} - Action {action} - Reward {reward} - Terminal {terminal} - Epsilon {self.epsilon}")
                self.updateQ(state, action, reward, next_state)
                
                if res:
                    nbSteps[e] = i + 1  
                    break
                
                state = next_state
                
            self.epsilon = max(self.epsilon - self.eProfile.dec_episode / (episodes - 1.), self.eProfile.final)
            
        # print(f"Machine Learning done using the following parameters : gamma={self.gamma}, alpha={self.alpha}, episodes={episodes}, nbSteps={iterations}")
        

        """saving q values in an npy file"""
        newPath = f"qtable_{self.gamma}_{self.alpha}_{episodes}_{iterations}.npy"
        self.save(newPath)
        
        print(f"Machine Learning finished.")

    def updateQ(self, state : tuple[int, int], action : int, reward : float, next_state : tuple[int, int]):
        # self.Q[state, action]+=self.alpha*(reward+self.gamma*np.max(self.Q[next_state, action])-self.Q[state, action])
        self.Q[state][action]+=self.alpha*(reward+self.gamma*np.max(self.Q[next_state])-self.Q[state][action])

        
        # self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))

        # self.Q[state][action] += self.alpha * (reward + self .gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        
        # print(self.Q[state])
        
        # self.learningCounter+=1
        # self.updateCandS(self.learningCounter, self.Q[state])

    def select_action(self, state : tuple[int, int]):
        if np.random.rand() < self.epsilon:
            nextAction = np.random.randint(0, self.na)
        else:
            nextAction = self.greedy(state)
            
        return nextAction
    
    def save(self, path):
        np.save(path, self.Q)

    def greedy(self, state : tuple[int, int]):
        maximum = np.max(self.Q[state])
        return np.random.choice(np.where(self.Q[state] == maximum)[0])
    
    def load(self, path):
        self.Q = np.load(path)
    
    def updateCandS(self, time, state):
        self.Time.append(time)
        self.State.append(state)