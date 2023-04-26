import numpy as np
from game.SpaceInvaders import SpaceInvaders

class EpsilonProfile:
    def __init__(self, initial=1., final=0., dec_episode=1., dec_step=0.):
        self.initial = initial          # initial epsilon in epsilon-greedy
        self.final = final              # final epsilon in epsilon-greedy
        self.dec_episode = dec_episode  # amount of decrement of epsilon in each episode is dec_episode / (number of episodes - 1)
        self.dec_step = dec_step        # amount of decrement of epsilon in each step

class QAgent():
    def __init__(self, game: SpaceInvaders, gamma: float=1, alpha: float = 0.2):
        self.Q = np.zeros([41, 41, 31, game.na])

        self.game = game
        self.na = game.na

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = EpsilonProfile(1, 0.1)
        self.epsilon = self.eps_profile.initial

    def learn(self, env, nbEpisodes=1000, maxSteps=200):
        nbSteps = np.zeros(nbEpisodes) + maxSteps
        
        for episode in range(nbEpisodes):
            state = env.reset()
            
            for step in range(maxSteps):
                action = self.select_action(state)
                next_state, reward, terminal = env.step(action)
                
                print(f"Episode {episode} - Step {step} - Action {action} - Reward {reward} - Terminal {terminal} - Epsilon {self.epsilon}")
                self.updateQ(state, action, reward, next_state)
                
                if terminal:
                    nbSteps[episode] = step + 1  
                    break
                
                state = next_state
                
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (nbEpisodes - 1.), self.eps_profile.final)
            
        print(f"Machine Learning done using the following parameters : gamma={self.gamma}, alpha={self.alpha}, episodes={nbEpisodes}, nbSteps={maxSteps}")
        unique_path = f"qtable_{self.gamma}_{self.alpha}_{nbEpisodes}_{maxSteps}.npy"
        self.save(unique_path)

    def updateQ(self, state : tuple[int, int], action : int, reward : float, next_state : tuple[int, int]):
        new_q_value = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
        self.Q[state][action] = new_q_value

    def select_action(self, state : tuple[int, int]):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.na)
            
        else:
            action = self.select_greedy_action(state)
            
        return action

    def select_greedy_action(self, state : tuple[int, int]):
        mx = np.max(self.Q[state])
        return np.random.choice(np.where(self.Q[state] == mx)[0])
    
    def save(self, path):
        np.save(path, self.Q)
    
    def load(self, path):
        self.Q = np.load(path)