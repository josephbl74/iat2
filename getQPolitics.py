import numpy as np
import matplotlib.pyplot as plt

class Analyst():
    def __init__(self):
        # self.Q=self.load(path)
        self.Q=np.zeros([1, 1, 1, 1])
        self.Time=[0]
        self.Rewards=[0]
    
    def displayQ(self):
        print("Value Function:")
        # print(np.max(self.Q, axis=1))
        print(self.Q)
    
    def load(self, path):
        lp = np.load(path)
        self.Q=self.load(lp)
    
    def displayR(self):
        print("Rewards:")
        
        # plot rewards table as a func. of time table
        plt.plot(self.Time, self.Rewards)
        plt.xlabel('Time (simulated)')
        plt.ylabel('Reward (Time)')
        plt.show()
        
    def updateTandR(self, t, r):
        self.Time.append(t)
        self.Rewards.append(r)
        
        