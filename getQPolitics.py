import numpy as np

class Analyst():
    def __init__(self, path):
        self.Q=self.load(path)
    
    def display(self):
        print("Value Function:")
        # print(np.max(self.Q, axis=1))
        print(self.Q)
    
    def load(self, path):
        return np.load(path)
        
        