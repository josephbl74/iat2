import numpy as np
import matplotlib.pyplot as plt

class Analyst():
    def __init__(self):
        # self.Q=self.load(path)
        self.Q=np.zeros([1, 1, 1, 1])
        self.Time=[0]
        self.Rewards=[0]
        
    """Q global"""
    
    def displayQ(self):
        print("Value Function:")
        # print(np.max(self.Q, axis=1))
        print(self.Q)
    
    def load(self, path):
        # lp = np.load(path)
        self.Q = np.load(path)
        # self.Q=self.load(lp)
        
    """rewards"""
    
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
        
    """Q[state][action]"""
    
    def displayQS(self, time, state):
        print("Q[state]:")
        # print(self.Q[state][action])
        
        '''plot all actions evaluation' evolution as a func. of time'''
        # plt.plot(time, state)
        # plt.xlabel('Learning Time (simulated)')
        # plt.ylabel('Actions evaluation')
        # plt.show()
        
        # plt.style.use('_mpl-gallery')




        # make data
        # x=np.arange(0, len(time), 2)
        # a1y=[]
        # a2y=[]
        # a3y=[]
        # a4y=[]
        
        # for s in state:
        #     a1y.append(s[0])
        #     a2y.append(s[1])
        #     a3y.append(s[2])
        #     a4y.append(s[3])
        # y = np.vstack([a1y,a2y,a3y,a4y])

        # # plot
        # fig, ax = plt.subplots()

        # ax.stackplot(x, y)

        # ax.set(xlim=(0, 100), xticks=np.arange(1, 1000),
        #     ylim=(0, 10), yticks=np.arange(1, 1000))

        # plt.show()
        
        
        
        
        
        
        
        data = np.array(state) 

        plt.plot(np.arange(1, 5), data.transpose())
        # plt.plot(np.arange(1, 2), data.transpose())
        plt.show()
        
        
        
        
        # print(state)
        # print("======================================")
        # print(data) -> better
        
                
        
        
        