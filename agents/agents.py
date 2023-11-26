from agents.TileCoding import *
import numpy as np

class ValueFunction():
    def __init__(self, alpha=0.01,numOfTilings=8,maxSize=2048,actions=range(1,8),dimsMax=[1,8]):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings
        self.actions = actions
        # divide step size equally to each tiling
        self.alpha = alpha / numOfTilings  # learning rate for each tile

        self.hashTable = IHT(maxSize)

        # weight for each tile
        self.weights = np.zeros(maxSize)

        # state dimensions need scaling to satisfy the tile software
        self.dimScale = len(dimsMax) * [0]
        for i in range(len(dimsMax)):
            self.dimScale[i] = self.numOfTilings / dimsMax[i]

    # get indices of active tiles for given state and action
    def getActiveTiles(self, state, action):
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [ds * s for ds,s in zip(self.dimScale,state)],
                            [action])
        return activeTiles

    # estimate the value of given state and action
    def value(self, state, action):
        d1, d2 = state
        activeTiles = self.getActiveTiles([d1, d2], action)
        return np.sum(self.weights[activeTiles])

    # learn with given state, action and target
    def update(self, state, action, delta):
        d1, d2 = state
        activeTiles = self.getActiveTiles([d1, d2], action)

        delta *= self.alpha
        for activeTile in activeTiles:
            self.weights[activeTile] += delta

    def stateValue(self, state):
        if state[0] == 0:
            # no server available
            return self.value(state, 0)
        values = [self.value(state, a) for a in self.actions]
        return max(values)

class ServiceAgent():
    def __init__(self,epsilon=0.3,alpha=0.01,beta=0.01,actions=range(1,8),dimsMax=[1,8],random=False,DEBUG=False):
        self.DEBUG = DEBUG
        self.actions = actions
        self.epsilon = epsilon
        self.beta = beta
        self.vf = ValueFunction(alpha=alpha,maxSize=4096,dimsMax=dimsMax)
        self.reset_environment()
        self.random = random
        if self.DEBUG:
            print("Current state S = "+str(self.state))
            print("Taking action A = "+str(self.action))

    def reset_environment(self):
        self.state = [0,1]
        self.action = self.chooseAction(self.state)
        self.avg_reward = 0
        self.rewards = []

    def chooseAction(self,state):
        if np.random.uniform(0, 1) <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            values = {}
            for a in self.actions:
                v = self.vf.value(state,a)
                values[a] = v
            action = np.random.choice([k for k, v in values.items() if v == max(values.values())])
        return action

    def step(self,new_state,metrics,goals):
        #if metrics[0] == 0:
        #    return new_state[1]
        reward = self.reward(metrics,goals)
        self.rewards.append(reward)
        if self.random:
            return np.random.choice(self.actions)
        new_action = self.chooseAction(new_state)
        if self.DEBUG:
            print("New state S = "+str(new_state))
            print("Reward: "+str(reward))
            print("New action A' = "+str(new_action))
        delta = reward - self.avg_reward + self.vf.value(new_state, new_action) - self.vf.value(self.state, self.action)
        self.avg_reward += self.beta*delta
        self.vf.update(self.state, self.action, delta)
        self.state = new_state
        self.action = new_action
        if self.DEBUG:
            print("\n##################################################\n")
            print("Current state S = "+str(self.state))
            print("Taking action A = "+str(self.action))
        return self.action

    def reward(self,metrics,goals):
        #if goals[0] > metrics[0]:
        #    return 1
        #return np.exp(-2 * (metrics[0]-goals[0]) / goals[0])
        #return 1/metrics[0]#(9 - metrics[1]) / metrics[0]
        #print("Reward: "+str(metrics[2]))
        return metrics[2]

class VCG_allocator():
    def knapsack_allocation(self,total_amount,amounts,bids):
        allocation = amounts.copy()
        iterator = [x for _,x in sorted(zip(bids,list(range(len(bids)))),reverse=True)]
        for i in iterator:
            allocation[i] = min(total_amount,amounts[i])
            total_amount -= allocation[i]
        return allocation

    def social_welfare(self,allocation,bids):
        welfare = 0
        for a,b in zip(allocation,bids):
            welfare += a*b
        return welfare

    def allocate(self,total_amount,amounts,bids):
        # Allocation
        allocation = self.knapsack_allocation(total_amount,amounts,bids)

        # Payments calculation
        payments = []
        for i in range(len(amounts)):
            amounts2 = amounts.copy() # amounts without bidder i
            del amounts2[i]
            bids2 = bids.copy() # bids without bidder i
            del bids2[i]
            allocation2 = self.knapsack_allocation(total_amount,amounts2,bids2) # allocation without bidder i participating
            allocation_ = allocation.copy() # allocation with bidder i participating, without i
            del allocation_[i]
            payments.append(self.social_welfare(allocation2,bids2) - self.social_welfare(allocation_,bids2))

        return allocation,payments