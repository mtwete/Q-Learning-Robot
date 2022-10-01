"""
Matthew Twete

A program to train an agent using Q-learning to pick up "trash" in a small grid world.
"""


#Import libraries
import numpy as np
import matplotlib.pyplot as plt


#Q-Learning algorithm for the simple robby the robot in a grid world. 
#The goal of the algorithm is to learn a strategy for picking up cans in the
#grid world. 
class roboQLearn:
    
    #Class constructor
    def __init__(self):
        #Numpy array representing the grid, initially all empty
        self.grid = np.zeros((10,10))
        #Variable representing the x coordinate in the grid of robby
        self.x = 0
        #Variable representing the y coordinate in the grid of robby
        self.y = 0
        #Number of episodes for training and testing
        self.N = 5000
        #Number of steps allowed per episode
        self.M = 200
        #Eta, the learning rate
        self.eta = 0.2
        #Gamma, the discount rate
        self.gamma = 0.9
        #Epsilon, the probability to select a random action
        self.epsilon = 0.1
        #Array to hold the total reward for episodes during training 
        self.train_ep_rewards = np.zeros(self.N)
        #Array to hold the total reward for episodes during training 
        self.test_ep_rewards = np.zeros(self.N)
        #The Q-matrix
        self.QTable = np.zeros((3,3,3,3,3,5))
        #Contants to represent actions as integers
        self.PICKUP = 0
        self.NORTH = 1
        self.SOUTH = 2
        self.EAST = 3
        self.WEST = 4
        #Constant to represent a wall was sensed, to be used when getting the state
        self.WALL = 2
        #Tuple of possible actions robby can take
        self.actions = (self.PICKUP,self.NORTH,self.SOUTH,self.EAST,self.WEST)
        
        
    #Function to initialize the grid for each episode
    def init_grid(self):
        #Loop over every square in the grid
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                #With a 50% probability place a can in that square
                if (np.random.rand() < 0.5):
                    self.grid[i][j] = 1
    
    
    #Function to initialize robby's starting position for each episode
    def init_robby(self):
        #Randomly pick a x and y coordinate for robby to start in
        self.x = np.random.randint(0, high=10)
        self.y = np.random.randint(0, high=10)
     
        
    #Function to plot the total reward per episode during training
    def plot_train_reward(self):
        #Set up episode values for x-axis
        epochs = np.arange(1,self.N+1, 100)
        #Plot the training rewards per episode every 100th episode
        plt.plot(epochs, self.train_ep_rewards[::100])
        #Add labels and a title
        plt.xlabel('Episode')
        plt.ylabel('Total Reward Per Episode')
        plt.title("Training Reward Plot")
        #Show the plot
        plt.show()
        
        
    #Function to calculate the test performance statistics after training 
    def display_test_results(self):
        #Calculate the test average total reward per episode after training 
        test_ave = np.mean(self.test_ep_rewards)
        #Calculate the test standard deviation of the total reward per episode after training 
        test_std = np.std(self.test_ep_rewards)
        #Print out those values
        print("Test Sum Of Rewards Per Episode Average: ", test_ave)
        print("Test Sum Of Rewards Per Episode Standard Deviation: ", test_std)
    
    
    #Function to get the sensor inputs of the current and surrounding squares
    def get_sensors(self):
        #Sensors tuples are indexed in the following order: current, north, south, east, west
        #0 represents and empty square, 2 represents a wall and 1 represents a can
        
        #Variables to hold the sensor inputs for the current square and the surrounding squares
        #in the cardinal directions
        current = 0
        north = 0
        south = 0
        east = 0
        west = 0
        
        #Check the current then the surrounding squares and set the sensor inputs to the appropriate
        #values
        if (self.grid[self.x][self.y] == 1):
            current = 1
        else:
            current = 0
        #Check the square to the north
        if (self.y - 1 < 0):
            north = self.WALL
        else:
            north = int(self.grid[self.x][self.y - 1])
        #Check the square to the south
        if (self.y + 1 > 9):
            south = self.WALL
        else:
            south = int(self.grid[self.x][self.y + 1])
        #Check the square to the east
        if (self.x + 1 > 9):
            east = self.WALL
        else:
            east = int(self.grid[self.x + 1][self.y])
        #Check the square to the west
        if (self.x - 1 < 0):
            west = self.WALL
        else:
            west = int(self.grid[self.x - 1][self.y])
        #Return a tuple of the sensor inputs
        return (current,north,south,east,west)

    
    #Fucntion to pick the action of robby given the current state. The arguments are:
    #State, tuple containing the current state
    def pick_action(self,state):
        #If probability of epsilon occurs, pick and return a random action
        if (np.random.rand() < self.epsilon):
            return np.random.choice(self.actions)
        #Otherwise, pick the action that has the highest expected reward for that state from the QTable
        return np.argmax(self.QTable[state])
    
    
    #Get the reward for a given action in the current state. The arguments are:
    #action, the action robby chose to do in the current step
    #State, tuple containing the current state
    def get_reward(self,action,state):
        #Check to see if the picked action results in robby bumping into a wall,
        #if so, return a reward of -5
        result = state[action]
        if (result == self.WALL):
            return -5
        #If the action picked is to pick up a can, check to see if there is a can in the current square
        if (action == 0):
            #If there is a can in the current square pick it up, set the square to empty and return a reward 
            #of 10
            if (self.grid[self.x][self.y] == 1):
                self.grid[self.x][self.y] = 0
                return 10
            #Otherwise, the square was empty so return a reward of -1
            else:
                return -1
        #If the action doesn't result in bumping into a wall or attempting to pick up a can, then move robby in the 
        #direction the action took him and return a 0 reward
        #Move north
        elif (action == 1):
            self.y -= 1
        #Move south
        elif (action == 2):
            self.y += 1
        #Move east
        elif (action == 3):
            self.x += 1
        #Move west
        else:
            self.x -= 1
        return 0
            
            
    #Function to train robby using the Q-learning algorithm
    def train(self):
        #Start off epsilon at 0.1
        self.epsilon = 0.1
        #Loop over the number of episodes
        for i in range(self.N):
            #Initialize the grid and robby's starting position each episode
            self.init_grid()
            self.init_robby()
            #Variable to hold the episode total rewards earned
            total_reward = 0
            #Every 50 episodes decrease epsilon by 0.002 until it hits 0 and then 
            #have it stay at 0
            if (i % 50 == 0 and i != 0 and self.epsilon > 0):
                self.epsilon -= 0.002
            #Loop over the number of steps per episode
            for j in range(self.M):
                #Get the sensor inputs (ie state)
                state = self.get_sensors()
                #Pick an action given the state
                action = self.pick_action(state)
                #Get the reward for that action (if there is one)
                reward = self.get_reward(action,state)
                #Get the new state to update the Q-matrix
                new_state = self.get_sensors()
                #Update the Q-matrix given the information robby got for that action
                self.QTable[state][action] = self.QTable[state][action] + self.eta*(reward + self.gamma*np.amax(self.QTable[new_state]) - self.QTable[state][action])
                #Add the reward to the total reward for that episode
                total_reward += reward
            #Save the total reward earned that episode
            self.train_ep_rewards[i] = total_reward
    
    
    #Function to test the Q-learning algorithm's performance after training
    def test(self):
        #Set epsilon to 0.1 for testing
        self.epsilon = 0.1
        #Loop over the number of episodes
        for i in range(self.N):
            #Initialize the grid and robby's starting position each episode
            self.init_grid()
            self.init_robby()
            #Variable to hold the episode total rewards earned
            total_reward = 0
            #Loop over the number of steps per episode
            for j in range(self.M):
                #Get the sensor inputs (ie state)
                state = self.get_sensors()
                #Pick an action given the state
                action = self.pick_action(state)
                #Get the reward for that action (if there is one)
                reward = self.get_reward(action,state)
                #Add the reward to the total reward for that episode
                total_reward += reward
            #Save the total reward earned that episode
            self.test_ep_rewards[i] = total_reward
            
            
#Train the algorithm and then test it and display the results of training and testing
robby = roboQLearn()
robby.train()
robby.test()
robby.display_test_results()
robby.plot_train_reward()

        
                
                


        
    