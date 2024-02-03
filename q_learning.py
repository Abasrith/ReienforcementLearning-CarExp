import argparse
import numpy as np

from environment import MountainCar, GridWorld

from typing import Union, Tuple, Optional # for type annotations

import matplotlib.pyplot as plt

"""
class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *NOT* have the bias term 
folded in.
"""

def set_seed(seed: int):
    '''
    Sets the numpy random seed.
    '''
    np.random.seed(seed)


def round_output(places: int):
    '''
    Decorator to round output of a function to certain 
    number of decimal places. You do not need to know how this works.
    '''
    def wrapper(fn):
        def wrapped_fn(*args, **kwargs):
            return np.round(fn(*args, **kwargs), places)
        return wrapped_fn
    return wrapper


def parse_args() -> Tuple[str, str, str, str, int, int, float, float, float]:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
    
    Usage:
        (env_type, mode, weight_out, returns_out, 
         episodes, max_iterations, epsilon, gamma, lr) = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return (args.env, args.mode, args.weight_out, args.returns_out, 
            args.episodes, args.max_iterations, 
            args.epsilon, args.gamma, args.learning_rate)


@round_output(5) # DON'T DELETE THIS LINE
def Q(W: np.ndarray, state: np.ndarray, 
      action: Optional[int] = None) -> Union[float, np.ndarray]:
    '''
    Helper function to compute Q-values for function-approximation 
    Q-learning.

    Parameters:
        W     (np.ndarray): Weight matrix with folded-in bias with 
                            shape (action_space, state_space+1).
        state (np.ndarray): State encoded as vector with shape (state_space,).
        action       (int): Action taken. Satisfies 0 <= action < action_space.

    Returns:
        If action argument was provided, returns float Q(state, action).
        Otherwise, returns array of Q-values for all actions from state,
        [Q(state, a_0), Q(state, a_1), Q(state, a_2)...] for all a_i.
    '''
    state = np.append(1, state)
    state = np.expand_dims(state, axis=1)
    Q_values = np.dot(W, state)
    
    if action != None:
        return Q_values[action]
    else:
        return Q_values


if __name__ == "__main__":
    set_seed(10301) # DON'T DELETE THIS

    # Read in arguments
    (env_type, mode, weight_out, returns_out, 
     episodes, max_iterations, epsilon, gamma, lr) = parse_args()

    # Create environment
    if env_type == "mc":
        env = MountainCar(mode, False)
    elif env_type == "gw":
        env = GridWorld(mode, False)
    else: 
        raise Exception(f"Invalid environment type {env_type}")

    W = np.zeros((env.action_space, env.state_space+1))
    reward_list = []
    
    rolling_mean = 0
    rolling_mean_list = []
    rolling_mean_window = []

    for episode in range(episodes):

        current_state = env.reset()
        reward_accumulator = 0
         

        for iteration in range(max_iterations):

            if np.random.rand() < epsilon:
                # random action
                current_action = np.random.randint(env.action_space)
            else:
                # best possible action
                q_values = Q(W, current_state)
                current_action = np.argmax(q_values)

            next_state, reward, terminate_flag = env.step(current_action)

            q_delta = np.zeros((env.action_space, env.state_space+1))
            temp_state = np.insert(current_state, 0, 1)
            q_delta[current_action, :] = temp_state
            
            W = W - lr*(Q(W, current_state, current_action)-(reward+(gamma*np.max(Q(W,next_state)))))*q_delta
            current_state = next_state
            reward_accumulator += reward
            
            if(terminate_flag == True):
                break
        
        reward_list.append(reward_accumulator)
        
        rolling_mean_window.insert(-1, reward_accumulator)
        if(len(rolling_mean_window) > 24):
           rolling_mean_list.append(sum(rolling_mean_window)/len(rolling_mean_window))
           del rolling_mean_window[0]

        # if((episode+1)%25 == 0):
        #     rolling_mean += reward_accumulator
        #     rolling_mean = rolling_mean/25
        #     rolling_mean_list.append(rolling_mean)
        #     rolling_mean = 0
        # else:
        #     rolling_mean += reward_accumulator
            
    episodes_range = range(1, episodes+1, 1)
    episodes_25_fact_range = range(25,episodes+1, 1)
    episodes_all = list(episodes_range)
    episodes_25_fact = list(episodes_25_fact_range)
    plt.plot(episodes_all, reward_list, label='Total reward Vs episodes', color = 'r')
    plt.plot(episodes_25_fact, rolling_mean_list, label='Mean reward Vs episodes', color = 'b')
    plt.title("Analysis of reward Vs episodes with tile features on mountain car")
    plt.xlabel('Episodes #')
    plt.ylabel('Reward Values')
    plt.legend()
    plt.show()

    with open(weight_out, 'w') as weightsFile:
        np.savetxt(weightsFile, W, fmt="%.18e", delimiter=" ")
    
    with open(returns_out, 'w') as returnsFile:
        np.savetxt(returnsFile, reward_list, fmt="%.18e", delimiter=" ")
