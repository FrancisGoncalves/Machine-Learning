# ------------------------------------------- WSU - CPT_S 570 - HW4 -----------------------------------------
# --------------------------- Q-Learning algorithm with Boltzmann exploration policy -----------------------------------
# ----------------------------------------- Student: Francisco Goncalves ------------------------------------------
#
# This code optimizes the Q-values based on Boltzmann exploration policy
import copy
import numpy as np
from Qlearning import gridWorld_rows, gridWorld_columns, get_next_action_Boltzmann, get_next_location, rewards

print('This may take a while, please wait...\n')
# define training parameters
T = 10  # Temperature
beta = 0.9  # Discount factor
alpha = 0.01  # Learning rate
q_values = np.zeros((gridWorld_rows, gridWorld_columns, 4))
convergence = 1
convergence_threshold = 0.0001
iteration = 0
# Run until it reaches convergence threshold
while convergence > convergence_threshold:
    # get the starting location for this episode
    iteration += 1
    print('Iteration: ', iteration)
    row_index, column_index = 0, 0 # Setting up initial position
    T *= 0.95 # Decaying of T
    # continue taking actions until the agent reaches the goal state
    while [row_index, column_index] != [5, 5]:
        # choose which action to take
        action_index = get_next_action_Boltzmann(row_index, column_index, T)
        # perform the chosen action, and move to the next state
        old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        # print(row_index, column_index)
        # receive the reward for moving to the new state, and calculate the temporal difference
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (beta * np.max(q_values[row_index, column_index])) - old_q_value
        # update the Q-value for the previous state and action pair
        new_q_value = old_q_value + (alpha * temporal_difference)
        old_Q = copy.deepcopy(q_values)
        q_values[old_row_index, old_column_index, action_index] = new_q_value
    convergence = np.linalg.norm(np.matrix.flatten(q_values) - np.matrix.flatten(old_Q))
    # print(convergence)
print('Training complete!')
print(q_values)
