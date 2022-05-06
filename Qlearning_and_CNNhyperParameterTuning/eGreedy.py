# ------------------------------------------- WSU - CPT_S 570 - HW4 -----------------------------------------
# --------------------------- Q-Learning algorithm with epsilon-greedy exploration policy --------------------------
# ----------------------------------------- Student: Francisco Goncalves ------------------------------------------
#
# This code optimizes the Q-values based on epsilon-greedy exploration policy

import copy

import numpy as np
from Qlearning import gridWorld_rows, gridWorld_columns, get_next_action_epsilon, get_next_location, rewards

print('This may take a while, please wait...\n')
# define training parameters
epsilon = 0.3  # Probability of the agent taking a random action
beta = 0.9  # Discount factor for future rewards
alpha = 0.01  # Learning rate
q_values = np.zeros((gridWorld_rows, gridWorld_columns, 4))
convergence_threshold = 0.0001
convergence = 1
iteration = 0
# run through 1000 training episodes
while iteration <= 500:
    # get the starting location for this episode
    iteration += 1
    print('Iteration: ', iteration)
    row_index, column_index = 0, 0 # Setup initial position

    # continue taking actions until the agent reaches the goal state
    while [row_index, column_index] != [5, 5]:
        # Choose which action to take
        action_index = get_next_action_epsilon(row_index, column_index, epsilon)
        # Perform the chosen action, and move to the next state
        old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        # Receive the reward for moving to the new state, and calculate the temporal difference
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (beta * np.max(q_values[row_index, column_index])) - old_q_value

        # update the Q-value for the previous state and action pair
        new_q_value = old_q_value + (alpha * temporal_difference)
        old_Q = copy.deepcopy(q_values)
        q_values[old_row_index, old_column_index, action_index] = new_q_value
    # print(convergence)
convergence = np.linalg.norm(np.matrix.flatten(q_values) - np.matrix.flatten(old_Q))
print('Training complete!')
print(q_values)
