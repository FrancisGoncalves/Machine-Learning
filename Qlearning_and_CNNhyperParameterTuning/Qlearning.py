# ------------------------------------------- WSU - CPT_S 570 - HW4 -----------------------------------------
# -------------------------------------------- Q-Learning algorithm -----------------------------------
# ----------------------------------------- Student: Francisco Goncalves ------------------------------------------
#
# This code builds the environment of the Q-learning algorithm

import numpy as np
import random

# Setup the Grid World space
gridWorld_rows = 10
gridWorld_columns = 10

# Actions
# 0 = left, 1 = right, 2 = up, 3 = down
actions = ['left', 'right', 'up', 'down']
q_values = np.zeros((gridWorld_rows, gridWorld_columns, 4))

# Rewards of all states
rewards = np.full((gridWorld_rows, gridWorld_columns), 0.)
rewards[5, 5] = 1.  # set the reward of the goal state

neg_reward = {}
neg_reward[3] = [3]
neg_reward[4] = [5, 6]
neg_reward[5] = [6, 8]
neg_reward[6] = [8]
neg_reward[7] = [3, 5, 6]
neg_reward[8] = [3, 7]

# Set the rewards of the states with negative rewards
for row_index in range(3, 9):
    for column_index in neg_reward[row_index]:
        rewards[row_index, column_index] = -1


# Define an epsilon greedy algorithm that will choose which action to take next
def get_next_action_epsilon(current_row_index, current_column_index, epsilon):
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:  # choose a random action
        return np.argmax(q_values[current_row_index, current_column_index])


# Function to choose the next action based on Boltzmann policy exploration
def get_next_action_Boltzmann(row_index, column_index, T):
    if (row_index == 0 or ([row_index, column_index] in ([3, 1], [3, 2], [3, 6], [3, 7], [3, 8], [8, 4]))) and (
            [row_index, column_index] not in ([0, 0], [0, 9])):
        b = [np.exp(q_values[row_index, column_index, 0] / T) / (
                np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
            q_values[row_index, column_index, 1] / T) +
                np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 1] / T) / (
                     np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
                 q_values[row_index, column_index, 1] / T) +
                     np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 3] / T) / (
                     np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
                 q_values[row_index, column_index, 1] / T) +
                     np.exp(q_values[row_index, column_index, 3] / T))]
        if b[0] == b[1] and b[1] == b[2]:
            a = random.choice([0, 1, 2])
        else:
            a = np.argmax(b)

        if a == 2:
            return a + 1
        else:
            return a
    elif (row_index == 9 or (
            ([row_index, column_index] in ([1, 1], [1, 2], [1, 3], [1, 4], [1, 6], [1, 7], [1, 8])))) and (
            [row_index, column_index] not in ([9, 0], [9, 9])):
        b = [np.exp(q_values[row_index, column_index, 0] / T) / (
                np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
            q_values[row_index, column_index, 1] / T) +
                np.exp(q_values[row_index, column_index, 2] / T)),
             np.exp(q_values[row_index, column_index, 1] / T) / (
                     np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
                 q_values[row_index, column_index, 1] / T) +
                     np.exp(q_values[row_index, column_index, 2] / T)),
             np.exp(q_values[row_index, column_index, 2] / T) / (
                     np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
                 q_values[row_index, column_index, 1] / T) +
                     np.exp(q_values[row_index, column_index, 2] / T))]
        if b[0] == b[1] and b[1] == b[2]:
            a = random.choice([0, 1, 2])
        else:
            a = np.argmax(b)

        return a

    elif (column_index == 0 or ([row_index, column_index] in ([3, 5], [4, 5], [5, 5], [6, 5], [7, 5]))) and (
            [row_index, column_index] not in ([0, 0], [2, 0], [9, 0])):
        b = [np.exp(q_values[row_index, column_index, 1] / T) / (
                np.exp(q_values[row_index, column_index, 1] / T) + np.exp(
            q_values[row_index, column_index, 2] / T) +
                np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 2] / T) / (
                     np.exp(q_values[row_index, column_index, 1] / T) + np.exp(
                 q_values[row_index, column_index, 2] / T) +
                     np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 3] / T) / (
                     np.exp(q_values[row_index, column_index, 1] / T) + np.exp(
                 q_values[row_index, column_index, 2] / T) +
                     np.exp(q_values[row_index, column_index, 3] / T))]
        if b[0] == b[1] and b[1] == b[2]:
            a = random.choice([0, 1, 2])
        else:
            a = np.argmax(b)

        return a + 1
    elif column_index == 9 or ([row_index, column_index] in ([4, 3], [5, 3], [6, 3], [7, 3])) and [row_index,
                                                                                                   column_index] not in (
            [0, 9], [9, 9], [2, 9]):
        b = [np.exp(q_values[row_index, column_index, 0] / T) / (
                np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
            q_values[row_index, column_index, 2] / T) +
                np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 2] / T) / (
                     np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
                 q_values[row_index, column_index, 2] / T) +
                     np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 3] / T) / (
                     np.exp(q_values[row_index, column_index, 0] / T) + np.exp(
                 q_values[row_index, column_index, 2] / T) +
                     np.exp(q_values[row_index, column_index, 3] / T))]
        if (b[0] == b[1]) and (b[1] == b[2]):
            a = random.choice([0, 1, 2])
        else:
            a = np.argmax(b)

        if a > 0:
            return a + 1
        else:
            return a
    elif [row_index, column_index] == [0, 0]:
        b = [np.exp(q_values[row_index, column_index, 1] / T) / (np.exp(q_values[row_index, column_index, 1] / T) +
                                                                 np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 3] / T) / (np.exp(q_values[row_index, column_index, 1] / T) +
                                                                 np.exp(q_values[row_index, column_index, 3] / T))]
        if (b[0] == b[1]):
            a = random.choice([0, 1])
        else:
            a = np.argmax(b)

        if a == 0:
            return a + 1
        else:
            return a + 2
    elif [row_index, column_index] in ([0, 9], [3, 3]):
        b = [np.exp(q_values[row_index, column_index, 0] / T) / (np.exp(q_values[row_index, column_index, 0] / T) +
                                                                 np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 3] / T) / (np.exp(q_values[row_index, column_index, 0] / T) +
                                                                 np.exp(q_values[row_index, column_index, 3] / T))]
        if b[0] == b[1]:
            a = random.choice([0, 1])
        else:
            a = np.argmax(b)

        if a == 1:
            return a + 2
        else:
            return a
    elif [row_index, column_index] == [9, 0]:
        b = [np.exp(q_values[row_index, column_index, 1] / T) / (np.exp(q_values[row_index, column_index, 1] / T) +
                                                                 np.exp(q_values[row_index, column_index, 2] / T)),
             np.exp(q_values[row_index, column_index, 2] / T) / (np.exp(q_values[row_index, column_index, 1] / T) +
                                                                 np.exp(q_values[row_index, column_index, 2] / T))]
        if b[0] == b[1]:
            a = random.choice([0, 1])
        else:
            a = np.argmax(b)

        return a + 1
    elif [row_index, column_index] == [9, 9]:
        b = [np.exp(q_values[row_index, column_index, 0] / T) / (np.exp(q_values[row_index, column_index, 0] / T) +
                                                                 np.exp(q_values[row_index, column_index, 2] / T)),
             np.exp(q_values[row_index, column_index, 2] / T) / (np.exp(q_values[row_index, column_index, 0] / T) +
                                                                 np.exp(q_values[row_index, column_index, 2] / T))]
        if b[0] == b[1]:
            a = random.choice([0, 1])
        else:
            a = np.argmax(b)

        if a == 1:
            return a + 1
        else:
            return a
    elif [row_index, column_index] in ([2, 0], [2, 5], [2, 9]):
        b = [np.exp(q_values[row_index, column_index, 2] / T) / (np.exp(q_values[row_index, column_index, 2] / T) +
                                                                 np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 3] / T) / (np.exp(q_values[row_index, column_index, 2] / T) +
                                                                 np.exp(q_values[row_index, column_index, 3] / T))]
        if b[0] == b[1]:
            a = random.choice([0, 1])
        else:
            a = np.argmax(b)

        return a + 2
    else:
        b = [np.exp(q_values[row_index, column_index, 0] / T) / (np.exp(q_values[row_index, column_index, 0] / T) +
                                                                 np.exp(q_values[row_index, column_index, 1] / T) +
                                                                 np.exp(q_values[row_index, column_index, 2] / T) +
                                                                 np.exp(q_values[row_index, column_index, 3] / T)),
             np.exp(q_values[row_index, column_index, 1] / T) / (np.exp(q_values[row_index, column_index, 0] / T) +
                                                                 np.exp(q_values[row_index, column_index, 1] / T)) +
             np.exp(q_values[row_index, column_index, 2] / T) +
             np.exp(q_values[row_index, column_index, 3] / T),
             np.exp(q_values[row_index, column_index, 2] / T) / (np.exp(q_values[row_index, column_index, 0] / T) +
                                                                 np.exp(q_values[row_index, column_index, 1] / T)) +
             np.exp(q_values[row_index, column_index, 2] / T) +
             np.exp(q_values[row_index, column_index, 3] / T),
             np.exp(q_values[row_index, column_index, 3] / T) / (np.exp(q_values[row_index, column_index, 0] / T) +
                                                                 np.exp(q_values[row_index, column_index, 1] / T) +
                                                                 np.exp(q_values[row_index, column_index, 2] / T) +
                                                                 np.exp(q_values[row_index, column_index, 3] / T))]
        if b[0] == b[1] and b[1] == b[2] and b[2] == b[3]:
            a = random.choice([0, 1, 2, 3])
        else:
            a = np.argmax(b)

        return a


# Function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index != 0 and (
            [current_row_index, current_column_index] not in ([3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [8, 4])):
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < gridWorld_columns - 1 and (
            [current_row_index, current_column_index] not in ([3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [2, 0], [2, 5])):
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < gridWorld_rows - 1 and (
            [current_row_index, current_column_index] not in ([1, 1], [1, 2], [1, 3], [1, 4], [1, 6], [1, 7], [1, 8])):
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0 and (
            [current_row_index, current_column_index] not in ([2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [2, 9])):
        new_column_index -= 1
    return new_row_index, new_column_index
