from copy import deepcopy
import numpy as np

def utility_under_action(mdp, wanted_action, currentState, U_vec):
    sum = 0
    for index, actual_action in enumerate(mdp.actions):
        probability_of_action = mdp.transition_function[wanted_action][index]
        new_state_row, new_state_col = mdp.step(currentState, actual_action)
        sum += probability_of_action * U_vec[new_state_row][new_state_col]
    return sum


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    delta = float('inf')
    max_error = (epsilon*(1-mdp.gamma))/(mdp.gamma)
    U_curr = [row[:] for row in U_init]
    U_prev = [row[:] for row in U_init]
    while delta >= max_error:
        U_prev = [row[:] for row in U_curr]
        delta = 0
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if mdp.board[r][c] == 'WALL':
                    continue
                elif (r, c) in mdp.terminal_states:
                    U_curr[r][c] = float(mdp.board[r][c])
                else:
                    U_curr[r][c] = float(mdp.board[r][c]) + mdp.gamma*max([utility_under_action(mdp, action, (r,c), U_prev) for action in mdp.actions])
                delta = max(abs(U_curr[r][c] - U_prev[r][c]), delta)
    return U_prev


def get_policy(mdp, U):
    policy = [['' for c in range(mdp.num_col)] for r in range(mdp.num_row)]
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if (r,c) in mdp.terminal_states or mdp.board[r][c] == 'WALL':
                policy[r][c] = None
                continue
            desired_action = ''
            curr_max = -float('inf')
            for action in mdp.actions:
                new_utility = utility_under_action(mdp, action, (r,c) ,U)
                if new_utility > curr_max:
                    curr_max = new_utility
                    desired_action = action
            policy[r][c] = desired_action
    return policy

def get_probabilities_vector(mdp, policy, row, col):
    probability_matrix = np.zeros((mdp.num_row, mdp.num_col))
    wanted_action = policy[row][col]
    if wanted_action == None:
        return probability_matrix.flatten()
    for index, actual_action in enumerate(mdp.actions):
        probability_of_action = mdp.transition_function[wanted_action][index]
        new_state_row, new_state_col = mdp.step((row, col), actual_action)
        probability_matrix[new_state_row][new_state_col] += probability_of_action
    return probability_matrix.flatten()

def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    n = mdp.num_row*mdp.num_col
    I = np.identity(n)
    P = np.zeros((n,n))
    for i in range(n):
        P[i] = get_probabilities_vector(mdp, policy, i//mdp.num_col, i%mdp.num_col)
    R = np.zeros(n)
    for i in range(n):
        if mdp.board[i//mdp.num_col][i%mdp.num_col] == 'WALL':
            R[i] = 0
        else:
            R[i] = mdp.board[i//mdp.num_col][i%mdp.num_col]
    return np.dot(np.linalg.inv(I-mdp.gamma*P), R).reshape((mdp.num_row, mdp.num_col))
    


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    policy_table_curr = [row[:] for row in policy_init]
    unchanged = False
    while not unchanged:
        U = policy_evaluation(mdp, policy_table_curr) 
        policy_table_prev = [row[:] for row in policy_table_curr]
        unchanged = True
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if (r,c) in mdp.terminal_states or mdp.board[r][c] == 'WALL':
                    continue
                prev_action = policy_table_prev[r][c]
                prev_value = utility_under_action(mdp, prev_action, (r,c), U)
                for actual_action in mdp.actions:
                    new_value = utility_under_action(mdp, actual_action, (r,c), U)
                    if new_value > prev_value:
                        policy_table_curr[r][c] = actual_action
                        unchanged = False
    return policy_table_curr


"""For this functions, you can import what ever you want """

# if 2 numbersd are epsilon from each other, they are equal for this function
def get_all_policies(mdp, U, epsilon = 10**(-2)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #
    move_to_arrow = {"UP" : '↑', "RIGHT" : '→', "LEFT" : '←', "DOWN" : '↓'}
    policies = [['' for c in range(mdp.num_col)] for r in range(mdp.num_row)]
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            for action in mdp.actions:
                if (abs(utility_under_action(mdp, action, (r,c), U)- U[r][c]) <= epsilon):
                    policies[r][c] += move_to_arrow[action]
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            print ("| " + policies[r][c] + " "*(5-len(policies[r][c])) + "|", end='')
        print("")
    return policies


def convert_board(mdp, newR):
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if not ((r,c) in mdp.terminal_states or mdp.board[r][c] == 'WALL'):
                mdp.board[r][c] = newR
            

def get_policy_for_different_rewards(mdp, accuracy=0.01):  # You can add more input parameters as needed
    R = -5
    convert_board(mdp, R)
    cur_directions = get_policy(mdp,value_iteration(mdp, [[0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]))
    while R < 5:
        prev_R = R
        init_directions = [row[:] for row in cur_directions]
        while all(row1 == row2 for row1, row2 in zip(init_directions, cur_directions)):
            R += accuracy
            if R >=5:
                break
            convert_board(mdp, R)
            cur_directions = get_policy(mdp, value_iteration(mdp, [[0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]))
        mdp.print_policy(cur_directions)
        if prev_R == -5:
            print("R(s) < " + str(R))
        elif R >= 5:
            print("R(s) > " + str(prev_R))
        else:
            print(str(prev_R) + " < R(s) < " + str(R))
        print("")
        print("-------------------------------------------")

            
