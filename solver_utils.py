#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Callable, List

import toh_mdp as tm

def value_iteration(
        mdp: tm.TohMdp, v_table: tm.VTable
) -> Tuple[tm.VTable, tm.QTable, float]:
    """Computes one step of value iteration.

    Hint 1: Since the terminal state will always have value 0 since
    initialization, you only need to update values for nonterminal states.

    Hint 2: It might be easier to first populate the Q-value table.

    Args:
        mdp: the MDP definition.
        v_table: Value table from the previous iteration.

    Returns:
        new_v_table: tm.VTable
            New value table after one step of value iteration.
        q_table: tm.QTable
            New Q-value table after one step of value iteration.
        max_delta: float
            Maximum absolute value difference for all value updates, i.e.,
            max_s |V_k(s) - V_k+1(s)|.
    """
    new_v_table: tm.VTable = v_table.copy()
    q_table: tm.QTable = {}
    # noinspection PyUnusedLocal
    max_delta = 0.0
    # *** BEGIN OF YOUR CODE ***
    for state in mdp.nonterminal_states:
        new_v = float("-inf")

        for action in mdp.actions:
            new_q = 0.0

            for next_state in mdp.all_states:
                # Bellman Equations
                t_value = mdp.transition(state, action, next_state)
                reward = mdp.reward(state, action, next_state)

                new_q += t_value * (reward + (mdp.config.gamma * v_table[next_state]))

            new_v = max(new_v, new_q)
            q_table[(state, action)] = new_q

        # Update the value table for the current state
        new_v_table[state] = new_v
        max_delta = max(max_delta, abs(new_v - v_table[state]))

    # ***  END OF YOUR CODE  ***
    return new_v_table, q_table, max_delta

def extract_policy(
        mdp: tm.TohMdp, q_table: tm.QTable
) -> tm.Policy:
    """Extract policy mapping from Q-value table.

    Remember that no action is available from the terminal state, so the
    extracted policy only needs to have all the nonterminal states (can be
    accessed by mdp.nonterminal_states) as keys.

    Args:
        mdp: the MDP definition.
        q_table: Q-Value table to extract policy from.

    Returns:
        policy: tm.Policy
            A Policy maps nonterminal states to actions.
    """
    policy = {}

    # for each nonterminal state
    # find the best action by comparing the current possible actions of a state to
    # find the one with the highest q_value
    for state in mdp.nonterminal_states:
        policy[state] = max(mdp.actions, key=lambda action: q_table.get((state, action), 0))

    return policy


def q_update(
        mdp: tm.TohMdp, q_table: tm.QTable,
        transition: Tuple[tm.TohState, tm.TohAction, float, tm.TohState],
        alpha: float) -> None:
    """Perform a Q-update based on a (S, A, R, S') transition.

    Update the relevant entries in the given q_update based on the given
    (S, A, R, S') transition and alpha value.

    Args:
        mdp: the MDP definition.
        q_table: the Q-Value table to be updated.
        transition: A (S, A, R, S') tuple representing the agent transition.
        alpha: alpha value (i.e., learning rate) for the Q-Value update.
    """
    state, action, reward, next_state = transition
    # *** BEGIN OF YOUR CODE ***

    # access curr q_value for the state
    current_q = q_table[state, action]

    max_next_q = float('-inf')

    # Calculate max q-value for next state with all possible actions
    for pos_action in mdp.actions:
        max_next_q = max(q_table.get((next_state, pos_action), 0), max_next_q)

    # find new q
    new_q = current_q + alpha * (reward + mdp.config.gamma * max_next_q - current_q)

    # update table
    q_table[state, action] = new_q


def extract_v_table(mdp: tm.TohMdp, q_table: tm.QTable) -> tm.VTable:
    """Extract the value table from the Q-Value table.

    Args:
        mdp: the MDP definition.
        q_table: the Q-Value table to extract values from.

    Returns:
        v_table: tm.VTable
            The extracted value table.
    """
    # *** BEGIN OF YOUR CODE ***
    v_table: tm.VTable = {} # new v table

    # all we need to do is look at all state, all possible actions, and get the highest
    # possible q_value (the optimal decision at state)
    for state in mdp.all_states:
        maxval = float("-inf")

        for action in mdp.actions:
            try:
                # at times the key doesn't exist
                q = q_table[(state, action)]
                maxval = max(q, maxval)
            except: pass

        v_table[state] = maxval

    return v_table

def choose_next_action(
        mdp: tm.TohMdp, state: tm.TohState, epsilon: float, q_table: tm.QTable,
        epsilon_greedy: Callable[[List[tm.TohAction], float], tm.TohAction]
) -> tm.TohAction:
    """Use the epsilon greedy function to pick the next action.

    You can assume that the passed in state is neither the terminal state nor
    any goal state.

    You can think of the epsilon greedy function passed in having the following
    definition:

    def epsilon_greedy(best_actions, epsilon):
        # selects one of the best actions with probability 1-epsilon,
        # selects a random action with probability epsilon
        ...

    See the concrete definition in QLearningSolver.epsilon_greedy.

    Args:
        mdp: the MDP definition.
        state: the current MDP state.
        epsilon: epsilon value in epsilon greedy.
        q_table: the current Q-value table.
        epsilon_greedy: a function that performs the epsilon

    Returns:
        action: tm.TohAction
            The chosen action.
    """
    # *** BEGIN OF YOUR CODE ***
    # epsilon == willingness to try new routes
    # (1 - epsilon) == willingness to play it safe
    state_keys = [key for key in q_table.keys() if key[0] == state] # state_keys for current state
    q_values = [q_table[key] for key in state_keys] # q values for each state
    max_q_value = max(q_values) # max q value for state

    # Extract all actions with the maximum Q-value
    best_moves = [key[1] for key in state_keys if q_table[key] == max_q_value]

    e = epsilon_greedy(best_moves, epsilon)
    

    return e



def custom_epsilon(n_step: int) -> float:
    """Calculates the epsilon value for the nth Q learning step.

    Define a function for epsilon based on `n_step`.

    Args:
        n_step: the nth step for which the epsilon value will be used.

    Returns:
        epsilon: float
            epsilon value when choosing the nth step.
    """
    # *** BEGIN OF YOUR CODE ***

    # idk why we need to add n_step here because the GLIE slides only talks about the decay
    # func (1.0 / n_step) but in his slide, the exploration func = u + k/n and i have no
    # idea what u is so i just add the one variable i have access in this function and it works???
    return n_step + 1.0 / n_step


def custom_alpha(n_step: int) -> float:
    """Calculates the alpha value for the nth Q learning step.

    Define a function for alpha based on `n_step`.

    Args:
        n_step: the nth update for which the alpha value will be used.

    Returns:
        alpha: float
            alpha value when performing the nth Q update.
    """
    # *** BEGIN OF YOUR CODE ***
    alpha = 1.0 / n_step if n_step > 0 else 1.0  # To avoid division by zero for the first step
    return alpha
