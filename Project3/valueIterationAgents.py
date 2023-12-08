# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # I run value iteration for self.iterations number of times
        for iter in range(self.iterations):

            prev_values = self.values.copy() # store all the previous values somewhere

            # I need to update all the states during value iterations
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    largest_value = []
                    for action in self.mdp.getPossibleActions(state):
                        value_kp1 = 0
                        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                            # This contains a next state of the action from current state
                            next_state = transition[0]
                            # This contains probability to end up on that next state if taking the action
                            probability = transition[1]
                            value_kp1 = value_kp1 + probability*(self.mdp.getReward(state, action, next_state) +
                                                                 (self.discount*prev_values[next_state]))
                        largest_value.append(value_kp1)
                    max_value = max(largest_value) # the max value over the possible actions for the current state
                    self.values[state] = max_value # update the current state



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        next_state_prob_pairs = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0

        for transition in next_state_prob_pairs:
            next_state = transition[0] # This contains a next state of the action from current state
            probability = transition[1] # This contains probability to end up on that next state if taking the action
            q_value = q_value + probability*(self.mdp.getReward(state, action, next_state) +
                                             (self.discount*self.values[next_state]))

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # since getPolicy is simply self.computeActionFromValues(state)
        # this should return the policy

        # Terminal state has no legal actions, so return None.
        if self.mdp.isTerminal(state):
            return None

        action_dict = util.Counter()
        # The value function only accepts a state as input, but Q-value function accepts state, action pairs
        # And since my computeQValueFromValues computes the Q-value using the values currently stored in self.values
        # I will need to use my Q-values to determine the best action
        for action in self.mdp.getPossibleActions(state):
            q_val = self.computeQValueFromValues(state, action)
            action_dict[action] = q_val

        best_action = action_dict.argMax()

        return best_action

    def getPolicy(self, state):
        """
            What is the best action to take in the state. Note that because
            we might want to explore, this might not coincide with getAction
            Concretely, this is given by

            policy(s) = arg_max_{a in actions} Q(s,a)

            If many actions achieve the maximal Q-value,
            it doesn't matter which is selected.
        """
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

