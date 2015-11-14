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
import time

class AsynchronousValueIterationAgent(ValueEstimationAgent):
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
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
          self.values[state] = 0
        global num
        num = 0
        while num < self.iterations:
          for state in states:
            if num < self.iterations:
              if self.mdp.isTerminal(state):
                self.values[state] = 0
                num = num + 1
                continue
              current = -1*float('inf')
              for action in self.mdp.getPossibleActions(state):
                a = self.computeQValueFromValues(state, action)
                if a > current:
                  current = a
              self.values[state] = current
            num += 1

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
        value = 0
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
          # print self.mdp.getReward(state)
          value += probability * (self.mdp.getReward(state) + (self.discount * self.values[nextState]))
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
     
        if not actions:
          return None
        value = -1*float('inf')
        end = None
        for i in actions:
          a = self.computeQValueFromValues(state, i)
          if a > value:
            value = a
            end = i
        return end

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


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
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        #### Compute the predessors for each state
        #### state ----> set of that state's predecessors
        predecessors = {}

        for state in states:
          predecessors[state] = set()

        for state in states:
          if self.mdp.isTerminal(state) == False:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
              for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                if probability > 0:
                  predecessors[nextState].add(state)


        pq = util.PriorityQueue()

        for state in states:
          if self.mdp.isTerminal(state) == False:

            maxValue = -1*float('inf')
            for action in self.mdp.getPossibleActions(state):
              qVal = self.computeQValueFromValues(state, action)
              if qVal > maxValue:
                maxValue = qVal

            diff = abs(self.values[state] - maxValue)
            pq.push(state, -diff)

        for i in range(self.iterations):

          if pq.isEmpty() == False:
            poppedState = pq.pop()
            if self.mdp.isTerminal(poppedState) == False:

              maxValue = -1*float('inf')
              for action in self.mdp.getPossibleActions(poppedState):
                qVal = self.computeQValueFromValues(poppedState, action)
                if qVal > maxValue:
                  maxValue = qVal

              self.values[poppedState] = maxValue

              for predecessor in predecessors[poppedState]:

                maxValue = -1*float('inf')
                for action in self.mdp.getPossibleActions(predecessor):
                  qVal = self.computeQValueFromValues(predecessor, action)
                  if qVal > maxValue:
                    maxValue = qVal

                diff = abs(self.values[predecessor] - maxValue)

                if diff > theta:
                  pq.update(predecessor, -diff)
