# Author: Sophie Zhao


import random as rand
import numpy as np


class QLearner(object):
    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        # Initiate Q table
        self.Q = np.zeros((self.num_states, self.num_actions))

        if dyna > 0:
            # Initiate transition model T :
            # Record the probability that we are in state S and take action A, will end up in S'.
            self.T = np.full((self.num_states, self.num_actions, self.num_states), 0.00001)

            # Initiate T count:
            # Record the total number that we are in state S and take action A, will end up in S'.
            self.Tc = np.zeros((self.num_states, self.num_actions, self.num_states))

            # Initialize reward model R:
            # Record the expected reward if we are in state S and take action A.
            self.R = np.zeros((self.num_states, self.num_actions))

        self.visited_state_action_pairs = []

    def get_action(self, s):
        if np.random.uniform(0.0, 1.0) < self.rar:
            return rand.randint(0, self.num_actions - 1)  # choose the random direction
        else:
            return np.argmax(self.Q[s])

    def querysetstate(self, s):
        '''Update the state without updating the Q-table'''
        self.s = s
        action = self.get_action(s)
        if self.verbose:
            print(f"s = {s}, a = {action}")
        self.a = action
        return action

    def update_T_R(self, s, a, s_prime, r):
        self.Tc[s, a, s_prime] += 1
        # Total count of transitions from (s, a)
        total_count = np.sum(self.Tc[s, a])

        if total_count != 0:
            self.T[s, a, s_prime] = self.Tc[s, a, s_prime] / total_count

        self.R[s, a] = (1 - self.alpha) * self.R[s, a] + self.alpha * r

    def update_Q(self, s, a, s_prime, r):
        r_existing = self.Q[s][a]
        a_later_max_reward = np.argmax(self.Q[s_prime])
        r_later_max = self.Q[s_prime][a_later_max_reward]

        # Update the Q table
        self.Q[s][a] = (1 - self.alpha) * r_existing + self.alpha * (r + self.gamma * r_later_max)

    def query(self, s_prime, r):
        """Update the Q table and return an action """
        # Update Q-table
        self.update_Q(self.s, self.a, s_prime, r)

        if self.dyna > 0:
            # Update model T and R
            self.update_T_R(self.s, self.a, s_prime, r)

            # Dyna-Q updates
            self.visited_state_action_pairs.append((self.s, self.a))
            # self.dyna_q()

            for i in range(self.dyna):
                # s_rand = rand.randint(0, self.num_states - 1)
                # a_rand = rand.randint(0, self.num_actions - 1)

                # Only selects from states it has seen before
                s_rand, a_rand = rand.choice(self.visited_state_action_pairs)
                s_prime_rand = np.argmax(self.T[s_rand, a_rand])
                r_rand = self.R[s_rand, a_rand]

                # Update the Q-table using the simulated experience
                self.update_Q(s_rand, a_rand, s_prime_rand, r_rand)

        self.rar = self.rar * self.radr
        action = self.get_action(s_prime)
        self.a = action
        self.s = s_prime

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        return action

# if __name__ == "__main__":
# q = QLearner(verbose=True)
