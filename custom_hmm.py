
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
from tqdm import tqdm
from scipy.special import logsumexp
from scipy.stats import poisson





class custom_PoissonHMM:
    
    def __init__(self, A, B, pi, eps=1e-12):
        '''
        Initialize the HMM with the transition matrix (A), emission probability matrix (B) and prior probability distribution (pi)
        '''
        self.eps = eps

        self.A = np.array(A)
        self.log_A = np.log(A + self.eps)
        self.log_A[self.A == 0] = -np.inf # setting any zero entries in A to -inf in log-space

        self.B = np.array(B)

        self.pi = np.array(pi)
        self.log_pi = np.log(self.pi)

        self.N = self.A.shape[0]

        self.obs = None
        self.T = 0

        self.log_emissions = None


    def generate_samples(self, Nsamples):
        '''generate a random markov chain from the transition matrix A and sample obervations from each hidden state'''

        # initialize the hidden states
        hidden_states = np.zeros(Nsamples, dtype=int)
        hidden_states[0] = np.random.choice(self.N, p = self.pi)

        # initialize the obervations
        obervations = np.zeros((Nsamples, self.B.shape[1]))
        obervations[0] = np.random.poisson(lam = self.B[hidden_states[0]])

        for t in range(1, Nsamples):
            hidden_states[t] = np.random.choice(self.N, p=self.A[hidden_states[t-1]])
            obervations[t] = np.random.poisson(lam = self.B[hidden_states[t]])

        return hidden_states, obervations


    def compute_emissions(self):
        '''compute the emission probabilities of the observations given the emission probability matrix (B)'''

        self.log_emissions_by_dim = poisson.logpmf(
            self.obs[:, None, :],
            self.B[None, :, :]
        )

        self.log_emissions = self.log_emissions_by_dim.sum(axis=2)

    
    def set_observations(self, obs):

        self.obs = np.asarray(obs)
        self.T = len(obs)

        self.compute_emissions()
    

    def forward_probability(self):
        '''
        computes the logarithm of the forward probabilities (alpha) of the observations given the hmm
        '''

        # initialize the forward probabilities
        log_alpha = np.zeros((self.T, self.N))
        log_alpha[0] = (
            self.log_pi
            + self.log_emissions[0]
        )

        # run recursion to solve the rest
        for t in range(1, self.T):

            # compute the next forward probability
            log_alpha[t] = (
                logsumexp(
                    log_alpha[t-1][:,None]
                    + self.log_A, axis = 0
                )
                + self.log_emissions[t]
            )

        return log_alpha
    
    
    def backward_probability(self):
        '''
        compute the logarithm of the backward probabilities (beta) of the observations given the hmm
        '''

        # initialize the backward probabilities
        log_beta = np.zeros((self.T, self.N))
        log_beta[-1] = 0.0 # log(1) = n0

        for t in range(self.T - 2, -1, -1):
            
            # compute the previous backward probability
            log_beta[t] = logsumexp(
                self.log_emissions[t + 1][None, :]
                + self.log_A
                + log_beta[t + 1][None, :],
                axis = 1
            ) 
        
        return log_beta
    

    def viterbi(self):
        '''
        using the viterbi algorithm in log-space to find the latent variables which maximize the probability of the emission sequence
        '''

        # initialize the viterbi and the backpointers
        log_viterbi = np.zeros((self.T, self.N))
        backpointers = np.zeros((self.T, self.N), dtype=int)

        log_viterbi[0] = (
            self.log_pi 
            + self.log_emissions[0]
        )
        

        # run the recursion 
        for t in range(1, self.T):
            log_values = np.array(
                log_viterbi[t - 1][:, None]
                + self.log_A
            )
            
            log_viterbi[t] = np.max(log_values, axis = 0) + self.log_emissions[t]
            backpointers[t] = np.argmax(log_values, axis = 0)

        # inbitialize the most likely states
        states = np.zeros(self.T, dtype = int)
        states[-1] = np.argmax(log_viterbi[-1])

        # loop backwards over time and use the backpointers to find the most likely state sequence
        for t in range(self.T - 1, 0, -1):
            states[t - 1] = backpointers[t, states[t]]

        return states
    

    def compute_log_xi(self, log_alpha, log_beta):

        log_likelihood = logsumexp(log_alpha[-1])  # scalar

        log_xi = (
            log_alpha[:-1, :, None]          # (T-1) x N x 1
            + self.log_A[None, :, :]         # 1 x N x N
            + self.log_emissions[1:, None, :] # (T-1) x 1 x N
            + log_beta[1:, None, :]          # (T-1) x 1 x N
            - log_likelihood
        )

        # normalize
        log_xi -= logsumexp(
            log_xi,
            axis=(1,2),
            keepdims=True
        )

        return log_xi # (T-1) x N x N


    def compute_log_gamma(self, log_alpha, log_beta):
        

        log_gamma = (
            log_alpha       # T x N
            + log_beta      # T x N
        )

        # Normalize
        log_gamma -= logsumexp(
            log_gamma,
            axis=1,
            keepdims=True
        )

        return log_gamma # T x N
    

    def forward_backward(self, transition_update_mask = None, save_dir = None, use_cloned_emissions = False):
        '''using the forward backward algorithm to update the transition matrix (A) and the emissions matrix (B) using the forward and backwards probabilities'''

        #---------------------------Expectation----------------------------------

        # compute the forward and reverse probabilities
        self.log_alpha = self.forward_probability()   # T x N
        self.log_beta = self.backward_probability()   # T x N


        # compuite xi
        log_xi = self.compute_log_xi(self.log_alpha, self.log_beta) # (T-1) x N x N

        if transition_update_mask is not None:
            log_xi = np.where(
                transition_update_mask,
                log_xi,
                -np.inf
            )

        # compute gamma
        log_gamma = self.compute_log_gamma(self.log_alpha, self.log_beta) # T x N

        if save_dir is not None:
            # plot the xi values as a heatmap
            avg_log_xi = logsumexp(log_xi, axis=0) - np.log(log_xi.shape[0]) # N x N
            avg_xi = np.exp(avg_log_xi)
            plt.figure(figsize=(22, 20))
            sns.heatmap(avg_xi, annot=True, fmt=".1e", cmap="Greys" )
            plt.title("Average Xi Values (Transition Probabilities)")
            plt.xlabel("Next State")
            plt.ylabel("Current State")
            plt.savefig(os.path.join(save_dir, "average_xi_heatmap.png"))
            plt.close()



        #---------------------------Maximization---------------------------------

        # numerator
        log_sum_xi = logsumexp(log_xi, axis = 0)      # N x N

        # denominator
        log_row_norm = logsumexp(log_sum_xi, axis=1)  # N

        log_A_hat = (
            log_sum_xi  # N x N
            - log_row_norm[:, None] # N x 1
        )


        A_hat = np.exp(log_A_hat)

        # explicitly set any zero entries in the transition matrix to zero to avoid numerical issues with very small probabilities
        if transition_update_mask is not None:
            A_hat = np.where(
                transition_update_mask,
                A_hat,
                0.0
            )


        gamma = np.exp(log_gamma)

        if use_cloned_emissions:\
        
            nSequenceStates = (self.N - 1) // 2
            B_hat = np.zeros_like(self.B)
            mu_hat = np.zeros((nSequenceStates, self.B.shape[1]))

            # solving for the clone emissions states
            for i in range(nSequenceStates):
                w = gamma[:, i + 1] + gamma[:, nSequenceStates + i + 1]

                mu_hat[i] = (
                    w[:, None] * self.obs
                ).sum(axis=0) / w.sum()

                B_hat[i + 1] = mu_hat[i]
                B_hat[nSequenceStates + i + 1] = mu_hat[i]


            # estimating the base state
            B_hat[0] = (
                gamma[:, 0][:, None] * self.obs
            ).sum(axis=0) / gamma[:, 0].sum()

                
        else:
            B_hat = (
                gamma.T @ self.obs
                / gamma.sum(axis=0)[:, None]
            ) # N x D

            
        return A_hat, B_hat
    

    def fit_em(self, Niters, use_cloned_emissions = False, transition_update_mask = None, save_dir = None):
        '''fit the transition and emissions matrices using expectation maximization'''

        llhs = np.zeros(Niters + 1)
        llhs[0] = logsumexp(self.forward_probability()[-1])
        for iter in tqdm(range(Niters)):

            if save_dir is not None:
                current_save_dir = os.path.join(save_dir, f"iteration_{iter}")
                os.makedirs(current_save_dir, exist_ok=True)
            else:
                current_save_dir = None


            # run the forward backwards pass
            A_hat, B_hat = self.forward_backward(transition_update_mask = transition_update_mask, save_dir = current_save_dir, use_cloned_emissions = use_cloned_emissions)

            # update the transition matrix
            self.A = A_hat
            if transition_update_mask is not None:
                self.log_A = np.where(
                    transition_update_mask,
                    np.log(self.A + self.eps),
                    -np.inf
                )
                self.A[~transition_update_mask] = 0.0

            else:
                self.log_A = np.log(self.A + self.eps)

            # update the emissions matrix
            self.B = B_hat

            # recompute the emissions
            self.compute_emissions()

            log_alpha = self.forward_probability()

            # compute the log-likelihood
            llhs[iter + 1] = logsumexp(log_alpha[-1])

        return llhs



def calculate_cloned_transition_matrix(alpha, sequenceLength, nSequences, verbose=False, transition_epsilon=1e-8):

    '''calculate the transition matrix for the cloned HMM given the free parameters (alpha, beta, gamma)'''

    # the total number of states is the number of sequences times the sequence length and we add one for the non-sequence state
    nStates = sequenceLength * nSequences + 1

    gamma = (1 - alpha) / (2 * sequenceLength * nSequences) # this is the probability of transitioning into or out of any of the sequence states
    beta = 1 - gamma # this is the probability of transitioning between states in the sequence
    if verbose:
        print(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}")

    # initialize the transition matrix with zeros and then fill in the appropriate entries
    P = np.zeros((1 + 2 * sequenceLength * nSequences, 1 + 2 * sequenceLength * nSequences))
    P[0, 0] = alpha

    # define the transition probabilities into the sequences (can transition into any part of the sequence with equal probability)
    P[0, 1:] = gamma

    # define the transition probabilities out of the sequences (can transition out of any part of the sequence with equal probability)
    P[1:, 0] = gamma

    # define the transition probabilities through the sequences
    start = None
    for i in range(nSequences):
        start = 1 if start is None else end + 1
        end = start + 2 * sequenceLength - 1

        for idx, j in enumerate(range(start, end + 1)):

            # set the beta values for the forward sequence
            if idx < sequenceLength - 1:
                P[j, j + 1] = beta

            # set the beta values for the reverse sequence (these are the cloned states that have the same emission probabilities as the forward sequence states)
            elif idx > sequenceLength:
                P[j, j - 1] = beta
            else:
                P[j, 0] = 1

    # Tiny smoothing avoids exact zeros that can make EM objective report -inf via log(0).
    if transition_epsilon is not None and transition_epsilon > 0:
        P = P + transition_epsilon
        P = P / P.sum(axis=1, keepdims=True)

    return P
