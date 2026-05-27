import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from custom_hmm import PoissonHMM


def fit_hmm(N: int, emissions: np.ndarray, save_dir: str, Niters: int = 40, nStarts: int = 50):

    '''
    Fit a Poisson HMM to the given emissions using the Baum-Welch algorithm with multiple random restarts.
    The transition matrix is initialized with a structure that allows for cloned states, and only the allowed transitions are updated during EM.
    The emissions for the cloned states are enforced to be the same throughout EM. The function returns the best fitted model and its log-likelihood.

    Parameters:
    N (int): Number of states in the HMM (including cloned states).
    emissions (np.ndarray): The observed emissions with shape (T, emission_dim).
    save_dir (str): Directory to save the results and plots for each random restart.
    Niters (int): Number of EM iterations to run for each random restart.
    nStarts (int): Number of random restarts to perform.

    Returns:
    best_model (PoissonHMM): The best fitted HMM model.
    best_ll (float): The log-likelihood of the best fitted model.
    '''




    best_ll = -np.inf
    best_model = None
    all_lls = []


    # make the transition update mask (true = update, false = freeze)
    transition_update_mask = np.zeros((N, N), dtype = bool)
    transition_update_mask[0,:] = True
    transition_update_mask[:, 0] = True

    nSequenceStates = (N - 1) // 2
    for i in range(nSequenceStates):
        transition_update_mask[i + 1, i + 2] = True
        transition_update_mask[nSequenceStates + i + 1, nSequenceStates + i] = True

    plt.figure()
    sns.heatmap(transition_update_mask)
    plt.show()


    for seed in range(nStarts):

        current_save_dir = os.path.join(save_dir, f"restart_{seed}")
        os.makedirs(current_save_dir, exist_ok=True)

        # set random seed for reproducibility
        np.random.seed(seed)

        # draw random initial transition matrix from a dirichlet distribution
        initial_A = np.random.dirichlet(np.ones(N), size=N)

        # Set the off-diagonal and non-sequence transitions to zero to enforce the structure of the cloned transition matrix
        initial_A = np.where(
            transition_update_mask,
            initial_A,
            0.0
        )

        # renormalize the rows to sum to 1
        row_sums = initial_A.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError("At least one row of initial_A has no allowed transitions.")
        initial_A = initial_A / row_sums


        #set the initial_emission_rates as the mean rate for each neuron across all time points
        initial_emission_rates = emissions.mean(axis=0)

        # convert to matrix with shape (nStates, emission_dim)
        initial_emission_rates = np.repeat(initial_emission_rates[None, :], N, axis=0)

        # add some noise to the initial rates to break symmetry
        initial_emission_rates += 1e-6 * np.random.normal(size=initial_emission_rates.shape)

        # make sure all rates are positive
        initial_emission_rates = np.clip(initial_emission_rates, a_min=1e-9, a_max=3)

        # initialize the prior probabilties pi
        initial_pi = np.ones(N) / N

        
        # define the model
        hmm = PoissonHMM(
            initial_A,
            initial_emission_rates,
            initial_pi,
        )

        # set the observations
        hmm.set_observations(emissions)



        # run the baum-welch algorithm with cloned states
        lls = hmm.fit_em(Niters=Niters, use_cloned_emissions=True, transition_update_mask=transition_update_mask, save_dir=None)



        # plot the EM convergence
        plt.figure()
        plt.plot(np.asarray(lls))
        plt.title(f"EM Convergence for Restart {seed + 1}")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log-Likelihood")
        plt.tight_layout()
        plt.savefig(os.path.join(current_save_dir, f"em_convergence.png"))
        plt.close()

        # plot the transition matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.array(hmm.A), cmap="Blues", cbar_kws={"label": "Transition Probability (%)"}, vmin=0, vmax=1)
        # overlay the numbers for the transition probabilities
        for i in range(N):
            for j in range(N):
                plt.text(j + 0.5, i + 0.5, f"{(hmm.A[i, j] * 100):.0f}", ha="center", va="center", color="black")
        plt.title("Learned Transition Matrix")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.tight_layout()
        plt.savefig(os.path.join(current_save_dir, f"learned_transition_matrix.png"))
        plt.close()

        all_lls.append(lls[-1])
        if lls[-1] > best_ll:
            best_ll = lls[-1]
            best_model = hmm

    # plot the distribution of final log-likelihoods across restarts
    plt.figure()
    sns.histplot(all_lls, bins=20, kde=True)
    plt.title("Distribution of Final Log-Likelihoods Across Restarts")
    plt.xlabel("Final Log-Likelihood")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"final_ll_distribution.png"))

    return best_model, best_ll