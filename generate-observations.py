
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from custom_hmm import calculate_cloned_transition_matrix
import os
import time

#--------------------------------------------------------------------------------------MAIN FUNCTION---------------------------------------------------------------


def generate_observations(alpha: float = 0.1, sequenceLength: int = 5, nSequences: int = 2, emission_dim: int = 100, Nsteps: int = 20_000, base_save_dir: str = r'C:\Users\srafi\OneDrive\NeuroStatsLab\sequence-detection-figures', show_plots: bool = True, plot_len: int = 100, save_obs: bool = False):
    
    '''generate simulated observations from a cloned HMM with the given parameters and save the results and some diagnostic plots to a directory determined by the parameters
    
    Parameters
    ----------
    alpha : float
        the probability of staying in the non sequence state. This is the only free parameter we have to set, the rest of the transition probabilities are determined by this and the sequence length and number of sequences
    sequenceLength : int
        the number of latent states in each sequence
    nSequences : int
        the total number of sequences
    emission_dim : int
        the number of neurons we are simulating
    Nsteps : int
        the number of time steps to simulate
    base_save_dir : str
        the base directory where the results and plots will be saved
    show_plots : bool
        whether to display the plots
    plot_len : int
        how many time bins to plot for the simulated data (if show_plots is True)
    save_obs : bool
        whether to save the simulated observations as .npy files in the save directory

    Returns
    -------
    save_dir : str
        the directory where the results and plots are saved
    states : np.ndarray
        the simulated state sequence
    emissions : np.ndarray
        the simulated emissions
    emission_prob : np.ndarray
        the emission probabilities for each state
    P : np.ndarray
        the transition matrix

    '''

    save_dir = os.path.join(base_save_dir, f"cloned-HMM-simulated-data-alpha-{alpha}-sequenceLength-{sequenceLength}_nSequences-{nSequences}_emissionDim-{emission_dim}_Nsteps-{Nsteps}_{time.strftime('%Y-%m-%d-%H-%M-%S')}")
    os.makedirs(save_dir, exist_ok=True)

    # Compute the transition matrix given the free paramerters
    P = calculate_cloned_transition_matrix(alpha, sequenceLength, nSequences, verbose=True)
    nStates = P.shape[0]

    # simulate data from this transition matrix
    states = np.zeros(Nsteps, dtype=int)
    for t in range(1, Nsteps):
        states[t] = np.random.choice(nStates, p=P[states[t-1]])


    # draw the lambda parameters for the Poisson emissions from a lognormal distribution (this is a common choice for a prior on Poisson rates)
    emission_prob = np.zeros((nStates, emission_dim))
    print(emission_prob.shape)

    nStates = P.shape[0]
    pairs = []
    for seq in range(nSequences):
        for step in range(sequenceLength):
            new_pair = (seq * 2 * sequenceLength + step + 1, seq * 2 * sequenceLength  + sequenceLength + step + 1)
            pairs.append(new_pair)
    print(pairs)


    # set the emission probabilities for the paired states to be the same (these are the cloned states that have the same emissions as the forward sequence states)
    for pair in pairs:
        emissions = np.random.lognormal(
            mean=-3,
            sigma=1.2,
            size=(emission_dim,)
        )
        # clip the emissions at 2Hz
        emissions = emissions.clip(0, 2)
        emission_prob[pair[1], :], emission_prob[pair[0], :] = emissions, emissions

    # set the emission prob in state zero to be the median across all the other states
    emission_prob[0, :] = np.median(emission_prob[1:, :], axis=0)


    # simulate emissions from the HMM given the state sequence and the emission probabilities
    emissions = np.zeros((Nsteps, emission_dim), dtype=int)
    for t in range(Nsteps):
        emissions[t] = np.random.poisson(lam=emission_prob[states[t]])

    
    # saving the simulated observations as .npy files in the save directory
    if save_obs:
        np.save(os.path.join(save_dir, "states.npy"), states)
        np.save(os.path.join(save_dir, "emissions.npy"), emissions)
        np.save(os.path.join(save_dir, "emission_prob.npy"), emission_prob)
        np.save(os.path.join(save_dir, "transition_matrix.npy"), P)







    #------------------------------------------------------PLOTTING---------------------------------------------------------------

    if show_plots:



        # plot the transition matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(P, cmap="Blues", cbar_kws={"label": "Transition Probability (%)"}, vmin=0, vmax=1)
        # overlay the numbers for the transition probabilities
        for i in range(nStates):
            for j in range(nStates):
                plt.text(j + 0.5, i + 0.5, f"{(P[i, j] * 100):.0f}", ha="center", va="center", color="black")
        plt.title("Transition Matrix with Sequences")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.savefig(os.path.join(save_dir, "transition_matrix.png"))
        plt.close()



        # plot the distribution of emission probabilities for each state
        plt.figure(figsize=(4, 6))
        sns.histplot(emission_prob.flatten(), bins=20, stat='density', log_scale=(True, False), color="blue", edgecolor="black")
        plt.title("Distribution of Emission Probabilities")
        plt.xlabel("Emission Probability (Poisson Rate)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "emission_prob_distribution.png"))
        plt.close()



        # plot the emission prob matrix
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        sns.heatmap(emission_prob.T, cmap="Reds", ax=axs[0])
        axs[0].set_title("Emission Probability Matrix")
        axs[0].set_xlabel("State")
        axs[0].set_ylabel("Neuron")

        # assign neurons to states (for each neuron, the state of maximal firing is the state that neuron is assigned to)
        neuron_states = np.argmax(emission_prob, axis=0)

        # sort the neurons by their assigned states and plot the emision prob matrix again
        sorted_indices = np.argsort(neuron_states)
        sns.heatmap(emission_prob[:, sorted_indices].T, cmap="Reds", ax=axs[1], cbar_kws={"label": "Emission Probability (Poisson Rate)"}, vmin=0, vmax=np.percentile(emission_prob, 99))
        axs[1].set_title("Emission Probability Matrix\n(Neurons Sorted by Assigned State)")
        axs[1].set_xlabel("State")
        axs[1].set_ylabel("Neuron")
        plt.savefig(os.path.join(save_dir, "emission_prob_matrix.png"))
        plt.close()



        from matplotlib.gridspec import GridSpec

        seq_colors = sns.color_palette("Set2", nSequences)

        # Sort neurons by assigned state
        sorted_indices = np.argsort(neuron_states)

        # ----------------------------------
        # Counts over ENTIRE dataset
        # ----------------------------------

        # State occupancy counts
        state_counts = np.bincount(states, minlength=np.max(states)+1)

        # Total spikes/emissions per neuron
        neuron_counts = emissions.sum(axis=0)
        neuron_counts_sorted = neuron_counts[sorted_indices]

        # ----------------------------------
        # Figure layout
        # ----------------------------------

        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(
            2, 2,
            width_ratios=[20, 3],
            height_ratios=[1, 1],
            hspace=0.2,
            wspace=0.05
        )

        ax_state = fig.add_subplot(gs[0, 0])
        ax_emiss = fig.add_subplot(gs[1, 0], sharex=ax_state)

        ax_state_hist = fig.add_subplot(gs[0, 1], sharey=ax_state)
        ax_emiss_hist = fig.add_subplot(gs[1, 1], sharey=ax_emiss)

        # ----------------------------------
        # Main plots
        # ----------------------------------

        sns.lineplot(
            x=np.arange(plot_len),
            y=states[:plot_len],
            ax=ax_state,
            linewidth=3
        )

        ax_state.set_title("Simulated State Sequence")
        ax_state.set_ylabel("State")
        ax_state.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
        )


        sns.heatmap(
            emissions[:plot_len, sorted_indices].T,
            cmap="Greys",
            ax=ax_emiss,
            vmax=3,
            cbar=False
        )

        ax_emiss.invert_yaxis()
        ax_emiss.set_title("Simulated Emissions")
        ax_emiss.set_xlabel("Time Step")
        ax_emiss.set_ylabel("Neuron")

        # ----------------------------------
        # Histograms on right
        # ----------------------------------

        # State occupancy histogram
        ax_state_hist.barh(
            np.arange(len(state_counts)),
            state_counts,
            color='gray'
        )
        ax_state_hist.set_xlabel("Count")

        # Neuron firing histogram
        ax_emiss_hist.barh(
            np.arange(len(neuron_counts_sorted)) + 0.5,
            neuron_counts_sorted,
            color='gray'
        )
        ax_emiss_hist.set_xlabel("Spikes")

        # Clean up histogram axes
        ax_state_hist.tick_params(labelleft=False)
        ax_emiss_hist.tick_params(labelleft=False)

        # ----------------------------------
        # Sequence shading
        # ----------------------------------

        for i in range(nSequences):

            forward_start = 1 + i * 2 * sequenceLength
            forward_end = forward_start + sequenceLength

            reverse_start = forward_end
            reverse_end = reverse_start + sequenceLength

            ax_state.axhspan(
                forward_start,
                forward_end - 1,
                color=seq_colors[i],
                alpha=0.1
            )

            ax_state.axhspan(
                reverse_start,
                reverse_end - 1,
                color=seq_colors[i],
                alpha=0.1
            )

        sns.despine()
        plt.savefig(os.path.join(save_dir, "simulated_sequences.png"))
        plt.close()

    return save_dir, states, emissions, emission_prob, P




#--------------------------------------------------------------------------------------UTILITIES---------------------------------------------------------------

def calculate_cloned_transition_matrix(alpha: float, sequenceLength: int, nSequences: int, verbose: bool = False, transition_epsilon: float = 1e-8):


    '''calculate the transition matrix for the cloned HMM given the free parameters (alpha, beta, gamma)

    Parameters
    ----------
    alpha : float
        the probability of staying in the non sequence state. This is the only free parameter we have to set, the rest of the transition probabilities are determined by this and the sequence length and number of sequences
    sequenceLength : int
        the number of latent states in each sequence
    nSequences : int
        the total number of sequences
    verbose : bool
        whether to print the calculated probabilities
    transition_epsilon : float
        a small value to add to the transition matrix to avoid exact zeros

    Returns
    -------
    P : np.ndarray
        the transition matrix
    '''

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




if __name__ == "__main__":
    generate_observations()