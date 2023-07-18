import torch
import torch.nn.functional as Func
import numpy as np
import copy
import time

torch.manual_seed(1)


def calculate_average_input_distance(X):
    current_distance = np.zeros(X.shape[2])
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - 1):
            for k in range(X.shape[2]):
                current_distance[k] += abs(X[i, j + 1, k] - X[i, j, k])
    average_distance = current_distance / (X.shape[0] * (X.shape[1] - 1))
    return average_distance


def findByRow(mat, row):
    return np.where((mat == row).all(1))[0]


def build_NFA(model, X, Y, K=3, T=30, F=0.1, details=False):
    r"""
        Arguments:
            model (nn.Module): target rnn classifier
            X (numpy.array): time series data (sample_amount, time_step, feature_dim)
            Y (numpy.array): label (sample_amount, )
            K (int): >=1, hyper-parameter for k-DCP, denote the number of K-top prediction scores to be considered
            T (int): >=1, hyper-parameter for k-DCP, denote the number of partitions of the prediction scores
            F (float): (0,1], hyper-parameter for alphabet partition, ensure that comparing with the average distance
                       between feature points, the grading size of tokens are micro enough
            details (bool): if True, print the details of the building process
    """

    # --------------------------- Input Tokens Abstraction ---------------------------- #

    # getting min-max normalized alphabet
    alphabet = copy.deepcopy(X)
    min_token = np.min(alphabet)
    alphabet -= min_token
    max_token = np.max(alphabet)
    alphabet /= max_token

    # determining partition granularity of alphabet
    average_distance = calculate_average_input_distance(X)
    T_alphabet = 1
    for i in range(X.shape[2]):
        T_alphabet *= int(10 / (F * average_distance[i]))

    # abstracting input tokens as k-DCP format
    k_DCP = np.floor(alphabet * (T_alphabet - 1))
    k_DCP = np.reshape(k_DCP, (-1, alphabet.shape[2]))
    uniques = np.unique(k_DCP, axis=0)  # remove duplicates
    abst_inputs_number = len(uniques)

    # building labelling representation of abstract input tokens
    abst_alphabet_labels = np.zeros(X.shape[0] * X.shape[1]).astype(int)
    for i in range(len(uniques)):
        j = findByRow(k_DCP, uniques[i, :])
        abst_alphabet_labels[j] = i
    abst_alphabet_labels = np.reshape(abst_alphabet_labels, (X.shape[0], X.shape[1]))

    if details:
        print(f"\nNumber of Prediction Score Partitions: {T_alphabet}")
        print(f'Number of Abstracted Input Tokens: {abst_inputs_number}')

    # ------------ RNN Hidden States Abstraction (Under Sequential Inputs) ------------ #

    # extracting hidden states from the target model
    X_torch = torch.from_numpy(X).to(torch.float32)
    output, output_trace = model(X_torch)
    num_classes = len(np.unique(Y))  # Add this line to define num_classes variable
    item_classes = np.zeros(num_classes)  # Define item_classes variable


    # formatting hidden states to probability distribution
    output_trace = Func.softmax(output_trace, dim=2)  # softmax in last dim
    states = output_trace.detach().numpy()
    max_state = None  # Add this line to define max_state variable


    # abstracting the (prob) states as k-DCP format
    sorted_states = -np.sort(-states)[:, :, :K]  # (samples, timesteps, features)
    sorted_states_index = np.argsort(-states)[:, :, :K]  # (samples, timesteps, features)
    sorted_states_t = np.floor(sorted_states * (T - 1))
    k_DCP = np.append(sorted_states_index, sorted_states_t, axis=2)  # (samples, timesteps, 2*K)
    k_DCP = np.reshape(k_DCP, (-1, 2 * K))
    uniques = np.unique(k_DCP, axis=0)  # remove duplicates
    abst_states_number = len(uniques)

    # building labelling representation of the states
    abst_states_labels = np.zeros(X.shape[0] * X.shape[1]).astype(int)  # (samples, timesteps)
    for i in range(len(uniques)):
        j = findByRow(k_DCP, uniques[i, :])
        abst_states_labels[j] = i
    abst_states_labels = np.reshape(abst_states_labels, (X.shape[0], X.shape[1]))  # (samples, timesteps)
    abst_states_labels = abst_states_labels + 1  # Leaving the index 0 for the Initial Status of WFA

    if details:
        print(f'\nAn Example for k-DCP Algorithm:\n - Original State: {states[0, 0, :]}')
        print(f' - Corresponding Prediction Label: {sorted_states_index[0, 0, :]}')
        print(f' - Corresponding Confidence Level: {sorted_states_t[0, 0, :]}')
        print(f' - Abstracted State: {k_DCP[0, :]}')
        print(f'Number of Abstracted States: {abst_states_number}')

    # -------------------------- Initial Vector Establishment ------------------------- #

    initial_vector = np.zeros(abst_states_number + 1)
    initial_vector[0] = 1

    if details:
        print('\nShape of Initial Vector:\n', initial_vector.shape)
        print('Initial Vector:\n', initial_vector)

    # ----------------------- Transition Matrices Establishment ----------------------- #

    # statistic transition matrices establishment
    transition_matrices = np.zeros((abst_inputs_number, abst_states_number + 1, abst_states_number + 1))
    for item in range(abst_inputs_number):  # for every abstract token
        for i_0 in range(X.shape[0]):
            for i_1 in range(X.shape[1]):  # for every token
                if abst_alphabet_labels[i_0, i_1] == item:
                    if i_1 == 0:  # confirm front abstract state
                        front_abst_state = 0
                    else:
                        front_abst_state = int(abst_states_labels[i_0, i_1 - 1])
                    back_abst_state = int(abst_states_labels[i_0, i_1])  # confirm back abstract state
                    transition_matrices[item, front_abst_state, back_abst_state] += 1  # add a transition edge

    # deterministic transition matrices establishment
    for item in range(abst_inputs_number):
        for i in range(transition_matrices.shape[1]):
            i_sum = np.sum(transition_matrices[item, i, :])
            if i_sum != 0:
                max_state = np.argmax(transition_matrices[item, i, :])
            transition_matrices[item, i, :] = 0
            if max_state is not None:
                transition_matrices[item, i, max_state] = 1

    if details:
        print('Shape of Prob Transition Matrices:\n', transition_matrices.shape)
        print('Sample of a Prob Transition Matrix:\n', transition_matrices[0, :, :])
        print('Sum of the Element of the Sample Prob Transition Matrix:\n', np.sum(transition_matrices[0, :, :]))

    # -------------------------- Final Vector Establishment  -------------------------- #

    # statistic final vectors establishment
    output_size = int(np.max(Y)+1)
    non_prob_final_vector = np.zeros([abst_states_number + 1, output_size])
    for i_0 in range(X.shape[0]):
        for i_1 in range(X.shape[1]):  # for every state
            state_class = np.argsort(-states[i_0, i_1, :])[0]  # the class of state in original problem
            abst_label = int(abst_states_labels[i_0, i_1])  # the abst label of state
            non_prob_final_vector[abst_label, state_class] += 1  # corresponding class count++

    # probabilistic final vectors establishment
    final_vector = np.zeros([abst_states_number + 1, output_size])
    output_0 = np.zeros(output_size)  # create abstract state 0 ([0.333 0.333 0.333])
    output_0_tensor = torch.from_numpy(output_0)
    state_0_tensor = Func.softmax(output_0_tensor)
    state_0 = state_0_tensor.detach().numpy()
    final_vector[0, :] = state_0
    for item in range(1, abst_states_number + 1):  # for every abstract state
        max_class = np.argmax(non_prob_final_vector[item, :])
        non_prob_final_vector[item, :] = 0
        non_prob_final_vector[item, max_class] = 1
        item_sum = np.sum(item_classes)  # count the num of all the classes
        final_vector[item, :] = non_prob_final_vector[item, :]  # count the probabilistic of every class

    if details:
        print('\nShape of Final Vector:\n', final_vector.shape)
        print('Prob Final Vector:\n', final_vector)

    return abst_alphabet_labels, initial_vector, transition_matrices, final_vector


def run_NFA(X, Y, abstract_alphabet, initial_vector, transition_matrices, final_vector):
    r"""
        Arguments:
            X (numpy.array): time series data (sample_amount, time_step, feature_dim)
            Y (numpy.array): label (sample_amount, )
            abstract_alphabet (numpy.array): abstract tokens set, from build_NFA()
            initial_vector (numpy.array):  initial state of WFA, from build_NFA()
            transition_matrices (numpy.array): record transitions of WFA under different tokens, from build_NFA()
            final_vector (numpy.array): give prediction from output, from build_NFA()
    """
    output_size = int(np.max(Y) + 1)
    WFA_output = np.zeros([X.shape[0], output_size])

    for i_0 in range(X.shape[0]):
        outputs = [initial_vector]
        for i_1 in range(X.shape[1]):
            index = int(abstract_alphabet[i_0, i_1])  # confirm the abst label of the token case
            transition_matrix = transition_matrices[index, :, :]
            new_outputs = []
            for output in outputs:
                new_output = np.matmul(output, transition_matrix)
                new_outputs.append(new_output)
            outputs = new_outputs
        outputs = [np.matmul(output, final_vector) for output in outputs]
        output = np.mean(outputs, axis=0)
        WFA_output[i_0, :] = output

    return WFA_output


if __name__ == '__main__':
    from models.models_structure.DPOAG import RNN
    model = torch.load('models/DPOAG.pkl')
    dataset_name = 'DistalPhalanxOutlineAgeGroup'
    X = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TRAIN_X.npy')
    Y = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TRAIN_Y.npy')
    abst_alphabet, initial_vec, trans_matrices, final_vec = build_NFA(model, X, Y, K=3, T=30, F=0.1, details=True)
    wfa_output = run_NFA(X, Y, abst_alphabet, initial_vec, trans_matrices, final_vec)
    wfa_output = torch.from_numpy(wfa_output)  # transform to tensor
    wfa_pred_y = torch.max(wfa_output, 1)[1].data.numpy()
    wfa_accuracy = float((wfa_pred_y == Y).astype(int).sum()) / float(Y.size)
    print('WFA Accuracy: %.4f' % wfa_accuracy)
