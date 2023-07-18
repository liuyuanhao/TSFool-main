import torch
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.cluster import DBSCAN
from WFA import build_WFA, run_WFA, calculate_average_input_distance
from NFA import build_NFA, run_NFA 
from FSM import build_FSM, run_FSM
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(1)

def calculate_error_rate(perturbation, sample_X, model, target, weights):
    # Apply perturbation to the sample
    perturbed_sample_X = sample_X + weights * perturbation
    # Clip to ensure the sample is in valid range
    perturbed_sample_X = np.clip(perturbed_sample_X, 0, 1)
    # Convert to torch.Tensor and run the model on the perturbed sample
    perturbed_sample_X_torch = torch.from_numpy(perturbed_sample_X).unsqueeze(0).to(torch.float32)
    model_output, _ = model(perturbed_sample_X_torch)
    model_pred_y = torch.max(model_output, 1)[1].data.numpy()
    # Calculate the error rate
    error_rate = np.mean(model_pred_y != target)
    return error_rate


def generate_new_perturbation(perturbation, eps, weights):
    # Generate a small random noise
    noise = np.random.uniform(-eps, eps, size=perturbation.shape)
    # Add the noise to the current perturbation to generate a new perturbation
    new_perturbation = perturbation + weights * noise
    # Clip to ensure the new perturbation is in valid range
    new_perturbation = np.clip(new_perturbation, -eps, eps)
    return new_perturbation

def calculate_weights(X):
    if X.size == 0:
        return np.zeros_like(X)
    derivative = np.diff(X, axis=1)
    abs_derivative = np.abs(derivative)
    if abs_derivative.size == 0:
        return np.ones_like(X)
    max_abs_derivative = np.max(abs_derivative)
    if max_abs_derivative == 0:
        return np.ones_like(X)
    weights = abs_derivative / max_abs_derivative
    weights = np.pad(weights, ((0, 0), (1, 0), (0, 0)), mode='constant')
    return weights






def TSFool(model, X, Y, automaton_type='WFA', K=2, T=30, F=0.1, eps=0.1, N=20, P=0.9, C=1, target=-1, details=False):
    r"""
        Arguments:
            model (nn.Module): target rnn classifier
            X (numpy.array): time series data (sample_amount, time_step, feature_dim)
            Y (numpy.array): label (sample_amount, )
            K (int): >=1, hyper-parameter of build_WFA(), denotes the number of K-top prediction scores to be considered
            T (int): >=1, hyper-parameter of build_WFA(), denotes the number of partitions of the prediction scores
            F (float): (0, 1], hyper-parameter of build_WFA(), ensures that comparing with the average distance
                        between feature points, the grading size of tokens are micro enough
            eps (float): (0, 0.1], hyper-parameter for perturbation, denotes the perturbation amount under the
                        limitation of 'micro' (value 0.1 corresponds to the largest legal perturbation amount)
            N (int): >=1, hyper-parameter for perturbation, denotes the number of adversarial samples generated from
                     a specific minimum positive sample
            P (float): (0, 1], hyper-parameter for perturbation, denotes the possibility of the random mask
            C (int): >=1, hyper-parameter for perturbation, denote the number of minimum positive samples to be
                     considered for each of the sensitive negative samples
            target (int): [-1, max label], hyper-parameter for perturbation, -1 denotes untargeted attack, other
                          values denote targeted attack with the corresponding label as the target
            details (bool): if True, print the details of the attack process
    """

    # Depending on the automaton type, use different functions
    if automaton_type == 'WFA':
        build_automaton = build_WFA
        run_automaton = run_WFA
    elif automaton_type == 'FSM':
        build_automaton = build_FSM
        run_automaton = run_FSM
    elif automaton_type == 'NFA':
        build_automaton = build_NFA
        run_automaton = run_NFA
    else:
        raise ValueError(f"Invalid automaton_type: {automaton_type}")

    # Build, Run and Compare WFA and target RNN Model
    adv_index = []
    # build and run WFA
    abst_alphabet, initial_vec, trans_matrices, final_vec = build_automaton(model, X, Y, K, T, F, details)
    wfa_output = run_automaton(X, Y, abst_alphabet, initial_vec, trans_matrices, final_vec)
    rep_model_output = torch.from_numpy(wfa_output)
    rep_model_pred_y = torch.max(rep_model_output, 1)[1].data.numpy()

    # run target model
    X_torch = torch.from_numpy(X).to(torch.float32)
    model_output, _ = model(X_torch)
    model_pred_y = torch.max(model_output, 1)[1].data.numpy()

    # evaluate models similarity
    differ_record = []
    for i_0 in range(X.shape[0]):
        if rep_model_pred_y[i_0] != model_pred_y[i_0]:
            differ_record.append(i_0)

    # Find Target Positive Samples & Generate Minimum Positive Samples
    average_input_distance = calculate_average_input_distance(X)
    perturbation_amount = eps * average_input_distance

    target_positive_sample_X = []
    minimum_positive_sample_X = []
    minimum_positive_sample_Y = []
    candidate_perturbation_X = []

    for i in differ_record:
        if model_pred_y[i] != Y[i] and rep_model_pred_y[i] == Y[i]:
            if target != -1 and model_pred_y[i] != target:  # targeted / untargeted attack
                continue

            # order neighbor samples for each of the sensitive negative samples according to abstract distance
            abst_bias = []
            for k in range(X.shape[0]):
                current_abst_bias = 0
                for j in range(X.shape[1]):
                    current_abst_bias += abs(int(abst_alphabet[k, j]) - int(abst_alphabet[i, j]))
                abst_bias.append(current_abst_bias)
            candidate_samples_index = np.argsort(abst_bias)
            current_count = 0
            for candidate_sample_index in candidate_samples_index:
                # find corresponding target positive sample(s) from the candidates
                if model_pred_y[candidate_sample_index] != Y[candidate_sample_index] \
                        or Y[candidate_sample_index] != Y[i]:
                    continue
                current_negative_x = copy.deepcopy(np.array(X[i]))  # initialized as the sensitive negative sample
                current_positive_x = X[candidate_sample_index]  # initialized as the target positive sample
                adv_index.append(candidate_sample_index)

                # iterative sampling between (variable) pos and neg samples
                while True:
                    sampled_instant_X = []
                    sampled_instant_Y = []

                    # build sampler
                    sampling_bias_x = []
                    for j in range(X.shape[1]):
                        sampling_bias_f2 = []
                        for k in range(X.shape[2]):
                            real_bias = current_positive_x[j, k] - current_negative_x[j, k]
                            sampling_bias = real_bias / 10
                            sampling_bias_f2.append(sampling_bias)
                        sampling_bias_x.append(sampling_bias_f2)
                    sampling_bias_x = np.array(sampling_bias_x)

                    # sampling
                    sampled_instant_x = copy.deepcopy(current_negative_x)
                    for j in range(11):
                        sampled_instant_X.append(copy.deepcopy(sampled_instant_x))
                        sampled_instant_Y.append(Y[i])
                        sampled_instant_x += sampling_bias_x

                    # find pos and neg distribution
                    sampled_instant_X = np.array(sampled_instant_X)
                    sampled_instant_Y = np.array(sampled_instant_Y)
                    sampled_instant_X = torch.from_numpy(sampled_instant_X).to(torch.float32)
                    sampled_instant_output, _ = model(sampled_instant_X)
                    pred_sampled_instant_y = torch.max(sampled_instant_output, 1)[1].data.numpy()
                    sampled_instant_acc_record = (pred_sampled_instant_y == sampled_instant_Y)

                    # update sampling range
                    for j in range(len(sampled_instant_acc_record) - 1):
                        if not sampled_instant_acc_record[j] and sampled_instant_acc_record[j + 1]:
                            current_negative_x = copy.deepcopy(np.array(sampled_instant_X[j]))
                            current_positive_x = copy.deepcopy(np.array(sampled_instant_X[j + 1]))

                    # end condition
                    end_flag = True
                    for j in range(X.shape[2]):
                        if sampling_bias_x[:, j].max() > perturbation_amount[j] / 10:
                            end_flag = False
                    if end_flag:
                        target_positive_sample_X.append(X[candidate_sample_index])
                        minimum_positive_sample_X.append(current_positive_x)
                        minimum_positive_sample_Y.append(Y[i])
                        candidate_perturbation_X.append(sampling_bias_x)
                        break

                current_count += 1
                if current_count >= C:
                    break

    # Implement Random Mask Perturbations to Generate Adversarial Samples
    adv_X = []
    adv_Y = []

    for i in range(len(minimum_positive_sample_X)):
        minimum_positive_sample_x = minimum_positive_sample_X[i]
        minimum_positive_sample_y = minimum_positive_sample_Y[i]
        current_perturbation_x = candidate_perturbation_X[i]

        # Initialize the perturbation
        perturbation = np.zeros_like(minimum_positive_sample_x)
        # Set the initial temperature
        T = 10.0
        # Set the cooling rate
        cooling_rate = 0.95
        # Set the maximum number of iterations
        max_iter = 1000
        # Set the tolerance for error rate improvement
        tol = 0.001
        # Set the patience for error rate improvement
        patience = 20
        # Initialize the best error rate as infinity
        best_error_rate = np.inf
        # Initialize the counter for error rate improvement
        counter = 0
        # Calculate the weights based on the approximate derivative
        weights = calculate_weights(minimum_positive_sample_x)
        # Simulated annealing
        for _ in range(max_iter):
            # Generate a new perturbation
            new_perturbation = generate_new_perturbation(perturbation, eps, weights)
            # Calculate the error rates of the old and new perturbations
            old_error_rate = calculate_error_rate(perturbation, minimum_positive_sample_x, model, minimum_positive_sample_y, weights)
            new_error_rate = calculate_error_rate(new_perturbation, minimum_positive_sample_x, model, minimum_positive_sample_y, weights)
            # Decide whether to accept the new perturbation
            if new_error_rate > old_error_rate or np.random.rand() < np.exp((new_error_rate - old_error_rate) / T):
                perturbation = new_perturbation
                # If the new error rate is better than the best error rate, update the best error rate
                if new_error_rate < best_error_rate - tol:
                    best_error_rate = new_error_rate
                    counter = 0
                else:
                    counter += 1
            # Cool down
            T *= cooling_rate
            # If the error rate has not improved for a certain number of iterations, stop the iteration
            if counter >= patience:
                break
        # Apply the final perturbation to the minimum positive sample to generate adversarial samples
        adv_x = minimum_positive_sample_x + perturbation
        adv_X.append(adv_x)
        adv_Y.append(minimum_positive_sample_y)

    # Use DBSCAN to cluster the minimum positive samples
    scaler = MinMaxScaler()
    minimum_positive_sample_X_scaled = scaler.fit_transform(np.vstack(minimum_positive_sample_X).reshape(len(minimum_positive_sample_X), -1))
    dbscan = DBSCAN(eps=0.5)
    dbscan.fit(minimum_positive_sample_X_scaled)
    unique_clusters = np.unique(dbscan.labels_)
    adv_X_clustered = []
    adv_Y_clustered = []
    for cluster in unique_clusters:
        # Select one sample from each cluster
        cluster_indices = np.where(dbscan.labels_ == cluster)[0]
        selected_sample_index = np.random.choice(cluster_indices)
        adv_X_clustered.append(adv_X[selected_sample_index])
        adv_Y_clustered.append(adv_Y[selected_sample_index])

    return np.array(adv_X_clustered), np.array(adv_Y_clustered), np.array(target_positive_sample_X), adv_index

if __name__ == '__main__':
    from models.models_structure.ECG200 import RNN
    model = torch.load('models/GP.pkl')
    dataset_name = 'GunPoint'
    X = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_X.npy')
    Y = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_Y.npy')
    adv_X, adv_Y, target_X, index = TSFool(model, X, Y, automaton_type='NFA', K=2, T=30, F=0.1, eps=0.01, N=20, P=0.9, C=4, target=-1, details=False)
    print(adv_X,adv_Y,target_X,index)
    X_adv_torch = torch.from_numpy(adv_X).to(torch.float32)
    output_adv, _ = model(X_adv_torch)
    pred_adv = torch.max(output_adv, 1)[1].data.numpy()

    # Calculate the attack success rate
    attack_success_rate = np.mean(pred_adv != adv_Y)

    print('Attack success rate:', attack_success_rate)
