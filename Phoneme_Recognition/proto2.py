#!pip install tools2
import numpy as np
#from tools2 import *

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    hmm = dict()
    startprob1 = hmm1['startprob']
    startprob2 = hmm2['startprob']
    transmat1 = hmm1['transmat']
    transmat2 = hmm2['transmat']
    means1 = hmm1['means']
    means2 = hmm2['means']
    covars1 = hmm1['covars']
    covars2 = hmm2['covars']

    # build concatenated startprob

    startprob1ext = np.append(startprob1,  np.tile(startprob1[-1], (1, startprob2.size - 1)))
    startprob2ext = np.concatenate((np.ones(startprob1.size - 1), startprob2))

    hmm["startprob"] = np.multiply(startprob1ext, startprob2ext)

    # build transition matrix

    transmat1ext = np.append(transmat1[:-1, :],  np.tile(transmat1[:-1, [-1]], (1, startprob2.size - 1)), 1)

    startrep2 = np.tile(startprob2, (startprob1.size - 1, 1))
    startrep2ext = np.concatenate((np.ones((startprob1.size - 1, startprob1.size - 1)), startrep2), axis = 1)

    firsthalf = np.multiply(transmat1ext, startrep2ext)
    secondhalf = np.concatenate((np.zeros((startprob2.size, startprob1.size - 1)), transmat2), axis = 1)

    hmm["transmat"] = np.concatenate((firsthalf, secondhalf))

    # build means matrix

    hmm["means"] = np.concatenate((means1, means2))

    # build covariances matrix

    hmm["covars"] = np.concatenate((covars1, covars2))

    return hmm
    

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    gmmloglik = 0
    arrayloglik = np.empty(gmm_emlik.shape)
    for i in range(gmm_emlik.shape[0]):
      loglik = gmm_emlik[i]
      log_weights = np.log(W)
      logsum = loglik + log_weights
      arrayloglik[i] = logsum
      print(logsum)
      gmmloglik = gmmloglik + logsumexp(logsum)
    print(gmmloglik)
    
    return arrayloglik

    

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    alpha = np.empty((log_emlik.shape[0], log_emlik.shape[1]))
    alpha[0] = log_startprob[0:-1] + log_emlik[0, :]
    for i in range (1, alpha.shape[0]):
      for j in range (0, alpha.shape[1]):
        alpha[i][j] = logsumexp(alpha[i - 1] + log_transmat[:-1, j]) + log_emlik[i, j]

    return alpha

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

    bet = np.empty((log_emlik.shape[0], log_emlik.shape[1]))
    bet[log_emlik.shape[0] - 1] = np.zeros(log_emlik.shape[1])
    for i in range (log_emlik.shape[0] - 2,  -1, -1):
      for j in range (0, bet.shape[1]):
        bet[i][j] = logsumexp(log_transmat[j, :-1] + log_emlik[i + 1] + bet[i + 1])
    return bet


def backtrack_viterbi(B, last_idx):
    best_path = [last_idx]
    for i in range(B.shape[0]-1, 0, -1):
      best_path.append(int(B[i, best_path[-1]]))

    best_path.reverse()
    return best_path


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    emissions = log_emlik.shape[0]
    states = log_emlik.shape[1]
    B = np.zeros((emissions, states))
    V = np.zeros((emissions, states))
    V[0] = log_startprob[0:-1] + log_emlik[0, :]
    for i in range(1, emissions):
      for j in range(states):
        V[i][j] = np.max(V[i - 1, :] + log_transmat[:-1, j]) + log_emlik[i, j]
        B[i][j] = np.argmax(V[i - 1, :] + log_transmat[:-1, j])
      
    max_last_index = np.argmax(V[emissions-1])
    viterbi_path = backtrack_viterbi(B, max_last_index);
    viterbi_loglik = np.max(V[emissions-1]) #to check at time t(last emission)
    return viterbi_loglik, viterbi_path

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

    log_gamma = np.empty((log_alpha.shape[0], log_alpha.shape[1]))

    log_gamma = log_alpha + log_beta - logsumexp(log_alpha[-1])

    return log_gamma

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    means = np.empty((log_gamma.shape[1], X.shape[1]))
    covars = np.empty(means.shape)
    gamma = np.exp(log_gamma)

    gamma_sum = np.sum(gamma, axis=0)

    for i in range(means.shape[0]):
        means[i] = np.sum(gamma[: , [i]] * X, axis = 0) / gamma_sum[i]
        covars[i] = np.sum(gamma[: , [i]] * (X - means[i])**2, axis = 0) / gamma_sum[i]
        covars[i, covars[i] < varianceFloor] = varianceFloor
    return (means, covars)

def train(data, model, it = 20, threshold = 1):
    prevloglik = - math.inf
    for i in range(it) :
      obsloglik = log_multivariate_normal_density_diag(data, model['means'], model['covars'])

      alpha = forward(obsloglik , np.log(model['startprob']), np.log(model['transmat']))
      loglik = logsumexp(alpha[-1])

      print(loglik)

      if((np.abs(loglik - prevloglik) < threshold) and prevloglik != - math.inf):
            break

      prevloglik = loglik
      beta = backward(obsloglik, np.log(model['startprob']), np.log(model['transmat']))
      log_gamma = statePosteriors(alpha, beta)

      model['means'], model['covars'] = updateMeanAndVar(data, log_gamma)

    print("Final loglik: " + str(prevloglik))
    return model

import numpy as np
import matplotlib.pyplot as plt
import time

def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    if vmax.ndim > 0:
        vmax[~np.isfinite(vmax)] = 0
    elif not np.isfinite(vmax):
        vmax = 0
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(arr - vmax), axis=0))
        out += vmax
        return out

def log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model

    Args:
        X: array like, shape (n_observations, n_features)
        means: array like, shape (n_components, n_features)
        covars: array like, shape (n_components, n_features)

    Output:
        lpr: array like, shape (n_observations, n_components)
    From scikit-learn/sklearn/mixture/gmm.py
    """
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr