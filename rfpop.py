import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from piecewise import PiecewiseQuadratic, QFunction, BiWeight, Huber, L2Loss, L1Loss



def rfpop(y, loss, beta, verbose=False):   
    
    Q_star_intervals = [(np.min(y), np.max(y))]
    Q_star_coefficients = [np.array([0.,0.,0.])]
    tau_Q_star = [0]
    
    Q_star = QFunction(Q_star_intervals, Q_star_coefficients, tau_Q_star)
    
    cp = []
    
    for t in tqdm(range(y.shape[0])):
        Q = algo2(Q_star, loss, y, t)
        Q_t, tau_t = algo3(Q)
        cp.append((Q_t, tau_t))
        C = Q_t + beta
        Q_star = algo4(Q, C, t)
        if verbose:
            show_Q_Q_star(Q, Q_star, y)
    
    return cp



def show_Q_Q_star(Q, Q_star, y):
    x = np.linspace(np.min(y), np.max(y), 1000)
    _, ax = plt.subplots(1, 2, figsize=(20, 4))
    y_Q = [Q(i) for i in x]
    y_Q_star = [Q_star(i) for i in x]
    ax[0].plot(x, y_Q)
    ax[0].set_ylabel('Qt')
    for i in range(len(Q.intervals)):
        ax[0].axvline(Q.intervals[i][0], color='r', linestyle='--')
        ax[0].axvline(Q.intervals[i][1], color='r', linestyle='--')
    ax[1].plot(x, y_Q_star)
    ax[1].set_ylabel('Qt*')
    for i in range(len(Q_star.intervals)):
        ax[1].axvline(Q_star.intervals[i][0], color='r', linestyle='--')
        ax[1].axvline(Q_star.intervals[i][1], color='r', linestyle='--')
    plt.show()
    
    

def algo2(Q_star, loss, y, t):
    if loss.loss_type == 'biweight':
        gamma = BiWeight(y[t], loss.K, np.min(y), np.max(y))
    elif loss.loss_type == 'huber':
        gamma = Huber(y[t], loss.K, np.min(y), np.max(y))
    elif loss.loss_type == 'l2':
        gamma = L2Loss(y[t], np.min(y), np.max(y))
    elif loss.loss_type == 'l1':
        gamma = L1Loss(y[t], np.min(y), np.max(y))
    
    intervals_Q = []
    coefficients_Q = []
    tau_Q = []
    
    N = 0
    i = 0
    j = 0
    
    while i < len(Q_star.intervals) and j < len(gamma.intervals):
        N += 1
        intervals_Q.append((max(Q_star.intervals[i][0], gamma.intervals[j][0]), min(Q_star.intervals[i][1], gamma.intervals[j][1])))
        coefficients_Q.append(Q_star.coefficients[i]+gamma.coefficients[j])
        tau_Q.append(Q_star.taus[i])
        if min(Q_star.intervals[i][1], gamma.intervals[j][1]) == Q_star.intervals[i][1]:
            i += 1
        else:
            j += 1
            
    return QFunction(intervals_Q, coefficients_Q, tau_Q)



def algo3(Q):
    Q_t = np.inf
    tau_t = 0
    
    for i in range(len(Q.intervals)):
        m = Q.min_on_interval(i)
        if m < Q_t:
            Q_t = m
            tau_t = Q.taus[i]
    
    return Q_t, tau_t



def algo4(Q, C, t):
    Q_minus_C = PiecewiseQuadratic(Q.intervals, [coeffs-np.array([0,0,C]) for coeffs in Q.coefficients])
    intervals_Q_star = []
    coefficients_Q_star = []
    tau_Q_star = []
    
    for i in range(len(Q.intervals)):
        R_tmp = Q_minus_C.roots_on_interval(i)
        R = [Q.intervals[i][0]] + R_tmp + [Q.intervals[i][1]]
        for j in range(len(R_tmp)+1):
            intervals_Q_star.append((R[j], R[j+1]))
            middle = (R[j]+R[j+1])/2
            if Q(middle) >= C:
                coefficients_Q_star.append(np.array([0,0,C]))
                tau_Q_star.append(t)
            else:
                coefficients_Q_star.append(Q.coefficients[i])
                tau_Q_star.append(Q.taus[i])
    Q_star = QFunction(intervals_Q_star, coefficients_Q_star, tau_Q_star)
    Q_star.merge_similar_pieces()
    
    return Q_star



def get_breakpoints(taus):
    breakpoints = []
    current_point = len(taus)-1
    while current_point > 0:
        breakpoints.append(taus[current_point])
        current_point = taus[current_point]
    return breakpoints[::-1][1:]