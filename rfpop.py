import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class PiecewiseQuadratic:
    
    def __init__(self, intervals, coefficients):
        self.intervals = intervals
        self.coefficients = coefficients
        
    def __call__(self, x):
        for i in range(len(self.intervals)):
            if x > self.intervals[i][0] and x <= self.intervals[i][1]:
                return np.polyval(self.coefficients[i], x)
        return 0
    
    def min_on_interval(self, i):
        l, r = self.intervals[i]
        coefs = self.coefficients[i]
        a, b, c = coefs[0], coefs[1], coefs[2]
        if a==0:
            if b==0:
                return c
            if b>0:
                return np.polyval(coefs, l)
            if b<0:
                return np.polyval(coefs, r)
        if a>0:
            if -b/(2*a) > l and -b/(2*a) < r:
                return np.polyval(coefs, -b/(2*a))
            if -b/(2*a) <= l:
                return np.polyval(coefs, l)
            if -b/(2*a) >= r:
                return np.polyval(coefs, r)
        if a<0:
            r_val = np.polyval(coefs, l)
            l_val = np.polyval(coefs, r)
            return min(r_val, l_val)
    
    def roots_on_interval(self, i):
        r, l = self.intervals[i]
        coefs = self.coefficients[i]
        a, b, c = coefs[0], coefs[1], coefs[2]
        if a==0:
            if b==0:
                roots = []
            else:
                roots = [-c/b]
        else:
            delta = b**2 - 4*a*c
            if delta < 0:
                roots = []
            elif delta == 0:
                roots = [-b/(2*a)]
            else:
                if a>0:
                    roots = [(-b-np.sqrt(delta))/(2*a), (-b+np.sqrt(delta))/(2*a)]
                if a<0:
                    roots = [(-b+np.sqrt(delta))/(2*a), (-b-np.sqrt(delta))/(2*a)]
        return [root for root in roots if root > r and root < l]
                
    
class BiWeight(PiecewiseQuadratic):
    
    def __init__(self, y, K, y_min, y_max):
        """ The bi-weight is defined as:
            gamma(theta) = (y-theta)^2 if |y-theta| <= K
            gamma(theta) = K^2 otherwise
        """
        if y_min < y-K and y+K < y_max:
            intervals = [(y_min, y-K), (y-K, y+K), (y+K, y_max)]
            coefficients = [np.array([0,0,K**2]), np.array([1, -2*y, y**2]), np.array([0,0,K**2])]
        elif y_min >= y-K and y+K < y_max:
            intervals = [(y_min, y+K), (y+K, y_max)]
            coefficients = [np.array([1, -2*y, y**2]), np.array([0,0,K**2])]
        elif y_min < y-K and y+K >= y_max:
            intervals = [(y_min, y-K), (y-K, y_max)]
            coefficients = [np.array([0,0,K**2]), np.array([1, -2*y, y**2])]
        else:
            intervals = [(y_min, y_max)]
            coefficients = [np.array([1, -2*y, y**2])]
        PiecewiseQuadratic.__init__(self, intervals, coefficients)


def rfpop(y, K, beta, verbose=False):   
    
    n = y.shape[0]
    Q_star_intervals = [(np.min(y), np.max(y))]
    Q_star_coefficients = [np.array([0.,0.,0.])]
    Q_star = PiecewiseQuadratic(Q_star_intervals, Q_star_coefficients)
    tau_Q_star = [0]
    cp = []
    
    #print(f"Q_star.coefficients: {Q_star.coefficients}")
    #print(f"Q_star.intervals: {Q_star.intervals}")
    #print(f"tau_Q_star: {tau_Q_star}")
    for t in tqdm(range(n)):
        Q, tau_Q = algo2(Q_star, tau_Q_star, K, y, t)
        #print(f"Q.coefficients: {Q.coefficients}")
        #print(f"Q.intervals: {Q.intervals}")
        #print(f"tau_Q: {tau_Q}")
        Q_t, tau_t = algo3(Q, tau_Q)
        #print(f"Q_t: {Q_t}")
        #print(f"tau_t: {tau_t}")
        cp.append((Q_t, tau_t))
        C = Q_t + beta
        Q_star, tau_Q_star = algo4(Q, tau_Q, C, t)
        #print(f"Q_star.coefficients: {Q_star.coefficients}")
        #print(f"Q_star.intervals: {Q_star.intervals}")
        #print(f"tau_Q_star: {tau_Q_star}")
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
    for i in range(len(Q_star.intervals)):
        ax[0].axvline(Q_star.intervals[i][0], color='r', linestyle='--')
        ax[0].axvline(Q_star.intervals[i][1], color='r', linestyle='--')
    ax[1].plot(x, y_Q_star)
    ax[1].set_ylabel('Qt*')
    for i in range(len(Q_star.intervals)):
        ax[1].axvline(Q_star.intervals[i][0], color='r', linestyle='--')
        ax[1].axvline(Q_star.intervals[i][1], color='r', linestyle='--')
    plt.show()

def algo2(Q_star, tau_Q_star, K, y, t):
    #print("----- ALGO 2 -----")
    N_star = len(Q_star.intervals)
    gamma = BiWeight(y[t], K, np.min(y), np.max(y))
    
    l = len(gamma.intervals)
    intervals_Q = []
    coefficients_Q = []
    tau_Q = []
    
    N = 0
    i = 0
    j = 0
    
    while i < N_star and j < l:
        N += 1
        intervals_Q.append((max(Q_star.intervals[i][0], gamma.intervals[j][0]), min(Q_star.intervals[i][1], gamma.intervals[j][1])))
        coefficients_Q.append(Q_star.coefficients[i]+gamma.coefficients[j])
        tau_Q.append(tau_Q_star[i])
        if min(Q_star.intervals[i][1], gamma.intervals[j][1]) == Q_star.intervals[i][1]:
            i += 1
        else:
            j += 1
            
    return PiecewiseQuadratic(intervals_Q, coefficients_Q), tau_Q

def algo3(Q, tau_Q):
    #print("----- ALGO 3 -----")
    N = len(Q.intervals)
    
    Q_t = np.inf
    tau_t = 0
    
    for i in range(N):
        m = Q.min_on_interval(i)
        if m < Q_t:
            Q_t = m
            tau_t = tau_Q[i]
    
    return Q_t, tau_t

def algo4(Q, tau_Q, C, t):
    #print("----- ALGO 4 -----")
    N = len(Q.intervals)
    Q_minus_C = PiecewiseQuadratic(Q.intervals, [Q.coefficients[i]-np.array([0,0,C]) for i in range(N)])
    intervals_Q_star = []
    coefficients_Q_star = []
    tau_Q_star = []
    
    for i in range(N):
        R_tmp = Q_minus_C.roots_on_interval(i)
        #print(f"R_tmp: {R_tmp}")
        R = [Q.intervals[i][0]] + R_tmp + [Q.intervals[i][1]]
        #print(f"R: {R}")
        for j in range(len(R_tmp)+1):
            intervals_Q_star.append((R[j], R[j+1]))
            middle = (R[j]+R[j+1])/2
            if Q(middle) >= C:
                coefficients_Q_star.append(np.array([0,0,C]))
                tau_Q_star.append(t)
            else:
                coefficients_Q_star.append(Q.coefficients[i])
                tau_Q_star.append(tau_Q[i])
        #print(intervals_Q_star)
    
    return PiecewiseQuadratic(intervals_Q_star, coefficients_Q_star), tau_Q_star

def get_breakpoints(taus):
    breakpoints = []
    current_point = len(taus)-1
    while current_point > 0:
        breakpoints.append(taus[current_point])
        current_point = taus[current_point]
    return breakpoints[::-1][1:]