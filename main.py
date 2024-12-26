import numpy as np

class PiecewiseQuadratic:
    
    def __init__(self, intervals, coefficients):
        self.intervals = intervals
        self.coefficients = coefficients
        
    def __call__(self, x):
        for i in range(len(self.intervals)):
            if x > self.intervals[i][0] and x <= self.intervals[i][1]:
                return np.polyval(self.coefficients[i], x)
        return 0
    
class BiWeight(PiecewiseQuadratic):
    
    def __init__(self, y, K):
        """ The bi-weight is defined as:
            gamma(theta) = (y-theta)^2 if |y-theta| <= K
            gamma(theta) = K^2 otherwise
        """
        intervals = [(-np.inf, y-K), (y-K, y+K), (y+K, np.inf)]
        coefficients = [[K**2], [1, -2*y, y**2], [K**2]]
        PiecewiseQuadratic.__init__(self, intervals, coefficients)
        
    def __call__(self, x):
        for i in range(len(self.intervals)):
            if x > self.intervals[i][0] and x <= self.intervals[i][1]:
                return np.polyval(self.coefficients[i], x)
        return 0


def algo1(y, K, beta):   
    
    n = y.shape[0]
    Q_star = PiecewiseQuadratic([(np.min(y), np.max(y))], [[0]])
    tau_Q_star = [0]
    cp = np.zeros(n)
    
    for t in range(n):
        Q, tau_Q = algo2(Q_star, tau_Q_star, K, y, t)
        Q_t, tau_t = algo3(Q, tau_Q)
        cp[t] = (Q_t, tau_t)
        C = Q_t + beta
        Q_star, tau_Q_star = algo4(Q, tau_Q, C)
    
    return cp

def algo2(Q_star, tau_Q_star, K, y, t):
    N_star = len(Q_star.intervals)
    gamma = BiWeight(y[t], K)
    l = len(gamma.intervals)
    intervals_Q = []
    
    N = 0
    i = 1
    j = 1
    
    while i <= N_star and j <= l:
        N += 1
        intervals_Q.append((