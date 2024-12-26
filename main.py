import numpy as np

# QUESTION:
# WHY IS IT BETTER THAN "PELT" ALGORITHM? OR OTHER PRUNING ALGORITHMS...


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
            return np.polyval(coefs, -b/(2*a))
        if a<0:
            r_val = np.polyval(coefs, l)
            l_val = np.polyval(coefs, r)
            return min(r_val, l_val)
    
    def roots_on_interval(self, i):
        coefs = self.coefficients[i]
        a, b, c = coefs[0], coefs[1], coefs[2]
        if a==0:
            if b==0:
                return []
            return [-c/b]
        delta = b**2 - 4*a*c
        if delta < 0:
            return []
        if delta == 0:
            return [-b/(2*a)]
        if a>0:
            return [(-b-np.sqrt(delta))/(2*a), (-b+np.sqrt(delta))/(2*a)]
        if a<0:
            return [(-b+np.sqrt(delta))/(2*a), (-b-np.sqrt(delta))/(2*a)]
        
    
class BiWeight(PiecewiseQuadratic):
    
    def __init__(self, y, K):
        """ The bi-weight is defined as:
            gamma(theta) = (y-theta)^2 if |y-theta| <= K
            gamma(theta) = K^2 otherwise
        """
        intervals = [(-np.inf, y-K), (y-K, y+K), (y+K, np.inf)]
        coefficients = [np.array([0,0,K**2]), np.array([1, -2*y, y**2]), np.array([0,0,K**2])]
        PiecewiseQuadratic.__init__(self, intervals, coefficients)
        
    def __call__(self, x):
        for i in range(len(self.intervals)):
            if x > self.intervals[i][0] and x <= self.intervals[i][1]:
                return np.polyval(self.coefficients[i], x)
        return 0


def algo1(y, K, beta):   
    
    n = y.shape[0]
    Q_star = PiecewiseQuadratic([(np.min(y), np.max(y))], [np.array([0,0,0])])
    tau_Q_star = [0]
    cp = np.zeros(n)
    
    for t in range(n):
        Q, tau_Q = algo2(Q_star, tau_Q_star, K, y, t)
        Q_t, tau_t = algo3(Q, tau_Q)
        cp[t] = (Q_t, tau_t)
        C = Q_t + beta
        Q_star, tau_Q_star = algo4(Q, tau_Q, C, t)
    
    return cp

def algo2(Q_star, tau_Q_star, K, y, t):
    N_star = len(Q_star.intervals)
    gamma = BiWeight(y[t], K)
    # GAMMA SUPPOSED TO BE DEFINED ON SAME INTERVAL AS Q_STAR :(((
    
    l = len(gamma.intervals)
    intervals_Q = []
    coefficients_Q = []
    tau_Q = []
    
    N = 0
    i = 1
    j = 1
    
    while i <= N_star and j <= l:
        N += 1
        intervals_Q.append((max(Q_star.intervals[i-1][0], gamma.intervals[j-1][0]), min(Q_star.intervals[i-1][1], gamma.intervals[j-1][1])))
        coefficients_Q.append(Q_star.coefficients[i-1]+gamma.coefficients[j-1])
        tau_Q.append(tau_Q_star[i-1])
        if min(Q_star.intervals[i-1][1], gamma.intervals[j-1][1]) == Q_star.intervals[i-1][1]:
            i += 1
        else:
            j += 1
            
    return PiecewiseQuadratic(intervals_Q, coefficients_Q), tau_Q

def algo3(Q, tau_Q):
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
    N = len(Q.intervals)
    Q_minus_C = PiecewiseQuadratic(Q.intervals, [Q.coefficients[i]-C for i in range(N)])
    intervals_Q_star = []
    coefficients_Q_star = []
    tau_Q_star = []
    
    for i in range(N):
        R_tmp = Q_minus_C.roots_on_interval(i)
        R = [Q.intervals[i][0]] + R_tmp + [Q.intervals[i][1]]
        for j in range(len(R_tmp)+1):
            intervals_Q_star.append((R[j], R[j+1]))
            middle = (R[j]+R[j+1])/2
            if Q(middle) >= C:
                coefficients_Q_star.append(C)
                tau_Q_star.append(t)
            else:
                coefficients_Q_star.append(Q.coefficients[i])
                tau_Q_star.append(tau_Q[i])
    
    return PiecewiseQuadratic(intervals_Q_star, coefficients_Q_star), tau_Q_star