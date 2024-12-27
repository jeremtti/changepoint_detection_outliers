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
        

class QFunction(PiecewiseQuadratic):
    
    def __init__(self, intervals, coefficients, taus):
        PiecewiseQuadratic.__init__(self, intervals, coefficients)
        self.taus = taus
    
    def merge_similar_pieces(self):
        new_intervals = [self.intervals[0]]
        new_coefficients = [self.coefficients[0]]
        new_taus = [self.taus[0]]
        
        for i in range(1, len(self.intervals)):
            if not np.allclose(self.coefficients[i], new_coefficients[-1], rtol=1e-5):
                new_intervals.append(self.intervals[i])
                new_coefficients.append(self.coefficients[i])
                new_taus.append(self.taus[i])
            else:
                new_intervals[-1] = (new_intervals[-1][0], self.intervals[i][1])
        
        self.intervals = new_intervals
        self.coefficients = new_coefficients
        self.taus = new_taus