import sympy

def is_multiindex(multiindex, n, dimension):
    if multiindex == None:
        return True
    if len(multiindex) > dimension:
        return False
    for value in multiindex:
        if value < 0 or value >= n:
            return False
    
    return True

class Tensor:
    '''
    This class represents a tensor object in a given basis.

    To construct a (p,q)-Tensor, one must pass two arguments:
    1. type: a pair of values p and q.
    2. the non-zero values, which is a dict whose value are pairs of the form (a, b)
    where a and b are multi-indices such that $\Gamma^b_a = value$, the values that
    don't appear in this list are assumed to be 0.
    '''
    def __init__(self, basis, type, dict_of_values):
        self.basis = basis
        self.covariant_dim = type[0]
        self.contravariant_dim = type[1]
        self.dict_of_values = dict_of_values
    
    def __getitem__(self, pair):
        a, b = pair
        if is_multiindex(a, len(self.basis), self.covariant_dim) and is_multiindex(b, len(self.basis), self.contravariant_dim):
            if (a, b) in self.dict_of_values:
                return self.dict_of_values[(a, b)]
            else:
                return 0
        else:
            raise KeyError

    def __repr__(self):
        string = ''
        for key in self.dict_of_values:
            string += '({})'.format(self.dict_of_values[key])
            a, b = key
            if b != None:
                for index in b:
                    string += '{}*\\otimes'.format(self.basis[index])
                string = string[:-len('\\otimes')]
            if a != None:
                for index in a:
                    string += '{}\\otimes'.format(self.basis[index])
                    string = string[:-len('\\otimes')]
            string += ' + '
        string = string[:-3]
        return string
