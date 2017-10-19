import sympy

def is_multiindex(multiindex, n, dimension):
    if multiindex == None:
        if dimension == 0:
            return True
        else:
            return False
    if len(multiindex) != dimension:
        return False
    for value in multiindex:
        if value < 0 or value >= n:
            return False
    
    return True

class Tensor:
    '''
    This class represents a tensor object in a given basis.

    To construct a (p,q)-Tensor, one must pass two arguments:
    1. _type: a pair of values p and q.
    2. the non-zero values, which is a dict whose value are pairs of the form (a, b)
    where a and b are multi-indices such that $\Gamma^a_b = value$, the values that
    don't appear in this list are assumed to be 0.
    '''
    def __init__(self, basis, _type, dict_of_values):
        self.basis = basis
        self.covariant_dim = _type[0]
        self.contravariant_dim = _type[1]
        self.dict_of_values = dict_of_values
    
    def __getitem__(self, pair):
        a, b = pair
        if (is_multiindex(a, len(self.basis), self.contravariant_dim)
             and is_multiindex(b, len(self.basis), self.covariant_dim)):
            if (a, b) in self.dict_of_values:
                return self.dict_of_values[(a, b)]
            else:
                return 0
        else:
            raise KeyError('There\'s something wrong with the pair of multiindices {} and {}'.format(a, b))

    def __repr__(self):
        string = ''
        for key in self.dict_of_values:
            string += '({})'.format(self.dict_of_values[key])
            a, b = key
            if a != None:
                substring_of_a = ''
                for ind in a:
                    substring_of_a += '{}* \\otimes '.format(self.basis[ind])
                substring_of_a = substring_of_a[:-len(' \\otimes ')]
                string += substring_of_a
            if b != None:
                if a != None:
                    string += ' \\otimes '
                for ind in b:
                    string += '{} \\otimes '.format(self.basis[ind])
                string = string[:-len(' \\otimes ')]
            string += ' + '
        string = string[:-3]
        return string
