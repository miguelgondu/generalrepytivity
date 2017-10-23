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

    To-Do:
    1. add an incosistency check for values and (p,q).
    '''
    def __init__(self, basis, _type, dict_of_values):
        self.basis = basis
        self.covariant_dim = _type[0]
        self.contravariant_dim = _type[1]
        self.type = _type
        self.dict_of_values = dict_of_values
    
    def __getitem__(self, pair):
        a, b = pair
        if isinstance(a, int):
            if isinstance(b, int):
                if ((a, ), (b, )) in self.dict_of_values:
                    return self.dict_of_values[((a, ), (b, ))]
                else:
                    return 0
            if is_multiindex(b, len(self.basis), self.covariant_dim):
                if ((a, ), b) in self.dict_of_values:
                    return self.dict_of_values[((a, ), b)]
                else:
                    return 0
        if is_multiindex(a, len(self.basis), self.contravariant_dim):
            if isinstance(b, int):
                if (a, (b, )) in self.dict_of_values:
                    return self.dict_of_values[(a, (b, ))]
                else:
                    return 0
            if is_multiindex(b, len(self.basis), self.covariant_dim):
                if (a, b) in self.dict_of_values:
                    return self.dict_of_values[(a, b)]
                else:
                    return 0
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

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise ValueError('Cannot add a tensor with a {}'.format(type(other)))
        
        if other.basis != self.basis or other.type != self.type:
            raise ValueError('Tensors should be of the same type and have the same basis.')

        result_dict = self.dict_of_values.copy()
        for key in other.dict_of_values:
            if key in result_dict:
                result_dict[key] = result_dict[key] + other.dict_of_values[key]
            if key not in result_dict:
                result_dict[key] = other.dict_of_values[key]
        
        result_basis = self.basis
        result_type = self.type
        return Tensor(result_basis, result_type, result_dict)

    def subs(list_of_substitutions):
        for value in self.dict_of_values.items:
            value.subs(list_of_substitutions)

def tensor_from_matrix(matrix, basis):
    '''
    This function takes a square matrix and a basis and retruns a (0,2)-tensor in that basis.
    '''
    dict_of_values = {}
    for i in range(len(matrix.tolist())):
        for j in range(len(matrix.tolist())):
            dict_of_values[(i,j), None] = matrix[i, j]
    return Tensor(basis, (0, 2), dict_of_values)

class Metric:
    def __init__(self, _matrix, basis):
        self.matrix = _matrix
        self.basis = basis
        self.as_tensor = tensor_from_matrix(self.matrix, self.basis)