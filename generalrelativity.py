import sympy
import itertools

def get_all_multiindices(p, n):
    '''
    This function returns a list of all the tuples of the form (a_1, ..., a_p)
    with a_i between 1 and n-1. These tuples serve as multiindices for tensors.

    To-Do: I could just return the itertools.product iterable object, no?
    '''
    if p == 0:
        return [None]
    if p == 1:
        return [(k, ) for k in range(n)]
    if p > 1:
        return list(itertools.product(range(n), repeat=p))

def is_multiindex(multiindex, n, c_dimension):
    '''
    This function determines if a tuple (or None object) is a multiindex or not
    according to these rules:
    1. None is a multiindex of length 0 (i.e. if the covariant or contravariant dimension
    is 0, None is the only multiindex)
    2. The length of a multiindex must be equal to the c_dimension
    3. Each value in the multiindex varies between 0 and n-1.
    '''
    if multiindex == None:
        if c_dimension == 0:
            return True
        else:
            return False
    if len(multiindex) != c_dimension:
        return False
    for value in multiindex:
        if value < 0 or value >= n:
            return False

    return True

class Tensor:
    '''
    This class represents a tensor object in a given basis.

    To construct a (p,q)-Tensor, one must pass two arguments:
    1. _type: a pair of values p (the covariant dimension) and q (the contravariant dimension).
    2. the non-zero values, which is a dict whose keys are pairs of the form (a, b)
    where a and b are multi-indices such that $\Gamma^a_b = value$, the values that
    don't appear in this dict are assumed to be 0.
    '''
    def __init__(self, basis, _type, dict_of_values):
        self.basis = basis
        self.covariant_dim = _type[0]
        self.contravariant_dim = _type[1]
        self.type = _type

        for key in dict_of_values:
            a, b = key
            if not is_multiindex(a, len(self.basis), self.covariant_dim):
                raise ValueError('The multiindex {} is inconsistent with dimension {}'.format(a, self.covariant_dim))
            if not is_multiindex(b, len(self.basis), self.contravariant_dim):
                raise ValueError('The multiindex {} is inconsistent with dimensions {}'.format(b, self.contravariant_dim))

        self.dict_of_values = dict_of_values

    def __eq__(self, other):
        if self.basis == other.basis:
            if self.type == other.type:
                if self.get_all_values() == other.get_all_values():
                    return True
        return False

    def __getitem__(self, pair):
        a, b = pair
        if isinstance(a, int):
            if isinstance(b, int):
                if (is_multiindex((a, ), len(self.basis), self.covariant_dim) and
                    is_multiindex((b, ), len(self.basis), self.contravariant_dim)):
                    if ((a, ), (b, )) in self.dict_of_values:
                        return self.dict_of_values[((a, ), (b, ))]
                    else:
                        return 0
                else:
                    raise KeyError('There\'s a problem with the multiindices ({}, ) and ({}, )'.format(a, b))
            if is_multiindex(b, len(self.basis), self.contravariant_dim):
                if ((a, ), b) in self.dict_of_values:
                    return self.dict_of_values[((a, ), b)]
                else:
                    return 0
        if is_multiindex(a, len(self.basis), self.covariant_dim):
            if isinstance(b, int):
                if (a, (b, )) in self.dict_of_values:
                    return self.dict_of_values[(a, (b, ))]
                else:
                    return 0
            if is_multiindex(b, len(self.basis), self.contravariant_dim):
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
                    substring_of_a += '{} \\otimes '.format(self.basis[ind])
                substring_of_a = substring_of_a[:-len(' \\otimes ')]
                string += substring_of_a
            if b != None:
                if a != None:
                    string += ' \\otimes '
                for ind in b:
                    string += '{}* \\otimes '.format(self.basis[ind])
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

    def subs(self, list_of_substitutions):
        new_dict = {}
        for key, value in self.dict_of_values.items():
            new_dict[key] = value.subs(list_of_substitutions)
        return Tensor(self.basis, self.type, new_dict)

    def get_all_values(self):
        new_dict = {}
        dim = len(self.basis)
        covariant_multiindices = get_all_multiindices(self.covariant_dim, dim)
        contravariant_multiindices = get_all_multiindices(self.contravariant_dim, dim)
        for a in covariant_multiindices:
            for b in contravariant_multiindices:
                if (a,b) in self.dict_of_values:
                    new_dict[a, b] = self.dict_of_values[a, b]
                else:
                    new_dict[a, b] = 0
        return new_dict

def tensor_from_matrix(matrix, basis):
    '''
    This function takes a square matrix and a basis and retruns a (0,2)-tensor in that basis.
    '''
    dict_of_values = {}
    for i in range(len(matrix.tolist())):
        for j in range(len(matrix.tolist())):
            dict_of_values[None, (i,j)] = matrix[i, j]
    return Tensor(basis, (0, 2), dict_of_values)

class Metric:
    def __init__(self, _matrix, basis):
        if _matrix != _matrix.T:
            raise ValueError('Matrix should be symmetric.')
        if _matrix.det() == 0:
            raise ValueError('Matrix should be non-singular.')
        self.matrix = _matrix
        self.basis = basis
        self.as_tensor = tensor_from_matrix(self.matrix, self.basis)
    
    def __getitem__(self, key):
        return self.as_tensor[key]

def contract_indices(tensor, i, j):
    '''
    Returns the resulting tensor of formally contracting the indices i and j 
    of the given tensor.
    '''
    dim = len(tensor.basis)
    covariant_dim = tensor.covariant_dim
    contravariant_dim = tensor.contravariant_dim
    if covariant_dim < 1 or contravariant_dim < 1:
        raise ValueError('One of the dimensions in the type {} is less than one.'.format(tensor.type))
    if i < 0 or i >= covariant_dim:
        raise ValueError('{} is either negative or bigger than the contravariant dimension minus one {}'.format(i,
                                                                                    covariant_dim-1))
    if j < 0 or j >= contravariant_dim:
        raise ValueError('{} is either negative or bigger than the covariant dimension minus one {}'.format(j,
                                                                                    contravariant_dim-1))

    covariant_indices = get_all_multiindices(covariant_dim-1, dim)
    contravariant_indices = get_all_multiindices(contravariant_dim-1, dim)
    new_tensor_dict = {}
    for a in covariant_indices:
        for b in contravariant_indices:
            sumand = 0
            for r in range(dim):
                if a != None:
                    a_extended = a[:i] + (r, ) + a[i:]
                if a == None:
                    a_extended = (r, )
                if b != None:
                    b_extended = b[:j] + (r, ) + b[j:]
                if b == None:
                    b_extended = (r, )
                sumand += tensor[a_extended, b_extended]
            new_tensor_dict[a, b] = sumand

    return Tensor(tensor.basis, (covariant_dim - 1, contravariant_dim - 1), new_tensor_dict)

def lower_index(tensor, metric, i):
    if isinstance(metric, Metric):
        if metric.basis != tensor.basis:
            raise ValueError('Tensor and Metric should be on the same basis.')
    else:
        metric = Metric(metric, tensor.basis)
    
    if tensor.covariant_dim == 0:
        raise ValueError('There\'s no index to be lowered.')

    if i < 0 or i >= tensor.covariant_dim:
        raise ValueError('The index to be lowered ({}) must be between 0 and {}'.format(i,
                                                                        tensor.covariant_dim))

    basis = tensor.basis
    dim = len(basis)
    new_covariant_dim = tensor.covariant_dim - 1
    new_contravariant_dim = tensor.contravariant_dim + 1
    new_type = (new_covariant_dim, new_contravariant_dim)
    covariant_indices = get_all_multiindices(new_covariant_dim, dim)
    contravariant_indices = get_all_multiindices(new_contravariant_dim, dim)

    new_tensor_dict = {}
    for a in covariant_indices:
        for b in contravariant_indices:
            value = 0
            for r in range(dim):
                if a == None:
                    a_extended = (r, )
                if a != None:
                    a_extended = a[:i] + (r, ) + a[i:]
                b_reduced = b[1:]
                value += metric[None, (b[0], r)]*tensor[a_extended, b_reduced]
            new_tensor_dict[a, b] = value

    return Tensor(basis, new_type, new_tensor_dict)

def raise_index(tensor, metric, j):
    if isinstance(metric, Metric):
        if metric.basis != tensor.basis:
            raise ValueError('Tensor and Metric should be on the same basis.')
    else:
        metric = Metric(metric, tensor.basis)

    if tensor.contravariant_dim == 0:
        raise ValueError('There\'s no index to be lowered.')

    if j < 0 or j >= tensor.covariant_dim:
        raise ValueError('The index to be raised ({}) must be between 0 and {}'.format(j,
                                                                        tensor.convariant_dim))

    basis = tensor.basis
    dim = len(basis)
    new_covariant_dim = tensor.covariant_dim + 1
    new_contravariant_dim = tensor.contravariant_dim - 1
    new_type = (new_covariant_dim, new_contravariant_dim)
    covariant_indices = get_all_multiindices(new_covariant_dim, dim)
    contravariant_indices = get_all_multiindices(new_contravariant_dim, dim)
    inverse_metric_matrix = metric.matrix.inv()

    new_tensor_dict = {}
    for a in covariant_indices:
        for b in contravariant_indices:
            value = 0
            for r in range(dim):
                # Here, a[-1] is b_j.
                a_reduced = a[:-1]
                if b == None:
                    b_expanded = (r, )
                if b != None:
                    b_expanded = b[:j] + (r, ) + b[j:]
                value += inverse_metric_matrix[a[-1], r]*tensor[a_reduced, b_expanded]
            new_tensor_dict[a, b] = value

    return Tensor(basis, new_type, new_tensor_dict)


def _dict_completer(_dict, c_dimension, ct_dimension, dim):
    c_indices = get_all_multiindices(c_dimension, dim)
    ct_indices = get_all_multiindices(ct_dimension, dim)
    new_dict = _dict.copy()
    for a in c_indices:
        for b in ct_indices:
            if (a,b) in new_dict:
                pass
            if (a,b) not in new_dict:
                new_dict[a,b] = 0
    return new_dict

def _symmetry_completer(_dict):
    new_dict = _dict.copy()
    for a, b in new_dict.keys():
        inverted_b = (b[1], b[0])
        if (a, inverted_b) in new_dict:
            if new_dict[a, b] != new_dict[a, inverted_b]:
                raise ValueError('Inconsistent values for pairs {} and {} of subindices'.format(b, inverted_b))
        if (a, inverted_b) not in new_dict:
            new_dict[a, inverted_b] = new_dict[a, b]
    return new_dict

class ChristoffelSymbols:
    def __init__(self, basis, _dict):
        self._internal_dict = _symmetry_completer(_dict)
        self.dim = len(basis)
        self.basis = basis
    
    def __getitem__(self, pair):
        a, b = pair
        if isinstance(a, int):
            if is_multiindex(b, len(self.basis), 2):
                if ((a, ), b) in self._internal_dict:
                    return self._internal_dict[((a, ), b)]
                else:
                    return 0
            else:
                raise KeyError('{} should be a pair (i.e. a multiindex of length 2)'.format(b))
        elif is_multiindex(a, len(self.basis), 1):
            if is_multiindex(b, len(self.basis), 2):
                if (a, b) in self._internal_dict:
                    return self._internal_dict[(a, b)]
                else:
                    return 0
            else:
                raise KeyError('{} should be a pair (i.e. a multiindex of length 2)'.format(b))
        raise KeyError('There\'s something wrong with the pair of multiindices {} and {}'.format(a, b))
    
    def get_all_values(self):
        return _dict_completer(self._internal_dict, 1, 2, self.dim)

def get_chrisoffel_symbols_from_metric(metric):
    basis = metric.basis
    dim = len(basis)
    inverse_metric_matrix = metric.matrix.inv()
    _type = (1,2)
    covariant_indices = get_all_multiindices(1, dim)
    contravariant_indices = get_all_multiindices(2, dim)
    dict_of_values = {}
    for a in covariant_indices:
        for b in contravariant_indices:
            i, j = b
            c = a[0]
            sumand = 0
            for r in range(dim):
                L = (metric.matrix[j, r].diff(basis[i])
                     + metric.matrix[i, r].diff(basis[j])
                     - metric.matrix[i, j].diff(basis[r]))
                sumand += inverse_metric_matrix[r, c] * L
            if sumand != 0:
                dict_of_values[a, b] = (1/2) * sumand
    return ChristoffelSymbols(basis, dict_of_values)

class Universe:
    def __init__(self, _metric):
        s, t, x, y, z = sympy.symbols('s t x y z')
        self.metric = _metric
        self.basis = _metric.basis
        self.connection = get_connection_from_metric(_metric)
