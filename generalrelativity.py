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
    elif isinstance(multiindex, tuple):
        if len(multiindex) != c_dimension:
            return False
        for value in multiindex:
            if isinstance(value, int) or isinstance(value, float):
                if value < 0 or value >= n:
                    return False
            else:
                return False
        return True
    else:
        return False

def _is_valid_key(key, dim, contravariant_dim, covariant_dim):
    try:
        a, b = key
        if not is_multiindex(a, dim, contravariant_dim):
            raise ValueError('The multiindex {} is inconsistent with dimension {}'.format(a, contravariant_dim))
        if not is_multiindex(b, dim, covariant_dim):
            raise ValueError('The multiindex {} is inconsistent with dimensions {}'.format(b, covariant_dim))
        return True
    except ValueError:
        return False

def _dict_completer_for_tensor(_dict, _type, dim):
    contravariant_dim = _type[0]
    covariant_dim = _type[1]

    new_dict = {}
    if contravariant_dim > 0 and covariant_dim == 0:
        for key in _dict:
            if _is_valid_key(key, dim, contravariant_dim, covariant_dim):
                new_dict[key] = _dict[key]
            elif contravariant_dim == 1 and isinstance(key, int):
                new_dict[((key, ), None)] = _dict[key]
            elif is_multiindex(key, dim, contravariant_dim):
                new_dict[(key, None)] = _dict[key]
            else:
                raise ValueError('Can\'t extend the key {} because it isn\'t a {}-multiindex'.format(
                    key, contravariant_dim))
        return new_dict
    
    if contravariant_dim == 0 and covariant_dim > 0:
        for key in _dict:
            if _is_valid_key(key, dim, contravariant_dim, covariant_dim):
                new_dict[key] = _dict[key]
            elif covariant_dim == 1 and isinstance(key, int):
                new_dict[None, (key, )] = _dict[key]
            elif is_multiindex(key, dim, covariant_dim):
                new_dict[(None, key)] = _dict[key]
            else:
                raise ValueError('Can\'t extend the key {} because it isn\'t a {}-multiindex'.format(
                    key, covariant_dim))
        return new_dict

    if contravariant_dim == 1 and covariant_dim > 0:
        for key in _dict:
            if _is_valid_key(key, dim, contravariant_dim, covariant_dim):
                new_dict[key] = _dict[key]
            elif len(key) == 2:
                i, b = key
                if isinstance(i, int) and isinstance(b, int):
                    new_dict[((i, ), (b, ))] = _dict[key]
                elif isinstance(i, int) and is_multiindex(b, dim, covariant_dim):
                    new_dict[(i, ), b] = _dict[key]
                else:
                    raise ValueError('{} should be an integer and {} should be a {}-multiindex (or int in case 1).'.format(
                                                                            i, b, covariant_dim))
            else:
                raise ValueError('There should only be two things in {}'.format(key))
        return new_dict
    
    return _dict


class Tensor:
    '''
    This class represents a tensor object in a given basis.

    To construct a (p,q)-Tensor, one must pass two arguments:
    1. _type: a pair of values p (the contravariant dimension) and q (the covariant dimension).
    2. the non-zero values, which is a dict whose keys are pairs of the form (a, b)
    where a and b are multi-indices such that $\Gamma^a_b = value$, the values that
    don't appear in this dict are assumed to be 0.
    
    To-Do:
        -Redo indexing for the None cases.
        -Fix multiplication on (0,0) case.
    '''
    def __init__(self, basis, _type, dict_of_values):
        self.basis = basis
        self.contravariant_dim = _type[0]
        self.covariant_dim = _type[1]
        self.type = _type
        self.dim = len(self.basis)

        temp_dict = _dict_completer_for_tensor(dict_of_values, _type, self.dim)

        for key in temp_dict:
            a, b = key
            if not is_multiindex(a, len(self.basis), self.contravariant_dim):
                raise ValueError('The multiindex {} is inconsistent with dimension {}'.format(a, self.contravariant_dim))
            if not is_multiindex(b, len(self.basis), self.covariant_dim):
                raise ValueError('The multiindex {} is inconsistent with dimensions {}'.format(b, self.covariant_dim))

        self.dict_of_values = temp_dict

    def __eq__(self, other):
        if self.basis == other.basis:
            if self.type == other.type:
                if self.get_all_values() == other.get_all_values():
                    return True
        return False

    def __getitem__(self, pair):
        if self.contravariant_dim == 0 and self.covariant_dim > 0:
            if is_multiindex(pair, len(self.basis), self.covariant_dim):
                if (None, pair) in self.dict_of_values:
                    return self.dict_of_values[(None, pair)]
                else:
                    return 0
        
        if self.covariant_dim == 0 and self.contravariant_dim > 0:
            if is_multiindex(pair, len(self.basis), self.contravariant_dim):
                if (pair, None) in self.dict_of_values:
                    return self.dict_of_values[(pair, None)]
                else:
                    return 0
        
        if self.covariant_dim == 1 and self.contravariant_dim == 1:
            if len(pair) == 2:
                i, j = pair
                if isinstance(i, int) and isinstance(j, int):
                    if ((i, ), (j, )) in self.dict_of_values:
                        return self.dict_of_values[(i, ), (j, )]
                    else:
                        return 0
            else:
                raise KeyError('There should be two things in {}, but there are {}'.format(pair, len(pair)))

        a, b = pair
        if isinstance(a, int):
            if isinstance(b, int):
                if (is_multiindex((a, ), len(self.basis), self.contravariant_dim) and
                    is_multiindex((b, ), len(self.basis), self.covariant_dim)):
                    if ((a, ), (b, )) in self.dict_of_values:
                        return self.dict_of_values[((a, ), (b, ))]
                    else:
                        return 0
                else:
                    raise KeyError('There\'s a problem with the multiindices ({}, ) and ({}, )'.format(a, b))
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

    def _repr_latex_(self):
        string = '$'
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
                    string += '{}^* \\otimes '.format(self.basis[ind])
                string = string[:-len(' \\otimes ')]
            string += ' + '
        string = string[:-3]
        return (string + '$').replace('**', '^')

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
        
        for key in result_dict.copy():
            if result_dict[key] == 0:
                empty = result_dict.pop(key)
        
        result_basis = self.basis
        result_type = self.type
        return Tensor(result_basis, result_type, result_dict)

    def __mul__(self, other):
        if self.type == (0,0):
            if isinstance(other, Tensor):
                if other.basis != self.basis:
                    raise ValueError('The basis of {} should be the same as the other tensor'.format(other))
                new_dict = other.dict_of_values.copy()
                for key in other.dict_of_values:
                    new_dict[key] = new_dict[key] * self.dict_of_values[(None, None)]
                return Tensor(self.basis, other.type, new_dict)
        if isinstance(other, int) or isinstance(other, float):
            new_dict = self.dict_of_values.copy()
            for key in self.dict_of_values:
                new_dict[key] = self.dict_of_values[key] * other
            return Tensor(self.basis, self.type, new_dict)
        
        if isinstance(other, Tensor):
            if other.type != (0,0):
                raise ValueError('Can\'t multiply a tensor with a tensor that isn\'t (0,0)')
            if other.basis != self.basis:
                raise ValueError('The basis of {} should be the same as the other tensor'.format(other))
            
            other_value = other[None, None]
            new_dict = self.dict_of_values.copy()
            for key in self.dict_of_values:
                new_dict[key] = self.dict_of_values[key] * other_value
            return Tensor(self.basis, self.type, new_dict)
            
        raise ValueError('{} must be either an int, a float or a (0,0)-Tensor'.format(other))
    
    __rmul__ = __mul__

    def subs(self, list_of_substitutions):
        new_dict = {}
        for key, value in self.dict_of_values.items():
            new_dict[key] = value.subs(list_of_substitutions)
        return Tensor(self.basis, self.type, new_dict)

    def get_all_values(self):
        new_dict = {}
        dim = len(self.basis)
        contravariant_multiindices = get_all_multiindices(self.contravariant_dim, dim)
        covariant_multiindices = get_all_multiindices(self.covariant_dim, dim)
        for a in contravariant_multiindices:
            for b in covariant_multiindices:
                if (a,b) in self.dict_of_values:
                    new_dict[a, b] = self.dict_of_values[a, b]
                else:
                    new_dict[a, b] = 0
        return new_dict

def get_tensor_from_matrix(matrix, basis):
    '''
    This function takes a square matrix and a basis and retruns a (0,2)-tensor in that basis.
    '''
    dict_of_values = {}
    for i in range(len(matrix.tolist())):
        for j in range(len(matrix.tolist())):
            if matrix[i, j] != 0:
                dict_of_values[None, (i,j)] = matrix[i, j]
    return Tensor(basis, (0, 2), dict_of_values)

def get_matrix_from_tensor(tensor):
    '''
    This function takes an (0,2)-tensor and returns it matrix representation.
    '''
    matrix = sympy.zeros(len(tensor.basis))
    for i in range(len(tensor.basis)):
        for j in range(len(tensor.basis)):
            matrix[i, j] = tensor[None, (i,j)]
    
    return matrix

class Metric:
    def __init__(self, _matrix, basis):
        if _matrix != _matrix.T:
            raise ValueError('Matrix should be symmetric.')
        if _matrix.det() == 0:
            raise ValueError('Matrix should be non-singular.')
        self.matrix = _matrix
        self.basis = basis
        self.as_tensor = get_tensor_from_matrix(self.matrix, self.basis)
    
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
    if i < 0 or i >= contravariant_dim:
        raise ValueError('{} is either negative or bigger than the contravariant dimension minus one {}'.format(i,
                                                                                    contravariant_dim-1))
    if j < 0 or j >= covariant_dim:
        raise ValueError('{} is either negative or bigger than the covariant dimension minus one {}'.format(j,
                                                                                    covariant_dim-1))

    contravariant_indices = get_all_multiindices(contravariant_dim-1, dim)
    covariant_indices = get_all_multiindices(covariant_dim-1, dim)
    new_tensor_dict = {}
    for a in contravariant_indices:
        for b in covariant_indices:
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

    return Tensor(tensor.basis, (contravariant_dim - 1, covariant_dim - 1), new_tensor_dict)

def lower_index(tensor, metric, i):
    if isinstance(metric, Metric):
        if metric.basis != tensor.basis:
            raise ValueError('Tensor and Metric should be on the same basis.')
    elif isinstance(metric, Tensor):
        if metric.basis != tensor.basis:
            raise ValueError('Tensor and Metric should be on the same basis.')
        metric = Metric(metric, tensor.basis)
    else:
        raise ValueError('metric should be either a tensor or a metric object.')
    
    if tensor.contravariant_dim == 0:
        raise ValueError('There\'s no index to be lowered.')

    if i < 0 or i >= tensor.contravariant_dim:
        raise ValueError('The index to be lowered ({}) must be between 0 and {}'.format(i,
                                                                        tensor.contravariant_dim))

    basis = tensor.basis
    dim = len(basis)
    new_contravariant_dim = tensor.contravariant_dim - 1
    new_covariant_dim = tensor.covariant_dim + 1
    new_type = (new_contravariant_dim, new_covariant_dim)
    contravariant_indices = get_all_multiindices(new_contravariant_dim, dim)
    covariant_indices = get_all_multiindices(new_covariant_dim, dim)

    new_tensor_dict = {}
    for a in contravariant_indices:
        for b in covariant_indices:
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

    if tensor.covariant_dim == 0:
        raise ValueError('There\'s no index to be lowered.')

    if j < 0 or j >= tensor.covariant_dim:
        raise ValueError('The index to be raised ({}) must be between 0 and {}'.format(j,
                                                                        tensor.covariant_dim))

    basis = tensor.basis
    dim = len(basis)
    new_contravariant_dim = tensor.contravariant_dim + 1
    new_covariant_dim = tensor.covariant_dim - 1
    new_type = (new_contravariant_dim, new_covariant_dim)
    contravariant_indices = get_all_multiindices(new_contravariant_dim, dim)
    covariant_indices = get_all_multiindices(new_covariant_dim, dim)
    inverse_metric_matrix = metric.matrix.inv()

    new_tensor_dict = {}
    for a in contravariant_indices:
        for b in covariant_indices:
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

def _symmetry_completer(_dict):
    new_dict = _dict.copy()
    for a, b in _dict.keys():
        inverted_b = (b[1], b[0])
        if (a, inverted_b) in new_dict:
            if new_dict[a, b] != new_dict[a, inverted_b]:
                raise ValueError('Inconsistent values for pairs {} and {} of subindices'.format(b, inverted_b))
        if (a, inverted_b) not in new_dict:
            new_dict[a, inverted_b] = new_dict[a, b]
    return new_dict

def _dict_completer(_dict, c_dimension, ct_dimension, dim):
    c_indices = get_all_multiindices(c_dimension, dim)
    ct_indices = get_all_multiindices(ct_dimension, dim)
    new_dict = _symmetry_completer(_dict)
    for a in c_indices:
        for b in ct_indices:
            if (a,b) in new_dict:
                pass
            if (a,b) not in new_dict:
                new_dict[a,b] = 0
    return new_dict

class ChristoffelSymbols:
    def __init__(self, basis, _dict, _metric):
        self._internal_dict = _symmetry_completer(_dict)
        self.dim = len(basis)
        self.basis = basis
        self.metric = _metric
    
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
    contravariant_indices = get_all_multiindices(1, dim)
    covariant_indices = get_all_multiindices(2, dim)
    dict_of_values = {}
    for a in contravariant_indices:
        for b in covariant_indices:
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
    return ChristoffelSymbols(basis, dict_of_values, metric)

def get_Riemann_tensor(christoffel_symbols):
    cs = christoffel_symbols
    ct_dimension = 1
    c_dimension = 3
    dim = len(christoffel_symbols.basis)
    contravariant_indices = get_all_multiindices(ct_dimension, dim)
    covariant_indices = get_all_multiindices(c_dimension, dim)
    dict_of_values = {}
    for x in contravariant_indices:
        for y in covariant_indices:
            d = x[0]
            c, a, b = y
            sumand = cs[d, (b,c)].diff(basis[a]) - cs[d, (a,c)].diff(basis[b])
            for u in range(dim):
                sumand += cs[d, (a, u)]*cs[u, (c,b)] - cs[d, (b, u)]*cs[u, (c, a)]
            if sumand != 0:
                dict_of_values[x, y] = sumand
    return Tensor(cs.basis, (1, 3), dict_of_values)

def get_Ricci_tensor(christoffel_symbols):
    Riem = get_Riemann_tensor(christoffel_symbols)
    return contract_indices(Riem, 0, 1)

def get_scalar_curvature(christoffel_symbols):
    Temp = get_Ricci_tensor(christoffel_symbols)
    Temp = raise_index(Temp, christoffel_symbols.metric, 0)
    return contract_indices(Temp, 0, 0)

def get_Einstein_tensor(christoffel_symbols):
    Ric = get_Ricci_tensor(christoffel_symbols)
    R = get_scalar_curvature(christoffel_symbols)
    g = christoffel_symbols.metric
    return Ric - (1/2)*R*g

class Spacetime:
    def __init__(self, _metric):
        s, t, x, y, z = sympy.symbols('s t x y z')
        self.metric = _metric
        self.basis = _metric.basis
        self.christoffel_symbols = get_chrisoffel_symbols_from_metric(_metric)
        '''
        To-do:
            - Test Ricci and Riemann and friends.
            - Implement a simulation of Gödel's.
        '''