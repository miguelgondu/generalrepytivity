import itertools
import sympy


def get_all_multiindices(p, n):
    """
    This function returns a list of all the tuples of the form (a_1, ..., a_p)
    with a_i between 1 and n-1. These tuples serve as multiindices for tensors.
    """
    return list(itertools.product(range(n), repeat=p))


def is_multiindex(multiindex, n, c_dimension):
    """
    This function determines if a tuple is a multiindex or not
    according to these rules:
    1. () is a multiindex of length 0 (i.e. if the covariant or contravariant dimension
    is 0, the empty tuple is the only 0-multiindex)
    2. The length of a multiindex must be equal to the c_dimension
    3. Each value in the multiindex varies between 0 and n-1.
    """
    if isinstance(multiindex, tuple):
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


def _get_matrix_of_basis_change(basis1, basis2, _dict, jacobian=True):
    """
    This is an internal function. It is used in the change_basis method
    for tensor objects. It computes the matrix that represents the
    identity function from (V, basis1) to (V, basis2). It does so
    using derivatives.

    For example, for the variables

    basis1 = [e0, e1, e2, e3]
    basis2 = [f0, f1, f2, f3]
    _dict = {
        e0: f0 + f1,
        e1: f1,
        e2: f1 + f3,
        e3: f2
    }

    the resulting matrix would be
    [[1, 1, 0, 0],
     [0, 1, 0, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0]]

    if the jacobian keyword is set to True, and its transpose if
    it is false. The transpose works when you're trying to change
    basis in the algebraic sense (instead of the geometrical sense).
    """

    dim = len(basis1)
    L = sympy.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            if jacobian == True:
                L[i, j] = _dict[basis1[i]].diff(basis2[j])
            if jacobian == False:
                L[i, j] = _dict[basis1[j]].diff(basis2[i])
    if L.det() == 0:
        raise ValueError("The transformation is not invertible.")
    return L


def _is_valid_key(key, dim, ct_dim, c_dim):
    """
    This is an internal function, it checks whether a given key (i.e. a pair
    of multiindices) is a valid key for certain dimension dim, contravariant dimension
    ct_dim and covariant dimension c_dim. It does so using the is_multiindex function.
    """
    if len(key) != 2:
        return False
    a, b = key
    if not is_multiindex(a, dim, ct_dim):
        return False
    if not is_multiindex(b, dim, c_dim):
        return False

    return True


def _dict_completer_for_tensor(_dict, _type, dim):
    """
    This function checks that the _dict is in proper form and completes in certain cases.
    Those cases are:
        - If one of the dimensions is 0, it is allowed to put only one multiindex instead
          of a pair.
        - if one of the dimensions is 1, it is allowd to put an integer instead of a
          1-multiindex.
    """
    ct_dim = _type[0]
    c_dim = _type[1]

    new_dict = {}

    if _dict == {}:
        new_dict = {
            (tuple(0 for i in range(ct_dim)), tuple(0 for i in range(c_dim))): 0
        }
        return new_dict

    if ct_dim > 0 and c_dim == 0:
        for key in _dict:
            if _is_valid_key(key, dim, ct_dim, c_dim):
                new_dict[key] = _dict[key]
            elif ct_dim == 1 and isinstance(key, int):
                new_dict[((key,), ())] = _dict[key]
            elif is_multiindex(key, dim, ct_dim):
                new_dict[(key, ())] = _dict[key]
            else:
                raise ValueError(
                    "Can't extend key {} because it isn't a {}-multiindex".format(
                        key, ct_dim
                    )
                )
        return new_dict

    if ct_dim == 0 and c_dim > 0:
        for key in _dict:
            if _is_valid_key(key, dim, ct_dim, c_dim):
                new_dict[key] = _dict[key]
            elif c_dim == 1 and isinstance(key, int):
                new_dict[(), (key,)] = _dict[key]
            elif is_multiindex(key, dim, c_dim):
                new_dict[((), key)] = _dict[key]
            else:
                raise ValueError(
                    "Can't extend key {} because it isn't a {}-multiindex".format(
                        key, c_dim
                    )
                )
        return new_dict

    if ct_dim == 1 and c_dim > 0:
        for key in _dict:
            if _is_valid_key(key, dim, ct_dim, c_dim):
                new_dict[key] = _dict[key]
            elif len(key) == 2:
                i, b = key
                if isinstance(i, int) and isinstance(b, int):
                    new_dict[((i,), (b,))] = _dict[key]
                elif isinstance(i, int) and is_multiindex(b, dim, c_dim):
                    new_dict[(i,), b] = _dict[key]
                else:
                    raise ValueError(
                        "{} isn't an integer or {} isn't a {}-multiindex (or int).".format(
                            i, b, c_dim
                        )
                    )
            else:
                raise ValueError("There should only be two things in {}".format(key))
        return new_dict

    if ct_dim > 0 and c_dim == 1:
        for key in _dict:
            if _is_valid_key(key, dim, ct_dim, c_dim):
                new_dict[key] = _dict[key]
            elif len(key) == 2:
                a, j = key
                if isinstance(a, int) and isinstance(j, int):
                    new_dict[(a,), (j,)] = _dict[key]
                elif is_multiindex(a, dim, ct_dim) and isinstance(j, int):
                    new_dict[a, (j,)] = _dict[key]
                else:
                    raise ValueError(
                        "{} should be an integer and {} should be a {}-multiindex (or int in case 1).".format(
                            j, a, ct_dim
                        )
                    )
        return new_dict

    for key in _dict:
        if not _is_valid_key(key, dim, ct_dim, c_dim):
            raise ValueError("Key {} is not compatible with the dimensions")

    return _dict


def _symmetry_completer(_dict):
    new_dict = _dict.copy()
    for a, b in _dict.keys():
        inverted_b = (b[1], b[0])
        if (a, inverted_b) in new_dict:
            if new_dict[a, b] != new_dict[a, inverted_b]:
                raise ValueError(
                    "Inconsistent values for pairs {} and {} of subindices".format(
                        b, inverted_b
                    )
                )
        if (a, inverted_b) not in new_dict:
            new_dict[a, inverted_b] = new_dict[a, b]
    return new_dict


def _dict_completer(_dict, c_dimension, ct_dimension, dim):
    c_indices = get_all_multiindices(c_dimension, dim)
    ct_indices = get_all_multiindices(ct_dimension, dim)
    new_dict = _symmetry_completer(_dict)
    for a in c_indices:
        for b in ct_indices:
            if (a, b) in new_dict:
                pass
            if (a, b) not in new_dict:
                new_dict[a, b] = 0
    return new_dict


def _get_preimage(_dict, value):
    list_of_preimages = [key for (key, _value) in _dict.items() if _value == value]
    return list_of_preimages
