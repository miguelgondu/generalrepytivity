import generalrelativity as gr
import sympy

t, x, y, z = sympy.symbols('t x y z')

def test_return_value_tensor():
    dict_of_values = {
        (None, (0,0)): -1,
        (None, (1,1)): 1,
        (None, (2,2)): 1,
        (None, (3,3)): 1
    }
    basis = [t, x, y, z]
    dimensionality = (0, 2)
    metric = gr.Tensor(basis, dimensionality, dict_of_values)
    assert metric[None, (0,0)] == -1 and metric[None, (1,2)] == 0 and metric[2,2] == 1

def test_return_value_tensor2():
    dict_of_values = {
        ((1,1), (0, )): 5,
        ((0,1), (0, )): -3,
        ((1,0), (2, )): 8,
    }
    basis = [t, x, y, z]
    dimensionality = (2, 1)
    tensor = gr.Tensor(basis, dimensionality, dict_of_values)
    assert tensor[(1,1), 0] == 5

def test_dict_completer_for_tensor_1():
    dict_of_values = {
        (1, (0, 1)): 1,
        ((2,), (0, 1)): 2,
        (0, (1, 1)): 3,
        (3, (0, 0)): 4
    }
    basis = [t, x, y, z]
    T = gr.Tensor(basis, (1, 2), dict_of_values)
    assert T[3, (0,0)] == 4 and T[(2, ), (0, 1)] == 2 and T[3, (0,1)] == 0

def test_dict_completer_for_tensor_2():
    dict_of_values = {
        ((0,1), 2): 1,
        ((2,1), (0, )): 2,
    }
    basis = [t, x, y, z]
    T = gr.Tensor(basis, (2, 1), dict_of_values)
    assert T[(0,1), 2] == 1 and T[(2,1), 3] == 0

def test_dict_completer_for_tensor_3():
    dict_of_values = {
        (0,0): -1,
        (1,1): 1,
        (2,2): 1,
        (3,3): 1
    }
    basis = [t, x, y, z]
    T = gr.Tensor(basis, (0, 2), dict_of_values)
    assert T[(0,0)] == -1 and T[(1,2)] == 0

def test_return_value_one_multiindex():
    dict_of_values = {
        ((0,0,0), None): 1,
        ((0,0,1), None): 2,
        ((0,1,0), None): 3
    }
    basis = [t, x, y, z]
    T = gr.Tensor(basis, (3, 0), dict_of_values)
    assert T[0,0,1] == 2

def test_return_value_one_one_tensor():
    dict_of_values = {
        ((1, ), (0, )): 1,
        ((0, ), (1, )): 2,
        ((1, ), (1, )): 3
    }
    basis = [t, x, y, z]
    T = gr.Tensor(basis, (1,1), dict_of_values)
    assert T[1, 0] == 1 and T[2, 2] == 0

def test_error_with_wrong_key():
    dict_of_values = {
        ((1,1), (0, )): 5,
        ((0,1), (0, )): -3,
        ((1,0), (2, )): 8,
    }
    basis = [t, x, y, z]
    dimensionality = (2, 1)
    tensor = gr.Tensor(basis, dimensionality, dict_of_values)
    try:
        assert tensor[(1,1), (1,1)] == 5
    except KeyError:
        assert True

def test_different_tensors():
    dict_of_values_1 = {
        ((1,1), (0, )): 5,
        ((0,1), (0, )): -3,
        ((1,0), (2, )): 8,
    }
    
    dict_of_values_2 = {
        ((1,1), (0, )): -5,
        ((0,1), (0, )): -3,
        ((1,0), (2, )): 8,
    }
    
    basis = [t, x, y, z]
    _type = (2, 1)
    tensor_1 = gr.Tensor(basis, _type, dict_of_values_1)
    tensor_2 = gr.Tensor(basis, _type, dict_of_values_2)

    assert tensor_1 != tensor_2 

def test_wrong_dict_in_creation():
    _type = (2, 2)
    basis = [t, x, y, z]
    dict_of_values = {
        ((0, 0), (1, 1)): 3,
        ((0, ), (1, 2)): -1
    }
    try:
        tensor = gr.Tensor(basis, _type, dict_of_values)
    except ValueError:
        assert True

def test_wrong_dict_in_creation2():
    _type = (2, 2)
    basis = [t, x, y, z]
    dict_of_values = {
        ((0, 0), (1, 1)): 3,
        ((0, 0), (1, 4)): -1
    }
    try:
        tensor = gr.Tensor(basis, _type, dict_of_values)
    except ValueError:
        assert True

def test_index_contraction1():
    _type = (2, 2)
    indices = gr.get_all_multiindices(2, 4)
    dict_of_values = {(a, b): sum(a)/(sum(b) + 1) for a in indices for b in indices}
    basis = [t, x, y, z]

    tensor = gr.Tensor(basis, _type, dict_of_values)
    new_tensor = gr.contract_indices(tensor, 1, 0)

    other_indices = gr.get_all_multiindices(1, 4)
    other_dict_of_values = {(a, b): sum([(sum(a) + r)/(sum(b) + r + 1) for r in range(4)]) for a in other_indices for b in other_indices}
    other_tensor = gr.Tensor(basis, (1,1), other_dict_of_values)
    assert other_tensor == new_tensor

def test_index_contraction2():
    _type = (2, 1)
    basis = [t, x, y, z]
    dim = len(basis)
    c_indices = gr.get_all_multiindices(2, 4)
    ct_indices = gr.get_all_multiindices(1, 4)
    dict_of_values = {(a, b): 2 ** a[0] * 3 ** a[1] * 5**b[0] for a in c_indices for b in ct_indices}
    tensor = gr.Tensor(basis, _type, dict_of_values)
    contracted_tensor_1 = gr.contract_indices(tensor, 1, 0)

    dict_of_values2 = {
        ((a, ), None): sum([2**a * 3**r * 5**r for r in range(dim)]) for a in range(dim)
    }
    contracted_tensor_2 = gr.Tensor(basis, (1, 0), dict_of_values2)
    assert contracted_tensor_1 == contracted_tensor_2

def test_lower_index1():
    _type = (2, 1)
    basis = [t, x, y, z]
    dim = len(basis)
    c_indices = gr.get_all_multiindices(2, 4)
    ct_indices = gr.get_all_multiindices(1, 4)
    dict_of_values = {(a, b): 2 ** a[0] * 3 ** a[1] * 5**b[0] for a in c_indices for b in ct_indices}
    tensor = gr.Tensor(basis, _type, dict_of_values)
    metric = gr.Metric(sympy.diag(-1, 1, 1, 1), basis)
    lowered_tensor_1 = gr.lower_index(tensor, metric, 1)

    _type2 = (1, 2)
    c_indices_2 = gr.get_all_multiindices(1, 4)
    ct_indices_2 = gr.get_all_multiindices(2, 4)
    dict_of_values_2 = {
        (a, b): sum([metric[None, (r, b[0])]*tensor[(a[0], r), b[1]] for r in range(dim)]) for a in c_indices_2 for b in ct_indices_2
    }
    lowered_tensor_2 = gr.Tensor(basis, _type2, dict_of_values_2)
    assert lowered_tensor_1 == lowered_tensor_2

def test_raise_index1():
    _type = (2, 1)
    basis = [t, x, y, z]
    dim = len(basis)
    c_indices = gr.get_all_multiindices(2, 4)
    ct_indices = gr.get_all_multiindices(1, 4)
    dict_of_values = {(a, b): 2 ** a[0] * 3 ** a[1] * 5**b[0] for a in c_indices for b in ct_indices}
    tensor = gr.Tensor(basis, _type, dict_of_values)
    metric = gr.Metric(sympy.diag(-1, 1, 1, 1), basis)
    raised_tensor_1 = gr.raise_index(tensor, metric, 0)
    inv_metric = metric.matrix.inv()

    _type2 = (3, 0)
    c_indices_2 = gr.get_all_multiindices(3, 4)
    ct_indices_2 = gr.get_all_multiindices(0, 4)
    dict_of_values_2 = {
        (a, b): sum([inv_metric[r, a[2]]*tensor[(a[0], a[1]), r] for r in range(dim)]) for a in c_indices_2 for b in ct_indices_2
    }
    raised_tensor_2 = gr.Tensor(basis, _type2, dict_of_values_2)
    assert raised_tensor_1 == raised_tensor_2

def test_tensor_from_matrix1():
    A = sympy.diag(-1, 1, 1, 1)
    basis = [t, x, y ,z]
    tensor = gr.get_tensor_from_matrix(A, basis)
    assert tensor[None, (0,0)] == -1

def test_subs_in_tensor1():
    _type = (1, 1)
    s = sympy.Symbol('s')
    dict_of_values = {
        ((1, ), (0, )): s**2,
        ((1, ), (1, )): s**3,
    }
    basis = [t, x, y, z]
    tensor_1 = gr.Tensor(basis, _type, dict_of_values)
    dict_of_values2 = {
        ((1, ), (0, )): 9,
        ((1, ), (1, )): 27
    }
    tensor_2 = gr.Tensor(basis, _type, dict_of_values2)
    assert tensor_2 == tensor_1.subs([(s, 3)])

def test_christoffel_symbols1_godel():
    x0, x1, x2, x3 = sympy.symbols('x_0 x_1 x_2 x_3')
    basis = [x0, x1, x2, x3]
    e = sympy.exp(1)
    matrix = sympy.Matrix([[1, 0, e ** x1, 0],
                           [0, -1, 0, 0],
                           [e ** x1, 0, (e**(2*x1)) / 2, 0],
                           [0, 0, 0, -1]])
    metric = gr.Metric(matrix, basis)
    christoffel_symbols_1 = gr.get_chrisoffel_symbols_from_metric(metric)
    christoffel_symbols_2 = {
        ((0, ), (0, 1)): 1,
        ((0, ), (1, 2)): (e ** x1) / 2,
        ((1, ), (0, 2)): (e ** x1) / 2,
        ((1, ), (2, 2)): (e ** (2*x1))/2,
        ((2, ), (0, 1)): -e ** (-x1),
    }
    christoffel_symbols_2 = gr._dict_completer(christoffel_symbols_2, 1, 2, 4)
    assert christoffel_symbols_1.get_all_values() == christoffel_symbols_2
