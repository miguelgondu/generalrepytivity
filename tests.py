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
    assert metric[None, (0,0)] == -1 and metric[None, (1,2)] == 0

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

def test_tensor_from_matrix1():
    A = sympy.diag(-1, 1, 1, 1)
    basis = [t, x, y ,z]
    tensor = gr.tensor_from_matrix(A, basis)
    assert tensor[None, (0,0)] == -1
