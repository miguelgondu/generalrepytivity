import generalrelativity as gr
import sympy

t, x, y, z = sympy.symbols('t x y z')

def test_return_value_tensor():
    dict_of_values = {
        ((0,0), None): -1,
        ((1,1), None): 1,
        ((2,2), None): 1,
        ((3,3), None): 1,
    }
    basis = [t, x, y, z]
    dimensionality = (0, 2)
    metric = gr.Tensor(basis, dimensionality, dict_of_values)
    assert metric[(0,0), None] == -1 and metric[(1,2), None] == 0

def test_return_value_tensor2():
    dict_of_values = {
        ((1,1), (0, )): 5,
        ((0,1), (0, )): -3,
        ((1,0), (2, )): 8,
    }
    basis = [t, x, y, z]
    dimensionality = (1, 2)
    tensor = gr.Tensor(basis, dimensionality, dict_of_values)
    assert tensor[(1,1), 0] == 5

def test_tensor_from_matrix1():
    A = sympy.diag(-1, 1, 1, 1)
    basis = [t, x, y ,z]
    tensor = gr.tensor_from_matrix(A, basis)
    assert tensor[(0,0), None] == -1
