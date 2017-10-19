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
