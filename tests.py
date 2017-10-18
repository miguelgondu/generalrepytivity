import generalrelativity as gr
import sympy

t, x, y, z = sympy.symbols('t x y z')

def test_return_value_tensor():
    dict_of_values = {
        (None,(0,0)): -1,
        (None,(1,1)): 1,
        (None,(2,2)): 1,
        (None,(3,3)): 1,
    }
    basis = [t, x, y, z]
    dimensionality = (0, 2)
    metric = gr.Tensor(basis, dimensionality, dict_of_values)
    assert metric[(None,(0,0))] == -1 and metric[(None, (1,2))] == 0
