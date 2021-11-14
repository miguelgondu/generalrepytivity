import sympy
from .utils import (
    _dict_completer_for_tensor,
    is_multiindex,
    get_all_multiindices,
    _get_matrix_of_basis_change,
    _get_preimage,
)


class Tensor:
    """
    This class represents a tensor object in some given coordinates.

    To construct a (p,q)-Tensor, one must pass three arguments:
    1. coordinates (or basis): a list of sympy symbols which represent the coordinates (or
       basis of tangent space).
    2. _type: a pair of values p (the contravariant dimension) and q (the
      covariant dimension).
    3. the non-zero values, which is a dict whose keys are pairs of the
    form (a, b) where a and b are multi-indices such that $\\Gamma^a_b = value$,
    the values that don't appear in this dict are assumed to be 0.

    For example:
    import generalrepytivity as gr
    import sympy

    t, x, y, z = sympy.symbols('t x y z')
    coordinates = [t, x, y, z]
    _type = (2, 1)
    values = {
        ((1,1), (0, )): 5,
        ((0,1), (0, )): -3,
        ((1,0), (2, )): t**2,
    }
    tensor = gr.Tensor(coordinates, _type, dict_of_values)
    """

    def __init__(self, coordinates, _type, values):
        """
        For some quirk, the name that's mostly used internally is basis (instead
        of coordinates).
        """
        self.coordinates = coordinates
        self.basis = coordinates
        self.ct_dim = _type[0]
        self.c_dim = _type[1]
        self.type = _type
        self.dim = len(self.basis)
        if values == "zero":
            self.values = {
                (
                    tuple(0 for i in range(self.ct_dim)),
                    tuple(0 for i in range(self.c_dim)),
                ): 0
            }
        else:
            self.values = _dict_completer_for_tensor(values, self.type, self.dim)

    def __eq__(self, other):
        if other == 0:
            if set(self.values.values()) == set([0]):
                return True
            else:
                return False
        if not isinstance(other, Tensor):
            return False
        if self.basis == other.basis:
            if self.type == other.type:
                if self.get_all_values() == other.get_all_values():
                    return True
        return False

    def __getitem__(self, pair):
        if self.ct_dim == 0 and self.c_dim > 0:
            if is_multiindex(pair, self.dim, self.c_dim):
                if ((), pair) in self.values:
                    return sympy.simplify(self.values[((), pair)])
                else:
                    return sympy.simplify(0)

        if self.c_dim == 0 and self.ct_dim > 0:
            if is_multiindex(pair, self.dim, self.ct_dim):
                if (pair, ()) in self.values:
                    return sympy.simplify(self.values[(pair, ())])
                else:
                    return sympy.simplify(0)

        if self.c_dim == 1 and self.ct_dim == 1:
            if len(pair) == 2:
                i, j = pair
                if isinstance(i, int) and isinstance(j, int):
                    if ((i,), (j,)) in self.values:
                        return sympy.simplify(self.values[(i,), (j,)])
                    else:
                        return sympy.simplify(0)
            else:
                raise KeyError(
                    "There should be two things in {}, but there are {}".format(
                        pair, len(pair)
                    )
                )

        a, b = pair
        if isinstance(a, int):
            if isinstance(b, int):
                if is_multiindex((a,), self.dim, self.ct_dim) and is_multiindex(
                    (b,), self.dim, self.c_dim
                ):
                    if ((a,), (b,)) in self.values:
                        return sympy.simplify(self.values[((a,), (b,))])
                    else:
                        return sympy.simplify(0)
                else:
                    raise KeyError(
                        "There's a problem with multiindices ({},) and ({},)".format(
                            a, b
                        )
                    )
            if is_multiindex(b, self.dim, self.c_dim):
                if ((a,), b) in self.values:
                    return sympy.simplify(self.values[((a,), b)])
                else:
                    return sympy.simplify(0)
            else:
                raise KeyError("There's a problem with multiindex {}".format(b))
        if is_multiindex(a, self.dim, self.ct_dim):
            if isinstance(b, int):
                if (a, (b,)) in self.values:
                    return sympy.simplify(self.values[(a, (b,))])
                else:
                    return sympy.simplify(0)
            if is_multiindex(b, self.dim, self.c_dim):
                if (a, b) in self.values:
                    return sympy.simplify(self.values[(a, b)])
                else:
                    return sympy.simplify(0)
        raise KeyError(
            "There's something wrong with the pair of multiindices {} and {}".format(
                a, b
            )
        )

    def __repr__(self):
        if set(self.values.values()) == set([0]):
            return "0"
        string = ""
        for key in self.values:
            string += "({})".format(self.values[key])
            a, b = key
            if a != ():
                substring_of_a = ""
                for ind in a:
                    substring_of_a += "{} \\otimes ".format(self.basis[ind])
                substring_of_a = substring_of_a[: -len(" \\otimes ")]
                string += substring_of_a
            if b != ():
                if a != ():
                    string += " \\otimes "
                for ind in b:
                    string += "{}* \\otimes ".format(self.basis[ind])
                string = string[: -len(" \\otimes ")]
            string += " + "
        string = string[:-3]
        return string

    def _repr_latex_(self):
        if set(self.values.values()) == set([0]):
            return "$0$"
        string = "$"
        for key in self.values:
            string += "({})".format(sympy.latex(self.values[key]))
            a, b = key
            if a != ():
                substring_of_a = ""
                for ind in a:
                    substring_of_a += "\\partial/\\partial {} \\otimes ".format(
                        sympy.latex(self.basis[ind])
                    )
                substring_of_a = substring_of_a[: -len(" \\otimes ")]
                string += substring_of_a
            if b != ():
                if a != ():
                    string += " \\otimes "
                for ind in b:
                    string += "d{} \\otimes ".format(self.basis[ind])
                string = string[: -len(" \\otimes ")]
            string += " + "
        string = string[:-3]
        return (string + "$").replace("**", "^")

    def __add__(self, other):
        if other == 0:
            return Tensor(self.basis, self.type, self.values)
        if not isinstance(other, Tensor):
            raise ValueError("Cannot add a tensor with a {}".format(type(other)))

        if other.basis != self.basis or other.type != self.type:
            raise ValueError(
                "Tensors should be of the same type and have the same basis."
            )

        result_dict = self.values.copy()
        for key in other.values:
            if key in result_dict:
                result_dict[key] = result_dict[key] + other.values[key]
            if key not in result_dict:
                result_dict[key] = other.values[key]

        for key in result_dict.copy():
            if result_dict[key] == 0:
                empty = result_dict.pop(key)

        result_basis = self.basis
        result_type = self.type
        return Tensor(result_basis, result_type, result_dict).simplify()

    def __mul__(self, other):
        if self.type == (0, 0):
            if isinstance(other, Tensor):
                if other.basis != self.basis:
                    raise ValueError(
                        "Basis of {} should be the same as other tensor's".format(other)
                    )
                new_dict = other.values.copy()
                for key in other.values:
                    new_dict[key] = new_dict[key] * self.values[((), ())]
                return Tensor(self.basis, other.type, new_dict).simplify()
        if isinstance(other, int) or isinstance(other, float):
            new_dict = self.values.copy()
            for key in self.values:
                new_dict[key] = self.values[key] * other
            return Tensor(self.basis, self.type, new_dict).simplify()

        if isinstance(other, Tensor):
            if other.type != (0, 0):
                raise ValueError(
                    "Can't multiply a tensor with a tensor that isn't (0, 0)"
                )
            if other.basis != self.basis:
                raise ValueError(
                    "The basis of {} should be the same as the other tensor".format(
                        other
                    )
                )

            other_value = other[(), ()]
            new_dict = self.values.copy()
            for key in self.values:
                new_dict[key] = self.values[key] * other_value
            return Tensor(self.basis, self.type, new_dict).simplify()

        try:
            new_dict = self.values.copy()
            for key in self.values:
                new_dict[key] = self.values[key] * other
            return Tensor(self.basis, self.type, new_dict).simplify()
        except:
            raise ValueError("Can't multiply a tensor with {}".format(other))

    __rmul__ = __mul__

    def simplify(self):
        """
        This function simplifies (using sympy.simplify) every value in
        the tensors dict.
        """
        new_dict = {}
        for key, value in self.values.items():
            new_dict[key] = sympy.simplify(value)
        if Tensor == 0:
            return Tensor(self.basis, self.type, "zero")
        return Tensor(self.basis, self.type, new_dict)

    def subs(self, substitutions):
        """
        This function substitutes (using sympy.subs) every value in
        the tensors dict with the list substitutions.
        """
        new_dict = {}
        for key, value in self.values.items():
            new_dict[key] = sympy.simplify(value).subs(substitutions)
        return Tensor(self.basis, self.type, new_dict).simplify()

    def evalf(self):
        """
        This function evaluates to floats (using sympy\'s evalf) every
        value in the tensors dict.
        """
        new_dict = {}
        for key, value in self.values.items():
            new_dict[key] = sympy.simplify(self.values[key]).evalf()
        return Tensor(self.basis, self.type, new_dict)

    def get_all_values(self):
        """
        This function returns a non-sparse values dict of the tensor (i.e.
        it fills all the missing zero values).
        """
        new_dict = {}
        dim = self.dim
        contravariant_multiindices = get_all_multiindices(self.ct_dim, dim)
        covariant_multiindices = get_all_multiindices(self.c_dim, dim)
        for a in contravariant_multiindices:
            for b in covariant_multiindices:
                if (a, b) in self.values:
                    new_dict[a, b] = self.values[a, b]
                else:
                    new_dict[a, b] = 0
        return new_dict

    def change_basis(self, new_basis, basis_change):
        """
        This function returns a new tensor object in the new basis according
        to the transormations stored in the dict basis_change. Note that this
        doens\'t happen inplace.
        """
        L = _get_matrix_of_basis_change(self.basis, new_basis, basis_change, False)
        contravariant_indices = get_all_multiindices(self.ct_dim, self.dim)
        covariant_indices = get_all_multiindices(self.c_dim, self.dim)
        new_tensor = Tensor(new_basis, self.type, "zero")
        for key in self.values:
            c, d = key
            for a in contravariant_indices:
                for b in covariant_indices:
                    m = self.values[key]
                    for i in range(self.ct_dim):
                        m *= L[a[i], c[i]]
                    for j in range(self.c_dim):
                        m *= (L.T)[b[j], d[j]]
                    temp_tensor_values = {(a, b): m}
                    temp_tensor = Tensor(new_basis, self.type, temp_tensor_values)
                    new_tensor += temp_tensor
        return new_tensor

    def change_coordinates(self, new_coordinates, coord_change):
        """
        change_coordinates returns a tensor with new coordinates
        with respect to the dict coord_change.

        Its arguments:
        - new_coordinates: a list of sympy symbols that represent
          the new coordinates
        - coord_change: a python dict whose keys are the former coordinates
          and whose values are their relationship with the new coordinates.

        Returns:
        - a new tensor in the new coordinates.

        For example:
        import generalrepytivity as gr
        import sympy

        x, y = sympy.symbols('x y')
        coordinates_1 = [x, y]
        r, theta = sympy.symbols('r \\theta')
        coordinates_2 = [r, theta]
        values = {
            ((), (0, 0)): 1,
            ((), (1, 1)): 1
        }
        g = gr.Tensor(coordinates_1, (0, 2), values)
        coord_change = {
            x: r*sympy.cos(theta),
            y: r*sympy.sin(theta)
        }
        new_g = g.change_coordinates(coordinates_2, coord_change)
        """
        L = _get_matrix_of_basis_change(self.basis, new_coordinates, coord_change, True)
        contravariant_indices = get_all_multiindices(self.ct_dim, self.dim)
        covariant_indices = get_all_multiindices(self.c_dim, self.dim)
        new_tensor = Tensor(new_coordinates, self.type, "zero")
        for key in self.values:
            c, d = key
            for a in contravariant_indices:
                for b in covariant_indices:
                    m = self.values[key]
                    for i in range(self.ct_dim):
                        m *= L[a[i], c[i]]
                    for j in range(self.c_dim):
                        m *= (L.T)[b[j], d[j]]
                    temp_tensor_values = {(a, b): m}
                    temp_tensor = Tensor(new_coordinates, self.type, temp_tensor_values)
                    new_tensor += temp_tensor
        return new_tensor.simplify()

    @classmethod
    def from_function(cls, basis, _type, func):
        """
        This method allows you to create a tensor from a function.
        The function func must take 2 multiindices and turn them
        into a value.

        For example:

        def func(a, b):
            return 2**a[0] * 3**b[0] * 5**b[1]

        func is a valid function for the creation of a (1,2)-Tensor.
        """
        ct_dim, c_dim = _type
        dim = len(basis)
        ct_indices = get_all_multiindices(ct_dim, dim)
        c_indices = get_all_multiindices(c_dim, dim)
        values = {
            (a, b): func(a, b) for a in ct_indices for b in c_indices if func(a, b) != 0
        }
        return cls(basis, _type, values)


def get_tensor_from_matrix(matrix, basis):
    """
    get_tensor_from_matrix returns a (0,2)-tensor whose values come
    from the given matrix

    Its arguments:
    - matrix: a square sympy matrix.
    - basis: a list of sympy symbols that represent the coordinates of
      the tensor.

    Returns:
    - a (0,2)-tensor T where T[(), (i, j)] == matrix[i,j].
    """
    values = {}
    for i in range(len(matrix.tolist())):
        for j in range(len(matrix.tolist())):
            if matrix[i, j] != 0:
                values[(), (i, j)] = matrix[i, j]
    return Tensor(basis, (0, 2), values).simplify()


def get_matrix_from_tensor(tensor):
    """
    get_matrix_from_tensor returns the matrix representation of a (0,2)
    tensor.

    Its arguments:
    - tensor: a (0,2)-tensor.

    Returns:
    - a sympy matrix A where A[i, j] == tensor[(), (i,j)].
    """
    matrix = sympy.zeros(len(tensor.basis))
    for i in range(len(tensor.basis)):
        for j in range(len(tensor.basis)):
            matrix[i, j] = tensor[(), (i, j)]

    return matrix


def _get_list_of_lines(tensor, symbol):
    list_of_lines = []
    if isinstance(tensor, Tensor):
        for value in set(tensor.values.values()):
            if value != 0:
                list_of_preimages = _get_preimage(tensor.values, value)
                line = "$$"
                for preimage in list_of_preimages:
                    a, b = preimage
                    line += symbol + "^{"
                    for i in a:
                        line += str(i)
                    line += "}_{"
                    for j in b:
                        line += str(j)
                    line += "} = "
                line += sympy.latex(value)
                line += "$$\n"
                list_of_lines.append(line)

        # Last line, the one about the zeros
        if list_of_lines != []:
            line = "$$" + symbol + "^{"
            for k in range(tensor.ct_dim):
                line += "a_{" + str(k) + "}"
            line += "}_{"
            for k in range(tensor.c_dim):
                line += "b_{" + str(k) + "}"
            line += "} = 0 \\mbox{ in any other case" + "}$$\n"
            list_of_lines.append(line)
        if list_of_lines == []:
            line = "$$" + symbol + "^{"
            for k in range(tensor.ct_dim):
                line += "a_{" + str(k) + "}"
            line += "}_{"
            for k in range(tensor.c_dim):
                line += "b_{" + str(k) + "}"
            line += "} = 0 \\mbox{ in every case" + "}$$\n"
            list_of_lines.append(line)
    else:
        list_of_lines.append("$$" + symbol + " = " + str(tensor) + "$$")
    return list_of_lines


def print_in_file(file_name, tensor, symbol, append_flag=False, _format="txt"):
    """
    print_in_file pretty prints a tensor in a file.

    print_in_file takes the following arguments:
    - file_name: a string with the name of the file to be created
    - tensor: the tensor object to be printed
    - symbol: a string, which is to represent the symbol (for example \\Gamma)
    - append_flag: a boolean which states whether to append or overwrite the file.
    - _format: either \'txt\' or \'tex\'.
    """
    if not append_flag:
        try:
            _file = open(file_name, "x")
        except:
            _file = open(file_name, "w")
    if append_flag:
        _file = open(file_name, "a")

    list_of_lines = _get_list_of_lines(tensor, symbol)

    if _format == "txt":
        _file.writelines(list_of_lines)
        _file.close()
    elif _format == "tex":
        complete_list_of_lines = []
        complete_list_of_lines.append("\\documentclass{article" + "}\n")
        complete_list_of_lines.append("\\usepackage[utf8]{inputenc" + "}\n")
        complete_list_of_lines.append("\\usepackage[T1]{fontenc" + "}\n")
        complete_list_of_lines.append("\\usepackage[english]{babel" + "}\n")
        complete_list_of_lines.append("\\usepackage{amsmath" + "}\n")
        complete_list_of_lines.append("\\usepackage{amssymb" + "}\n")
        complete_list_of_lines.append("\n")
        complete_list_of_lines.append("\\begin{document" + "}\n")
        complete_list_of_lines.append("\n")
        complete_list_of_lines += list_of_lines
        complete_list_of_lines.append("\n")
        complete_list_of_lines.append("\\end{document" + "}\n")
        _file.writelines(complete_list_of_lines)
        _file.close()
    else:
        raise ValueError("Expected txt or tex for format, but got {}".format(_format))
