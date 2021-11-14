from .utils import get_all_multiindices
from .tensor import Tensor, get_matrix_from_tensor, _get_list_of_lines
from .index_manipulation import raise_index, contract_indices


def get_chrisoffel_symbols_from_metric(metric):
    """
    get_christoffel_symbols_from_metric computes the christoffel symbols
    of the Levi-Civita connection associated with a given metric.

    Its arguments:
    - metric: a (0,2)-tensor which represents a non-degenerate symmetric
      bilinear function.
    - Ric (optionally): a (0,2)-tensor (expected to be the Ricci tensor).

    Returns:
    - a (0,0)-tensor, holding the scalar curvature.
    """
    basis = metric.basis
    dim = len(basis)
    metric_matrix = get_matrix_from_tensor(metric)
    inverse_metric_matrix = metric_matrix.inv()
    contravariant_indices = get_all_multiindices(1, dim)
    covariant_indices = get_all_multiindices(2, dim)
    values = {}
    for a in contravariant_indices:
        for b in covariant_indices:
            i, j = b
            c = a[0]
            sumand = 0
            for r in range(dim):
                L = (
                    metric_matrix[j, r].diff(basis[i])
                    + metric_matrix[i, r].diff(basis[j])
                    - metric_matrix[i, j].diff(basis[r])
                )
                sumand += inverse_metric_matrix[r, c] * L
            if sumand != 0:
                values[a, b] = (1 / 2) * sumand
    return Tensor(basis, (1, 2), values).simplify()


def get_Riemann_tensor(christoffel_symbols):
    """
    get_Riemann_tensor computes the Riemann tensor from some christoffel symbols.

    Its arguments:
    - christoffel_symbols: a (1,2)-tensor holding what's expected to be the
      christoffel symbols of a certain metric.

    Returns:
    - a (1,3)-tensor, holding the Riemann tensor.
    """
    cs = christoffel_symbols
    basis = christoffel_symbols.basis
    dim = len(christoffel_symbols.basis)
    contravariant_indices = get_all_multiindices(1, dim)
    covariant_indices = get_all_multiindices(3, dim)
    values = {}
    for x in contravariant_indices:
        for y in covariant_indices:
            d = x[0]
            c, a, b = y
            sumand = cs[d, (b, c)].diff(basis[a]) - cs[d, (a, c)].diff(basis[b])
            for u in range(dim):
                sumand += cs[d, (a, u)] * cs[u, (c, b)] - cs[d, (b, u)] * cs[u, (c, a)]
            if sumand != 0:
                values[x, y] = sumand
    return Tensor(cs.basis, (1, 3), values).simplify()


def get_Ricci_tensor(christoffel_symbols, Riem=None):
    """
    get_Ricci_tensor computes the Ricci tensor from some christoffel symbols.

    Its arguments:
    - christoffel_symbols: a (1,2)-tensor holding what's expected to be the christoffel
    symbols of a certain metric \\m
    - Riem (optionally): a (1,3)-tensor (expected to be the Riemman tensor).

    Returns:
    - a (0,2)-tensor, holding the Ricci tensor.
    """
    if Riem == None:
        Riem = get_Riemann_tensor(christoffel_symbols)
    return contract_indices(Riem, 0, 1)


def get_scalar_curvature(christoffel_symbols, metric, Ric=None):
    """
    get_scalar_curvature computes the scalar curvature from some christoffel symbols
    and some metric.

    Its arguments:
    - christoffel_symbols: a (1,2)-tensor holding what's expected to be the christoffel
    symbols of a certain metric.
    - metric: a (0,2)-tensor which represents a non-degenerate symmetric
      bilinear function.
    - Ric (optionally): a (0,2)-tensor (expected to be the Ricci tensor).

    Returns:
    - a (0,0)-tensor, holding the scalar curvature.
    """
    if Ric == None:
        Ric = get_Ricci_tensor(christoffel_symbols)
    Temp = raise_index(Ric, metric, 0)
    return contract_indices(Temp, 0, 0)


def get_Einstein_tensor(christoffel_symbols, metric, Ric=None, R=None):
    """
    get_Einstein_tensor computes the Einstein tensor from some christoffel symbols
    and some metric.

    Its arguments:
    - christoffel_symbols: a (1,2)-tensor holding what's expected to be the christoffel
    symbols of a certain metric.
    - metric: a (0,2)-tensor which represents a non-degenerate symmetric
      bilinear function.
    - Ric (optionally): a (0,2)-tensor (expected to be the Ricci tensor).
    - R (optionally): an (0,0)-tensor (or just a sympy expr), which is the
      scalar curvature.

    Returns:
    - a (0,2)-tensor, holding the Einstein tensor.
    """
    if Ric == None:
        Ric = get_Ricci_tensor(christoffel_symbols)
    if R == None:
        R = get_scalar_curvature(christoffel_symbols, metric)
    g = metric
    return Ric + (-1 / 2) * R * g


class Spacetime:
    """
    Spacetime takes a metric and computes the usual geometric invariants.

    To create a Spacetime object, one must pass a metric (i.e. a (0,2)-tensor).
    """

    def __init__(self, _metric, printing_flag=False):
        self.metric = _metric
        self.basis = _metric.basis
        if printing_flag:
            print("Computing Christoffel Symbols")
        self.christoffel_symbols = get_chrisoffel_symbols_from_metric(_metric)
        if printing_flag:
            print("Computing Riemann tensor")
        self.Riem = get_Riemann_tensor(self.christoffel_symbols)
        if printing_flag:
            print("Computing Ricci tensor")
        self.Ric = get_Ricci_tensor(self.christoffel_symbols, self.Riem)
        if printing_flag:
            print("Computing Scalar Curvature")
        self.R = get_scalar_curvature(self.christoffel_symbols, self.metric, self.Ric)[
            (), ()
        ]
        if printing_flag:
            print("Computing Einstein's tensor")
        self.G = get_Einstein_tensor(
            self.christoffel_symbols, self.metric, self.Ric, self.R
        )

    def print_summary(self, file_name="Spacetime.txt", _format="txt"):
        """
        print_summary pretty prints a summary of the Spacetime object in a file.

        print_in_file takes the following arguments:
        - file_name: a string with the name of the file to be created
        - _format: either \'txt\' or \'tex\'.
        """
        try:
            _file = open(file_name, "x")
        except:
            _file = open(file_name, "w")
        complete_list_of_lines = []

        # Metric
        complete_list_of_lines.append("Metric:\n")
        complete_list_of_lines += _get_list_of_lines(self.metric, "g")
        complete_list_of_lines.append("\n")

        # Christoffel Symbols
        complete_list_of_lines.append("Christoffel Symbols:\n")
        complete_list_of_lines += _get_list_of_lines(self.christoffel_symbols, "\Gamma")
        complete_list_of_lines.append("\n")

        # Riemann Tensor
        complete_list_of_lines.append("Riemman tensor:\n")
        complete_list_of_lines += _get_list_of_lines(
            self.Riem, "\\mbox{" + "Riem" + "}"
        )
        complete_list_of_lines.append("\n")

        # Ricci Tensor
        complete_list_of_lines.append("Ricci tensor:\n")
        complete_list_of_lines += _get_list_of_lines(self.Ric, "\\mbox{" + "Ric" + "}")
        complete_list_of_lines.append("\n")

        # Scalar curvature
        complete_list_of_lines.append("Scalar curvature:\n")
        complete_list_of_lines += _get_list_of_lines(self.R, "\\mbox{" + "R" + "}")
        complete_list_of_lines.append("\n")

        if _format == "txt":
            _file.writelines(complete_list_of_lines)
            _file.close()
        elif _format == "tex":
            final_list_of_lines = []
            final_list_of_lines.append("\\documentclass{article" + "}\n")
            final_list_of_lines.append("\\usepackage[utf8]{inputenc" + "}\n")
            final_list_of_lines.append("\\usepackage[T1]{fontenc" + "}\n")
            final_list_of_lines.append("\\usepackage[english]{babel" + "}\n")
            final_list_of_lines.append("\\usepackage{amsmath" + "}\n")
            final_list_of_lines.append("\\usepackage{amssymb" + "}\n")
            final_list_of_lines.append("\n")
            final_list_of_lines.append("\\begin{document" + "}\n")
            final_list_of_lines.append("\n")
            final_list_of_lines += complete_list_of_lines
            final_list_of_lines.append("\n")
            final_list_of_lines.append("\\end{document" + "}\n")
            _file.writelines(final_list_of_lines)
            _file.close()
        else:
            raise ValueError(
                "Expected txt or tex for format, but got {}".format(_format)
            )
