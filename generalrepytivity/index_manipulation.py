from .utils import get_all_multiindices
from .tensor import Tensor, get_matrix_from_tensor


def contract_indices(tensor, i, j):
    """
    contract_indices formally contracts the i-th superindex and the jth-subindex
    of a tensor.

    Its arguments:
    - tensor: any (p,q)-tensor, with p >= 1 and q >= 1.
    - i: an integer which represents the position of the superindex to
      be contracted (indexing in 0).
    - j: an integer which represents the position of the subindex to
      be raised (indexing in 0).

    Returns:
    - a (p-1,q-1)-tensor, the result of contracting the original tensors i-th
    superindex and j-th subindex.
    """
    dim = len(tensor.basis)
    c_dim = tensor.c_dim
    ct_dim = tensor.ct_dim
    if c_dim < 1 or ct_dim < 1:
        raise ValueError(
            "One dimension in the type {} is less than one.".format(tensor.type)
        )
    if i < 0 or i >= ct_dim:
        raise ValueError(
            "{} is an invalid index to be contracted".format(i, ct_dim - 1)
        )
    if j < 0 or j >= c_dim:
        raise ValueError("{} is an invalid index to be contracted".format(j, c_dim - 1))

    contravariant_indices = get_all_multiindices(ct_dim - 1, dim)
    covariant_indices = get_all_multiindices(c_dim - 1, dim)
    new_tensor_dict = {}
    for a in contravariant_indices:
        for b in covariant_indices:
            sumand = 0
            for r in range(dim):
                if a != ():
                    a_extended = a[:i] + (r,) + a[i:]
                if a == ():
                    a_extended = (r,)
                if b != ():
                    b_extended = b[:j] + (r,) + b[j:]
                if b == ():
                    b_extended = (r,)
                sumand += tensor[a_extended, b_extended]
            if sumand != 0:
                new_tensor_dict[a, b] = sumand

    return Tensor(tensor.basis, (ct_dim - 1, c_dim - 1), new_tensor_dict).simplify()


def lower_index(tensor, metric, i):
    """
    lower_index lowers the i-th index of a tensor with respect to some metric.

    Its arguments:
    - tensor: any (p,q)-tensor, with p >= 1.
    - metric: a (0,2)-tensor which represents a non-degenerate symmetric
      bilinear function.
    - i: an integer which represents the position of the superindex to
      be lowered (indexing in 0).

    Returns:
    - a (p-1,q+1)-tensor, the result of lowering the original tensors i-th superindex.
    """
    if isinstance(metric, Tensor):
        if metric.basis != tensor.basis:
            raise ValueError("Tensor and Metric should be on the same basis.")
        if metric.type != (0, 2):
            raise ValueError("metric should be a (0,2)-tensor")
    else:
        raise ValueError("metric should be a (0,2)-tensor")

    if tensor.ct_dim == 0:
        raise ValueError("There's no index to be lowered.")

    if i < 0 or i >= tensor.ct_dim:
        raise ValueError(
            "The index to be lowered ({}) must be between 0 and {}".format(
                i, tensor.ct_dim
            )
        )

    basis = tensor.basis
    dim = tensor.dim
    new_ct_dim = tensor.ct_dim - 1
    new_c_dim = tensor.c_dim + 1
    new_type = (new_ct_dim, new_c_dim)
    contravariant_indices = get_all_multiindices(new_ct_dim, dim)
    covariant_indices = get_all_multiindices(new_c_dim, dim)

    new_tensor_dict = {}
    for a in contravariant_indices:
        for b in covariant_indices:
            value = 0
            for r in range(dim):
                if a == ():
                    a_extended = (r,)
                if a != ():
                    a_extended = a[:i] + (r,) + a[i:]
                b_reduced = b[1:]
                value += metric[(), (b[0], r)] * tensor[a_extended, b_reduced]
            if value != 0:
                new_tensor_dict[a, b] = value

    return Tensor(basis, new_type, new_tensor_dict).simplify()


def raise_index(tensor, metric, j):
    """
    raise_index raises the j-th index of a tensor with respect to some metric.

    Its arguments:
    - tensor: any (p,q)-tensor, with q >= 1.
    - metric: a (0,2)-tensor which represents a non-degenerate symmetric
      bilinear function.
    - j: an integer which represents the position of the subindex to
      be raised (indexing in 0).

    Returns:
    - a (p+1,q-1)-tensor, the result of raising the original tensors j-th subindex.
    """
    if isinstance(metric, Tensor):
        if metric.basis != tensor.basis:
            raise ValueError("Tensor and Metric should be on the same basis.")
        if metric.type != (0, 2):
            raise ValueError("metric should be an (0,2)-tensor.")
    else:
        raise ValueError("metric should be an (0,2)-tensor.")

    if tensor.c_dim == 0:
        raise ValueError("There's no index to be lowered.")

    if j < 0 or j >= tensor.c_dim:
        raise ValueError(
            "The index to be raised ({}) must be between 0 and {}".format(
                j, tensor.c_dim
            )
        )

    basis = tensor.basis
    dim = len(basis)
    new_ct_dim = tensor.ct_dim + 1
    new_c_dim = tensor.c_dim - 1
    new_type = (new_ct_dim, new_c_dim)
    contravariant_indices = get_all_multiindices(new_ct_dim, dim)
    covariant_indices = get_all_multiindices(new_c_dim, dim)
    inverse_metric_matrix = get_matrix_from_tensor(metric).inv()

    new_tensor_dict = {}
    for a in contravariant_indices:
        for b in covariant_indices:
            value = 0
            for r in range(dim):
                # Here, a[-1] is b_j.
                a_reduced = a[:-1]
                if b == ():
                    b_expanded = (r,)
                if b != ():
                    b_expanded = b[:j] + (r,) + b[j:]
                value += inverse_metric_matrix[a[-1], r] * tensor[a_reduced, b_expanded]
            if value != 0:
                new_tensor_dict[a, b] = value

    # if new_tensor_dict == {}:
    #     new_tensor_dict = {(a, b): 0 for a in contravariant_indices for b in covariant_indices}

    return Tensor(basis, new_type, new_tensor_dict).simplify()
