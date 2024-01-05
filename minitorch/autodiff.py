from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_plus_epsilon = list(vals)
    vals_minus_epsilon = list(vals)

    vals_plus_epsilon[arg] += epsilon
    vals_minus_epsilon[arg] -= epsilon

    return (f(*vals_plus_epsilon) - f(*vals_minus_epsilon)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    sorted_list = []
    
    def DFS(_variable: Variable) -> None:
        if (_variable.unique_id in visited) or _variable.is_constant():
            return
        if not _variable.is_leaf():
            for parent in _variable.parents:
                DFS(parent)
        sorted_list.insert(0, _variable)
        visited.add(_variable.unique_id)

    DFS(variable)

    return sorted_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    top_sorted = topological_sort(variable)
    derivations = {var.unique_id : 0 for var in top_sorted}
    derivations[variable.unique_id] = deriv
    for x in top_sorted:
        d_out = derivations[x.unique_id]
        if x.is_leaf():
            x.accumulate_derivative(d_out)
        else:
            back = x.chain_rule(d_out)
            for back_var, back_d in back:
                derivations[back_var.unique_id] += back_d




@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
