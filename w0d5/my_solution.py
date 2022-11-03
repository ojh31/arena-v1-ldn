#%%
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, Union
from einops import repeat
import utils
import numpy as np

Arr = np.ndarray
grad_tracking_enabled = True

#%%
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backwards function for f(x) = log(x)

    grad_out: gradient of some loss wrt out
    out: the output of np.log(x)
    x: the input of np.log

    Return: gradient of the given loss wrt x
    '''
    return grad_out / x

import importlib
importlib.reload(utils)
utils.test_log_back(log_back)
# %%
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    '''Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''

    sum_keep_dims = []
    sum_drop_dims = []
    for i in range(1, len(broadcasted.shape) + 1):
        # print(f'i={i}')
        if i > len(original.shape):
            # print(f'summing axis {-i} with keepdims=False')
            sum_drop_dims.append(-i)
        elif original.shape[-i] == broadcasted.shape[-i]:
            # print(f'Skipping dim {-i}')
            continue
        else:
            # print(f'summing axis {-i} with keepdims=True')
            sum_keep_dims.append(-i)
    unbroadcasted = broadcasted.sum(axis=tuple(sum_keep_dims), keepdims=True)
    unbroadcasted = unbroadcasted.sum(axis=tuple(sum_drop_dims), keepdims=False)
    # print(f'broadcasted.shape={broadcasted.shape}, original.shape={original.shape}, unbroadcasted.shape={unbroadcasted.shape}')
    return unbroadcasted
            


utils.test_unbroadcast(unbroadcast)
# %%
def unbroadcast_vec(broadcasted: Arr, original: Arr) -> Arr:
    '''Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''
    orig_shape = np.array(original.shape )
    broad_shape = np.array(broadcasted.shape[-len(original.shape):])
    sum_keep_dims = np.where(orig_shape != broad_shape)[0] + len(broadcasted.shape) - len(original.shape)
    sum_drop_dims = [-i for i in range(1, len(broadcasted.shape) + 1) if i > len(original.shape)]
    unbroadcasted = broadcasted.sum(axis=tuple(sum_keep_dims), keepdims=True)
    unbroadcasted = unbroadcasted.sum(axis=tuple(sum_drop_dims), keepdims=False)
    # print(f'shape_diff = {original.shape != broadcasted.shape[-len(original.shape):]}')
    # print(f'sum_keep_dims={sum_keep_dims}, sum_drop_dims={sum_drop_dims}')
    # print(f'broadcasted.shape={broadcasted.shape}, original.shape={original.shape}, unbroadcasted.shape={unbroadcasted.shape}')
    return unbroadcasted
            


utils.test_unbroadcast(unbroadcast_vec)
# %%
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    '''Backwards function for x * y wrt argument 0 aka x.'''
    return unbroadcast(grad_out * y, x)

def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    '''Backwards function for x * y wrt argument 1 aka y.'''
    return unbroadcast(grad_out * x, y)

utils.test_multiply_back(multiply_back0, multiply_back1)
utils.test_multiply_back_float(multiply_back0, multiply_back1)
# %%
def forward_and_back(a: Arr, b: Arr, c: Arr) -> tuple[Arr, Arr, Arr]:
    '''
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    '''
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)

    final_grad_out = np.array([1.0])

    # Your code here
    dg_df = final_grad_out / f
    dg_dd = multiply_back0(dg_df, f, d, e)
    dg_de = multiply_back1(dg_df, f, d, e)
    dg_da = multiply_back0(dg_dd, d, a, b)
    dg_db = multiply_back1(dg_dd, d, a, b)
    dg_dc = dg_de / c
    return dg_da, dg_db, dg_dc

importlib.reload(utils)
utils.test_forward_and_back(forward_and_back)

#### Autograd
# %%
@dataclass(frozen=True)
class Recipe:
    '''Extra information necessary to run backpropagation. You don't need to modify this.'''

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."
    args: tuple
    "The input arguments passed to func."
    kwargs: dict[str, Any]
    "Keyword arguments passed to func. To keep things simple today, we aren't going to backpropagate with respect to these."
    parents: dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."
# %%
class BackwardFuncLookup:
    def __init__(self) -> None:
        self.lookup_dict = {}

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        self.lookup_dict[forward_fn, arg_position] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self.lookup_dict[forward_fn, arg_position]

utils.test_back_func_lookup(BackwardFuncLookup)

BACK_FUNCS = BackwardFuncLookup()
BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)
# %%
Arr = np.ndarray
from typing import Optional, Union

class Tensor:
    '''
    A drop-in replacement for torch.Tensor supporting a subset of features.
    '''

    array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    grad: Optional["Tensor"]
    "Backpropagation will accumulate gradients into this field."
    recipe: Optional[Recipe]
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Union[Arr, list], requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __eq__(self, other):
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    @property
    def T(self) -> "Tensor":
        return permute(self)

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def expand(self, new_shape):
        return expand(self, new_shape)

    def permute(self, dims):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        '''Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html'''
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
        return bool(self.item())

def empty(*shape: int) -> Tensor:
    '''Like torch.empty.'''
    return Tensor(np.empty(shape))

def zeros(*shape: int) -> Tensor:
    '''Like torch.zeros.'''
    return Tensor(np.zeros(shape))

def arange(start: int, end: int, step=1) -> Tensor:
    '''Like torch.arange(start, end).'''
    return Tensor(np.arange(start, end, step=step))

def tensor(array: Arr, requires_grad=False) -> Tensor:
    '''Like torch.tensor.'''
    return Tensor(array, requires_grad=requires_grad)

#%%
grad_tracking_enabled = True
def log_forward(x: Tensor) -> Tensor:
    recipe = Recipe(np.log, (x.array,), {}, {0: x})
    requires_grad = x.requires_grad and grad_tracking_enabled
    y = Tensor(
        np.log(x.array), requires_grad
    )
    if requires_grad:
        y.recipe = recipe
    return y

log = log_forward
utils.test_log(Tensor, log_forward)
utils.test_log_no_grad(Tensor, log_forward)
a = Tensor([1], requires_grad=True)
grad_tracking_enabled = False
b = log_forward(a)
grad_tracking_enabled = True
assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
assert b.recipe is None, "should not create recipe if grad tracking globally disabled"
# %%
def multiply_forward(a: Union[Tensor, int], b: Union[Tensor, int]) -> Tensor:
    a_requires_grad = isinstance(a, Tensor) and a.requires_grad
    b_requires_grad = isinstance(b, Tensor) and b.requires_grad
    requires_grad = grad_tracking_enabled and (a_requires_grad or b_requires_grad)
    a_array = a.array if isinstance(a, Tensor) else a
    b_array = b.array if isinstance(b, Tensor) else b
    parents = {}
    if isinstance(a, Tensor):
        parents[0] = a
    if isinstance(b, Tensor):
        parents[1] = b
    recipe = Recipe(np.multiply, (a_array, b_array), {}, parents)
    y = Tensor(a_array * b_array, requires_grad)
    if requires_grad:
        y.recipe = recipe
    return y

multiply = multiply_forward
utils.test_multiply(Tensor, multiply_forward)
utils.test_multiply_no_grad(Tensor, multiply_forward)
utils.test_multiply_float(Tensor, multiply_forward)
a = Tensor([2], requires_grad=True)
b = Tensor([3], requires_grad=True)
grad_tracking_enabled = False
b = multiply_forward(a, b)
grad_tracking_enabled = True
assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
assert b.recipe is None, "should not create recipe if grad tracking globally disabled"
# %%
def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    '''
    numpy_func: function. It takes any number of positional arguments, some of which may be NumPy arrays, and any number of keyword arguments which we aren't allowing to be NumPy arrays at present. It returns a single NumPy array.
    is_differentiable: if True, numpy_func is differentiable with respect to some input argument, so we may need to track information in a Recipe. If False, we definitely don't need to track information.

    Return: function. It has the same signature as numpy_func, except wherever there was a NumPy array, this has a Tensor instead.
    '''
   
    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        requires_grad = is_differentiable and any([
            isinstance(a, Tensor) and a.requires_grad for a in args
        ])
        parents = {i: a for i, a in enumerate(args) if isinstance(a, Tensor)}
        array_args = tuple([a.array if isinstance(a, Tensor) else a for a in args])
        recipe = Recipe(numpy_func, array_args, {}, parents)
        y = Tensor(numpy_func(*array_args, **kwargs), requires_grad)
        if requires_grad:
            y.recipe = recipe
        return y

    return tensor_func


def test_sum(wrap_forward_fn, Tensor):
    # This tests keyword arguments
    def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
        """Like torch.sum, calling np.sum internally."""
        return np.sum(x, axis=dim, keepdims=keepdim)
    global sum
    sum = wrap_forward_fn(_sum)
    a = Tensor(np.array([[0.0, 1.0], [2.0, 3.0]]), requires_grad=True)
    assert a.sum(0).shape == (2,)
    assert a.sum(0, True).shape == (1, 2)
    print("All tests in `test_sum` passed!")

importlib.reload(utils)
log = wrap_forward_fn(np.log)
multiply = wrap_forward_fn(np.multiply)
utils.test_log(Tensor, log)
utils.test_log_no_grad(Tensor, log)
utils.test_multiply(Tensor, multiply)
utils.test_multiply_no_grad(Tensor, multiply)
utils.test_multiply_float(Tensor, multiply)
test_sum(wrap_forward_fn, Tensor)
try:
    log(x=Tensor([100]))
except Exception as e:
    print("Got a nice exception as intended:")
    print(e)
else:
    assert False, "Passing tensor by keyword should raise some informative exception."
# %%
class Node:
    def __init__(self, *children):
        self.children = list(children)
        self.temporary_mark = False
        self.permanent_mark = False

def get_children(node: Node) -> list[Node]:
    return node.children

def topological_sort(node: Node, get_children_fn: Callable) -> list[Any]:
    '''
    Return a list of node's descendants in reverse topological order from future to past.

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    '''
    if node.temporary_mark:
        raise Exception("Seems like your graph is cyclic")
    if node.permanent_mark:
        return []
    node.temporary_mark = True
    nested_descendants = [topological_sort(node, get_children_fn) for node in get_children_fn(node)]
    descendants = [descendant for nest in nested_descendants for descendant in nest]
    print(nested_descendants)
    print(descendants)
    node.temporary_mark = False
    node.permanent_mark = True
    descendants.append(node)
    return descendants


def get_children(node: Node) -> list[Node]:
    return node.children

def test_topological_sort_linked_list(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    expected = [z, y, x]
    for e, a in zip(expected, topological_sort(x, get_children)):
        assert e is a
    print("All tests in `test_topological_sort_linked_list` passed!")

def test_topological_sort_branching(topological_sort):
    z = Node()
    y = Node()
    x = Node(y, z)
    w = Node(x)
    name_lookup = {w: "w", x: "x", y: "y", z: "z"}
    out = "".join([name_lookup[n] for n in topological_sort(w, get_children)])
    assert out == "zyxw" or out == "yzxw", f'out={out}'
    print("All tests in `test_topological_sort_branching` passed!")

def test_topological_sort_rejoining(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    w = Node(z, x)
    name_lookup = {w: "w", x: "x", y: "y", z: "z"}
    out = "".join([name_lookup[n] for n in topological_sort(w, get_children)])
    assert out == "zyxw"
    print("All tests in `test_topological_sort_rejoining` passed!")

def test_topological_sort_cyclic(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    z.children = [x]
    try:
        topological_sort(x, get_children)
    except:
        assert True
    else:
        assert False
    print("All tests in `test_topological_sort_cyclic` passed!")


test_topological_sort_linked_list(topological_sort)
test_topological_sort_branching(topological_sort)
test_topological_sort_rejoining(topological_sort)
test_topological_sort_cyclic(topological_sort)
# %%
