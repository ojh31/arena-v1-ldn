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
        recipe = Recipe(numpy_func, array_args, kwargs, parents)
        # print('numpy_func', numpy_func, array_args, kwargs)
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

def get_children(node: Node) -> list[Node]:
    return node.children

def topological_sort(node: Union[Node, Tensor], get_children_fn: Callable) -> list[Any]: 
    temporary_marks = set()
    permanent_marks = set()

    def topological_sort_recursion(node: Union[Node, Tensor], get_children_fn: Callable) -> list[Any]:
        '''
        Return a list of node's descendants in reverse topological order from future to past.

        Should raise an error if the graph with `node` as root is not in fact acyclic.
        '''
        if node in temporary_marks:
            raise Exception("Seems like your graph is cyclic")
        if node in permanent_marks:
            return []
        temporary_marks.add(node)
        nested_descendants = [topological_sort_recursion(node, get_children_fn) for node in get_children_fn(node)]
        descendants = [descendant for nest in nested_descendants for descendant in nest]
        temporary_marks.remove(node)
        permanent_marks.add(node)
        descendants.append(node)
        return descendants

    return topological_sort_recursion(node, get_children_fn)


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
    assert out == "zyxw", f'out={out}, expected zyxw'
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
def get_parents(node: Tensor):
    if not isinstance(node, Tensor) or node.recipe is None:
        return []
    return list(node.recipe.parents.values())

def sorted_computational_graph(node: Tensor) -> list[Tensor]:
    '''
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph, in reverse topological order.
    '''

    return topological_sort(node, get_parents)[::-1]

a = Tensor([1], requires_grad=True)
b = Tensor([2], requires_grad=True)
c = Tensor([3], requires_grad=True)
d = a * b
e = c.log()
f = d * e
g = f.log()
name_lookup = {a: "a", b: "b", c: "c", d: "d", e: "e", f: "f", g: "g"}

print([name_lookup[t] for t in sorted_computational_graph(g)])
# Should get something in reverse alphabetical order (or close)
# %%
def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    '''Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node: 
        The rightmost node in the computation graph. 
        If it contains more than one element, end_grad must be provided.
    end_grad: 
        A tensor of the same shape as end_node. 
        Set to 1 if not specified and end_node has only one element.
    '''

    # Get value of end_grad_arr
    end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array

    # Create dict to store gradients
    grads: dict[Tensor, Arr] = {end_node: end_grad_arr}

    # Iterate through the computational graph, using your sorting function
    for node in sorted_computational_graph(end_node):

        # Get the outgradient (recall we need it in our backward functions)
        outgrad = grads.pop(node)
        # We only store the gradients if this node is a leaf (see the is_leaf property of Tensor)
        if node.is_leaf:
            # Add the gradient to this node's grad (need to deal with special case grad=None)
            if node.grad is None:
                node.grad = Tensor(outgrad)
            else:
                node.grad.array += outgrad

        # If node has no recipe, then it has no parents, i.e. the backtracking through computational
        # graph ends here
        if node.recipe is None:
            continue

        # If node has a recipe, then we iterate through parents (which is a dict of {arg_posn: tensor})
        for argnum, parent in node.recipe.parents.items():

            # Get the backward function corresponding to the function that created this node,
            # and the arg posn of this particular parent within that function 
            back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)

            # Use this backward function to calculate the gradient
            # print('back_fn', back_fn, outgrad.shape, node.array.shape, node.recipe.args, node.recipe.kwargs)
            in_grad = back_fn(outgrad, node.array, *node.recipe.args, **node.recipe.kwargs)

            # Add the gradient to this node in the dictionary `grads`
            # Note that we only change the grad of the node itself in the code block above
            if grads.get(parent) is None:
                grads[parent] = in_grad
            else:
                grads[parent] += in_grad


# %%
def _argmax(x: Arr, dim=None, keepdim=False):
    '''Like torch.argmax.'''
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))

argmax = wrap_forward_fn(_argmax, is_differentiable=False)
eq = wrap_forward_fn(np.equal, is_differentiable=False)

a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
b = a.argmax()
assert not b.requires_grad
assert b.recipe is None
assert b.item() == 3


# %%
def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backward function for f(x) = -x elementwise.'''
    return -1*grad_out*np.ones(x.shape)

negative = wrap_forward_fn(np.negative)
BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

utils.test_negative_back(Tensor)
# %%

def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return out * grad_out

exp = wrap_forward_fn(np.exp)
BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

utils.test_exp_back(Tensor)
# %%
def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return (grad_out*np.ones(new_shape)).reshape(x.shape)

reshape = wrap_forward_fn(np.reshape)
BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

utils.test_reshape_back(Tensor)
# %%
def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    reverse_axes = [axes.index(i) for i in range(len(x.shape))]
    return np.transpose((grad_out*np.ones(grad_out.shape)),reverse_axes)

BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
permute = wrap_forward_fn(np.transpose)

utils.test_permute_back(Tensor)
# %%

def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out*np.ones(grad_out.shape), x)

def _expand(x: Arr, new_shape) -> Arr:
    '''Like torch.expand, calling np.broadcast_to internally.
    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    '''
    true_shape = [shape if shape > 0 else x.shape[i-len(new_shape)+len(x.shape)] for i, shape in enumerate(new_shape)]
    return np.broadcast_to(x, true_shape)

expand = wrap_forward_fn(_expand)
BACK_FUNCS.add_back_func(_expand, 0, expand_back)

utils.test_expand(Tensor)
utils.test_expand_negative_length(Tensor)


# %%

#%%
# def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
#     '''Basic idea: repeat grad_out over the dims along which x was summed'''
#     print('sum_back', grad_out.shape, x.shape)
#     back = np.broadcast_to(grad_out, x.shape)
#     print(f'back.shape={back.shape}')
#     return back

def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    """Basic idea: repeat grad_out over the dims along which x was summed"""
    # print('sum_back', grad_out.shape, x.shape, dim, keepdim)
    # If grad_out is a scalar, we need to make it a tensor (so we can expand it later)
    
    # if not isinstance(grad_out, Arr):
    #     grad_out = Tensor(grad_out)
    
    # If dim=None, this means we summed over all axes, and we want to repeat back to input shape
    if dim is None:
        dim = list(range(x.ndim))
        
    # If keepdim=False, then we need to add back in dims, so grad_out and x have same number of dims
    if keepdim == False:
        grad_out = np.expand_dims(grad_out, dim)
    
    # Finally, we repeat grad_out along the dims over which x was summed
    return np.broadcast_to(grad_out, x.shape)

def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    '''Like torch.sum, calling np.sum internally.'''
    return np.sum(x, axis = dim, keepdims = keepdim)

sum = wrap_forward_fn(_sum)
BACK_FUNCS.add_back_func(_sum, 0, sum_back)

importlib.reload(utils)
utils.test_sum_keepdim_false(Tensor)
utils.test_sum_keepdim_true(Tensor)
utils.test_sum_dim_none(Tensor)



# %%
Index = Union[int, tuple[int, ...], tuple[Arr], tuple[Tensor]]

def coerce_index(index):
    # print(index)
    if isinstance(index, tuple):
        return tuple(i.array if isinstance(i, Tensor) else i for i in index)
    # print(index)
    return index


def _getitem(x: Arr, index: Index) -> Arr:
    '''Like x[index] when x is a torch.Tensor.'''
    return x[coerce_index(index)]


def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    '''Backwards function for _getitem.
    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    '''
    new_grad_out = np.full_like(x, 0)
    np.add.at(new_grad_out, coerce_index(index), grad_out)
    return new_grad_out

getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)

utils.test_getitem_int(Tensor)
utils.test_getitem_tuple(Tensor)
utils.test_getitem_integer_array(Tensor)
utils.test_getitem_integer_tensor(Tensor)
# %%
add = wrap_forward_fn(np.add)
subtract = wrap_forward_fn(np.subtract)
true_divide = wrap_forward_fn(np.true_divide)


# %%
# Your code goes here
BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out,x))
BACK_FUNCS.add_back_func(np.add, 1, lambda grad_out, out, x, y: unbroadcast(grad_out,y))
BACK_FUNCS.add_back_func(np.subtract, 0, lambda grad_out, out, x, y: unbroadcast(grad_out,x))
BACK_FUNCS.add_back_func(np.subtract, 1, lambda grad_out, out, x, y: unbroadcast(-grad_out,y))
BACK_FUNCS.add_back_func(np.true_divide, 0, lambda grad_out, out, x, y: unbroadcast(grad_out/y,x))
BACK_FUNCS.add_back_func(np.true_divide, 1, lambda grad_out, out, x, y: unbroadcast(-x*grad_out/(y**2),y))

utils.test_add_broadcasted(Tensor)
utils.test_subtract_broadcasted(Tensor)
utils.test_truedivide_broadcasted(Tensor)

# %%
def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    '''Like torch.add_. Compute x += other * alpha in-place and return tensor.'''
    np.add(x.array, other.array * alpha, out=x.array)
    return x


def safe_example():
    '''This example should work properly.'''
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    a.add_(b)
    c = a * b
    c.sum().backward()
    assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


def unsafe_example():
    '''This example is expected to compute the wrong gradients.'''
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    c = a * b
    a.add_(b)
    c.sum().backward()
    if a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0]):
        print("Grad wrt a is OK!")
    else:
        print("Grad wrt a is WRONG!")
    if b.grad is not None and np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0]):
        print("Grad wrt b is OK!")
    else:
        print("Grad wrt b is WRONG!")


if __name__=="__main__":
    safe_example()
    unsafe_example()

a = Tensor([0, 1, 2, 3], requires_grad=True)
(a * 2).sum().backward()
b = Tensor([0, 1, 2, 3], requires_grad=True)
(2 * b).sum().backward()
assert a.grad is not None
assert b.grad is not None
assert np.allclose(a.grad.array, b.grad.array)

# %%
# %%
def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt x.'''
    # print(grad_out.shape)
    # print(out.shape)
    # print(x.shape)
    # print(y.shape)
    back = (grad_out*np.select([x>y,x==y],[1,0.5],default = 0))
    back = unbroadcast(back, x)
    # print(back.shape)
    return back

def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt y.'''
    # print(grad_out.shape)
    # print(out.shape)
    # print(x.shape)
    # print(y.shape)
    back = (grad_out*np.select([y > x,x == y],[1, 0.5],default = 0))
    back = unbroadcast(back, y)
    # print(back.shape)
    return back

maximum = wrap_forward_fn(np.maximum)

BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)

importlib.reload(utils)
utils.test_maximum(Tensor)
utils.test_maximum_broadcasted(Tensor)
# %%
def relu(x: Tensor) -> Tensor:
    '''Like torch.nn.function.relu(x, inplace=False).'''
    return maximum(x, 0)

utils.test_relu(Tensor)
# %%
def _matmul2d(x: Arr, y: Arr) -> Arr:
    '''Matrix multiply restricted to the case where both inputs are exactly 2D.'''
    return x @ y

def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return grad_out @ np.transpose(y)

def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return np.transpose(x) @ grad_out

matmul = wrap_forward_fn(_matmul2d)

BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)

utils.test_matmul2d(Tensor)
# %%
def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    '''Like torch.add_. Compute x += other * alpha in-place and return tensor.'''
    np.add(x.array, other.array * alpha, out=x.array)
    return x

#### Part 4: Putting everything together
class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        '''Share the array with the provided tensor.'''
        super().__init__(tensor.array, requires_grad)

    def __repr__(self):
        return f"Parameter containing: {super().__repr__()}"

x = Tensor([1.0, 2.0, 3.0])
p = Parameter(x)
assert p.requires_grad
assert p.array is x.array
assert repr(p) == "Parameter containing: Tensor(array([1., 2., 3.]), requires_grad=True)"
x.add_(Tensor(np.array(2.0)))
assert np.allclose(
    p.array, np.array([3.0, 4.0, 5.0])
), "in-place modifications to the original tensor should affect the parameter"
# %%
class Module:
    _modules: dict[str, "Module"]
    _parameters: dict[str, Parameter]

    def __init__(self):
        self._modules = dict()
        self._parameters = dict()

    def modules(self):
        '''Return the direct child modules of this module.'''
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        '''Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        '''
        l1_params = [p for p in self._parameters.values()]
        subm_params = [p for m in self._modules.values() for p in m.parameters(recurse=True)]
        if recurse:
            return l1_params + subm_params
        else:
            return l1_params


    def __setattr__(self, key: str, val: Any) -> None:
        '''
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call the superclass.
        '''
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Union[Parameter, "Module"]:
        '''
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        '''
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._modules:
            return self._modules[key]
        else:
            raise KeyError(f'key={key}')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        def _indent(s_, numSpaces):
            return re.sub("\n", "\n" + (" " * numSpaces), s_)
        lines = [f"({key}): {_indent(repr(module), 2)}" for key, module in self._modules.items()]
        return "".join([
            self.__class__.__name__ + "(",
            "\n  " + "\n  ".join(lines) + "\n" if lines else "", ")"
        ])


class TestInnerModule(Module):
    def __init__(self):
        super().__init__()
        self.param1 = Parameter(Tensor([1.0]))
        self.param2 = Parameter(Tensor([2.0]))

class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.inner = TestInnerModule()
        self.param3 = Parameter(Tensor([3.0]))

mod = TestModule()
assert list(mod.modules()) == [mod.inner]
assert list(mod.parameters()) == [
    mod.param3,
    mod.inner.param1,
    mod.inner.param2,
], "parameters should come before submodule parameters"
print("Manually verify that the repr looks reasonable:")
print(mod)
# %%
class Linear(Module):
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        sf = 1 / np.sqrt(in_features)
        
        weight = sf * (2 * np.random.rand(in_features, out_features) - 1)
        self.weight = Parameter(Tensor(weight, requires_grad=True))
        
        if bias:
            bias = sf * (2 * np.random.rand(out_features,) - 1)
            self.bias = Parameter(Tensor(bias, requires_grad=True))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        # print('Linear.forward()')
        # print(x.array.shape)
        # print(self.weight.array.shape)
        z = x @ self.weight
        if self.bias is not None: 
            # print(self.bias.array.shape)
            z += self.bias
        # print(z.shape)
        return z

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

#%%
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 64)
        self.linear2 = Linear(64, 64)
        self.output = Linear(64, 10)

    def forward(self, x):
        x = x.reshape((x.shape[0], 28 * 28))
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = self.output(x)
        return x
# %%
def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    '''Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    '''
    row_labels = list(range(len(true_labels.array)))
    idx = (np.array(row_labels), true_labels.array)
    logits_at_label = logits[idx]
    softmax = exp(logits_at_label) / exp(logits).sum(1)
    loss = -log(softmax)
    return loss

utils.test_cross_entropy(Tensor, cross_entropy)
#%%
class NoGrad:
    '''Context manager that disables grad inside the block. Like torch.no_grad.'''

    was_enabled: bool

    def __enter__(self):
        "SOLUTION"
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        "SOLUTION"
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled
# %%
(train_loader, test_loader) = utils.get_mnist(20)
utils.visualize(train_loader)

class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float):
        '''Vanilla SGD with no additional features.'''
        self.params = list(params)
        self.lr = lr
        self.b = [None for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        with NoGrad():
            for (i, p) in enumerate(self.params):
                assert isinstance(p.grad, Tensor)
                p.add_(p.grad, -self.lr)


def train(model, train_loader, optimizer, epoch):
    for (batch_idx, (data, target)) in enumerate(train_loader):
        data = Tensor(data.numpy())
        target = Tensor(target.numpy())
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, target).sum() / len(output)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]	Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader):
    test_loss = 0
    correct = 0
    with NoGrad():
        for (data, target) in test_loader:
            data = Tensor(data.numpy())
            target = Tensor(target.numpy())
            output = model(data)
            test_loss += cross_entropy(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
# %%
num_epochs = 5
model = MLP()
start = time.time()
optimizer = SGD(model.parameters(), 0.01)
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
    optimizer.step()
print(f"Completed in {time.time() - start: .2f}s")
# %%
