#%%
import torch as t
import utils
from typing import Callable, Iterable, Type, Dict
import importlib

#%%
def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1
#%%
x_range = [-2, 2]
y_range = [-1, 3]
fig = utils.plot_fn(rosenbrocks_banana, x_range, y_range, log_scale=True)
# %%
def opt_fn_with_sgd(
    fn: Callable, xy: t.Tensor, lr: float = 0.001, momentum: float = 0.98, n_iters: int = 100
):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    assert xy.requires_grad
    out = t.zeros((n_iters, 2))
    velocity = t.zeros(2)
    
    for i in range(n_iters):
        out[i, :] = xy.detach()
        f = fn(*xy)
        f.backward()
        velocity = momentum * velocity + (1 - momentum) * xy.grad
        with t.inference_mode():
            xy -= lr * velocity
        xy.grad = None
    return out

# %%
xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]

fig = utils.plot_optimization_sgd(opt_fn_with_sgd, rosenbrocks_banana, xy, x_range, y_range, lr=0.01, momentum=0.98, show_min=True, n_iters=1_000)

fig.show()
# %%
class SGD:
    params: list

    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        lr: float, 
        momentum: float, 
        weight_decay: float = 0,
    ):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        '''
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = t.zeros_like(param)

    def step(self) -> None:
        for i in range(len(self.params)):
            with t.inference_mode():
                grad = self.params[i].grad + self.params[i] * self.weight_decay
                self.velocity[i] = (
                    self.momentum * self.velocity[i] + 
                    grad
                )
                self.params[i] -= self.lr * self.velocity[i]

    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        return f"SGD(lr={self.lr}, mom={self.momentum}, wd={self.weight_decay}, params={self.params}"

importlib.reload(utils)
utils.test_sgd(SGD)
# %%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        '''
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.squares = [t.zeros_like(p) for p in self.params]
        self.buffer = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = t.zeros_like(param)

    def step(self) -> None:
        for i in range(len(self.params)):
            with t.inference_mode():
                grad = self.params[i].grad + self.params[i] * self.weight_decay
                self.squares[i] = (
                    self.alpha * self.squares[i] + 
                    (1 - self.alpha) * grad ** 2
                )
                self.buffer[i] = (
                    self.buffer[i] * self.momentum + grad / (t.sqrt(self.squares[i]) + self.eps)
                )
                self.params[i] -= self.lr * self.buffer[i]

    def __repr__(self) -> str:
        return (
            f"class RMSprop:(lr={self.lr}, alpha={self.alpha}, eps={self.eps}, "
            f"mom={self.momentum}, wd={self.weight_decay}, params={self.params}"
        )



utils.test_rmsprop(RMSprop)
# %%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float = 0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        '''
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.velocity = [t.zeros_like(p) for p in self.params]
        self.squares = [t.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = t.zeros_like(param)

    def step(self) -> None:
        self.t += 1
        for i in range(len(self.params)):
            with t.inference_mode():
                grad = self.params[i].grad + self.params[i] * self.weight_decay
                self.velocity[i] = (
                    self.betas[0] * self.velocity[i] + (1- self.betas[0]) * grad
                )
                self.squares[i] = (
                    self.betas[1] * self.squares[i] + 
                    (1 - self.betas[1]) * grad ** 2
                )
                v_hat = self.velocity[i] / (1 - self.betas[0] ** self.t)
                s_hat = self.squares[i] / (1 - self.betas[1] ** self.t)
                self.params[i] -= self.lr * v_hat / (
                    t.sqrt(s_hat) + self.eps
                )

    def __repr__(self) -> str:
        return (
            f"class Adam:(lr={self.lr}, betas={self.betas}, eps={self.eps}, "
            f"wd={self.weight_decay}, params={self.params}"
        )

utils.test_adam(Adam)
# %%
def opt_fn(
    fn: Callable, xy: t.Tensor, optimizer_class: Type, optimizer_kwargs: Dict, 
    n_iters: int = 100
) -> t.Tensor:
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    assert xy.requires_grad
    opt = optimizer_class(params=[xy], **optimizer_kwargs)
    out = t.zeros((n_iters, 2), device=xy.device)
    for i in range(n_iters):
        out[i, :] = xy.detach()
        f = fn(*xy)
        f.backward()
        opt.step()
        opt.zero_grad()
    return out


# %%
xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]
optimizers = [
    # (SGD, dict(lr=1e-3, momentum=0.98)),
    # (SGD, dict(lr=5e-4, momentum=0.98)),
    (Adam, dict(lr=.01, betas=[0.9, .999], eps=1e-8)),
]

fig = utils.plot_optimization(opt_fn, rosenbrocks_banana, xy, optimizers, x_range, y_range, n_iters=5_000)

fig.show()
# %% [markdown]
#### Schedulers
# %%
class ExponentialLR():
    def __init__(self, optimizer, gamma):
        '''Implements ExponentialLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
        '''
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self):
        # print(self.optimizer.lr, self.gamma)
        self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f'ExponentialLR: lr={self.optimizer.lr}, gamma={self.gamma}'
        
importlib.reload(utils)
utils.test_ExponentialLR(ExponentialLR, SGD)
# %%
class StepLR():
    def __init__(self, optimizer, step_size, gamma=0.1):
        '''Implements StepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        '''
        self.optimizer = optimizer
        self.gamma = gamma
        self.step_size = step_size
        self.step_idx = 0

    def step(self):
        self.step_idx += 1
        if self.step_idx % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f'StepLR: lr={self.optimizer.lr}, gamma={self.gamma}, step_size={self.step_size}'

utils.test_StepLR(StepLR, SGD)
# %%
class MultiStepLR():
    def __init__(self, optimizer, milestones, gamma=0.1):
        '''Implements MultiStepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
        '''
        self.optimizer = optimizer
        self.gamma = gamma
        self.milestones = milestones
        self.step_idx = 0

    def step(self):
        self.step_idx += 1
        if self.step_idx in self.milestones:
            self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f'MultiStepLR: lr={self.optimizer.lr}, gamma={self.gamma}'

utils.test_MultiStepLR(MultiStepLR, SGD)
# %%
def opt_fn_with_scheduler(
    fn: Callable, 
    xy: t.Tensor, 
    optimizer_class, 
    optimizer_kwargs, 
    scheduler_class = None, 
    scheduler_kwargs = dict(), 
    n_iters: int = 100
):
    '''Optimize the a given function starting from the specified point.

    scheduler_class: one of the schedulers you've defined, either ExponentialLR, StepLR or MultiStepLR
    scheduler_kwargs: keyword arguments passed to your optimiser (e.g. gamma)
    '''
    assert xy.requires_grad
    opt = optimizer_class([xy], **optimizer_kwargs)
    sched = None if scheduler_class is None else scheduler_class(opt, **scheduler_kwargs)
    out = t.zeros((n_iters, 2), device=xy.device)
    for i in range(n_iters):
        out[i, :] = xy.detach()
        f = fn(*xy)
        f.backward()
        opt.step()
        opt.zero_grad()
        if sched is not None:
            sched.step()
    return out

# %%
xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]
optimizers = [
    (SGD, dict(lr=1e-3, momentum=0.98)),
    (SGD, dict(lr=1e-3, momentum=0.98)),
    (Adam, dict(lr=.6, betas=[.9, .999], eps=1e-8)),
]
schedulers = [
    (), # Empty list stands for no scheduler
    (ExponentialLR, dict(gamma=0.99)),
    (ExponentialLR, dict(gamma=0.99)),
]

fig = utils.plot_optimization_with_schedulers(
    opt_fn_with_scheduler, rosenbrocks_banana, xy, optimizers, schedulers, 
    x_range, y_range, show_min=True
)

fig.show()