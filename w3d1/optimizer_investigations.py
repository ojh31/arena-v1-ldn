#%%
import torch as t
import utils
import importlib
importlib.reload(utils)
# %%
class SGD:

    def __init__(self, params, **kwargs):
        '''Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        kwargs can contain lr, momentum or weight_decay
        '''
        self.params = list(params)
        self.kwargs = kwargs
        assert len(self.params) > 0
        if self.has_groups:
            self.params = [
                {k: list(v) if k == 'params' else v for k, v in g.items()}
                for g in self.params
            ]
        if self.has_groups:
            all_params = set()
            for p in self.iter_params():
                assert p not in all_params
                all_params.add(p)

        if self.has_groups:
            self.velocity = [
                [t.zeros_like(p) for p in g['params']]
                for g in self.params
            ]
        else:
            self.velocity = [t.zeros_like(p) for p in self.params]

    @property
    def has_groups(self):
        return isinstance(self.params[0], dict)

    def iter_params(self):
        if self.has_groups:
            return (p for g in self.params for p in g['params'])
        else:
            return self.params

    def zero_grad(self) -> None:
        for p in self.iter_params():
            p.grad = None

    def step(self):
        if self.has_groups:
            for g_idx, g in enumerate(self.params):
                for p_idx, p in enumerate(g['params']):
                    lr = g.get('lr', self.kwargs.get('lr'))
                    momentum = g.get(
                        'momentum', 
                        self.kwargs.get('momentum', 0)
                    )
                    weight_decay = g.get(
                        'weight_decay', 
                        self.kwargs.get('weight_decay', 0)
                    )
                    print(p, momentum, weight_decay, lr)
                    with t.inference_mode():
                        grad = p.grad + p * weight_decay
                        self.velocity[g_idx][p_idx] = (
                            momentum * self.velocity[g_idx][p_idx] + 
                            grad
                        )
                        self.params[g_idx]['params'][p_idx] -= (
                            lr * self.velocity[g_idx][p_idx]
                        )
        else:
            lr = self.kwargs.get('lr')
            momentum = self.kwargs.get('momentum', 0)
            weight_decay = self.kwargs.get('weight_decay', 0)
            for i in range(len(self.params)):
                with t.inference_mode():
                    grad = (
                        self.params[i].grad + 
                        self.params[i] * weight_decay
                    )
                    self.velocity[i] = (
                        momentum * self.velocity[i] + 
                        grad
                    )
                    self.params[i] -= self.lr * self.velocity[i]


utils.test_sgd_param_groups(SGD)

# %%
