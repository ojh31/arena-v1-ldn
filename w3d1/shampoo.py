#%%
import torch as t
from typing import Iterable, List, Tuple, Dict, Callable, Type
from einops import rearrange, repeat
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%%
def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1
#%%

def plot_fn(fn: Callable, x_range=[-2, 2], y_range=[-1, 3], n_points=100, log_scale=True, show_min=False):
    """Plot the specified function over the specified domain.

    If log_scale is True, take the logarithm of the output before plotting.
    """
    x = t.linspace(*x_range, n_points)
    xx = repeat(x, "w -> h w", h=n_points)
    y = t.linspace(*y_range, n_points)
    yy = repeat(y, "h -> h w", w=n_points)

    z = fn(xx, yy)

    max_contour_label = int(z.log().max().item()) + 1
    contour_range = list(range(max_contour_label))

    fig = make_subplots(
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        rows=1, cols=2,
        subplot_titles=["3D plot", "2D log plot"]
    ).update_layout(height=700, width=1600, title_font_size=40).update_annotations(font_size=20)

    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z,
            colorscale="greys",
            showscale=False,
            hovertemplate = '<b>x</b> = %{x:.2f}<br><b>y</b> = %{y:.2f}<br><b>z</b> = %{z:.2f}</b>',
            contours = dict(
                x = dict(show=True, color="grey", start=x_range[0], end=x_range[1], size=0.2),
                y = dict(show=True, color="grey", start=y_range[0], end=y_range[1], size=0.2),
                # z = dict(show=True, color="red", size=0.001)
            )
        ), row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=t.log(z) if log_scale else z,
            customdata=z,
            hovertemplate = '<b>x</b> = %{x:.2f}<br><b>y</b> = %{y:.2f}<br><b>z</b> = %{customdata:.2f}</b>',
            colorscale="greys",
            # colorbar=dict(tickmode="array", tickvals=contour_range, ticktext=[f"{math.exp(i):.0f}" for i in contour_range])
        ),
        row=1, col=2
    )
    fig.update_traces(showscale=False, col=2)
    if show_min:
        fig.add_trace(
            go.Scatter(
                mode="markers", x=[1.0], y=[1.0], marker_symbol="x", marker_line_color="midnightblue", marker_color="lightskyblue",
                marker_line_width=2, marker_size=12, name="Global minimum"
            ),
            row=1, col=2
        )

    return fig
#%%
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
        x = xy[0, 0]
        y = xy[0, 1]
        f = fn(x, y)
        f.backward()
        opt.step()
        opt.zero_grad()
    return out
# %%
class Shampoo:
    params: list

    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        eps: float,
        lr: float, 
    ):
        '''
        https://arxiv.org/pdf/1802.09568.pdf
        '''
        self.params = list(params)
        self.eps = eps
        self.lr = lr
        for i, p in enumerate(self.params):
            assert len(p.shape) == 2
        self.left = [
            t.eye(p.shape[0], device=p.device, dtype=p.dtype) * eps 
            for p in self.params
        ]
        self.right = [
            t.eye(p.shape[1], device=p.device, dtype=p.dtype) * eps 
            for p in self.params
        ]
        self.steps = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None # t.zeros_like(param)

    def step(self) -> None:
        self.steps += 1
        for i in range(len(self.params)):
            with t.inference_mode():
                grad = self.params[i].grad
                if self.steps == 1:
                    print(
                        'pre update',
                        self.params[i], self.params[i].grad, 
                        self.left[i], self.right[i],
                        grad @ grad.T, grad.T @ grad,
                        t.pow(self.left[i], -.25), 
                        t.pow(self.right[i], -.25),
                        (
                            self.lr * 
                            t.pow(self.left[i], -.25) @ 
                            grad @ 
                            t.pow(self.right[i], -.25)
                        )
                    )
                self.left[i] += grad @ grad.T 
                self.right[i] += grad.T @ grad
                self.params[i] -= (
                    self.lr * 
                    t.pow(self.left[i], -.25) @ 
                    grad @ 
                    t.pow(self.right[i], -.25)
                )
                if self.steps == 1:
                    print(
                        'post update',
                        self.params[i], self.params[i].grad, 
                        self.left[i], self.right[i],
                        grad @ grad.T, grad.T @ grad,
                        t.pow(self.left[i], -.25), 
                        t.pow(self.right[i], -.25),
                        (
                            self.lr * 
                            t.pow(self.left[i], -.25) @ 
                            grad @ 
                            t.pow(self.right[i], -.25)
                        )
                    )

    def __repr__(self) -> str:
        return f"Shampoo(lr={self.lr}, eps={self.eps}, params={self.params}"
#%%
def format_name(name):
    return name.replace("(", "<br>   ").replace(")", "").replace(", ", "<br>   ")

#%%
def plot_optimization(
    opt_fn: Callable, fn: Callable, xy: t.Tensor, optimizers: list, 
    x_range=[-2, 2], y_range=[-1, 3], n_iters: int = 100, log_scale: bool = True, 
    n_points: int = 100, show_min=False,
):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    for i, (color, optimizer) in enumerate(zip(px.colors.qualitative.Set1, optimizers)):
        xys = opt_fn(fn, xy.clone().detach().requires_grad_(True), *optimizer, n_iters).numpy()
        x, y = xys.T
        z = fn(x, y)
        optimizer_active = optimizer[0]([xy.clone().detach().requires_grad_(True)], **optimizer[1])
        name = format_name(str(optimizer_active))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color=color), line=dict(width=1, color=color), name=name), row=1, col=2)

    fig.data = fig.data[::-1]

    return fig
# %%
xy = t.tensor([[-1.5, 2.5]], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]
optimizers = [
    # (SGD, dict(lr=1e-3, momentum=0.98)),
    # (SGD, dict(lr=5e-4, momentum=0.98)),
    (Shampoo, dict(lr=.001, eps=1e-8)),
]

fig = plot_optimization(opt_fn, rosenbrocks_banana, xy, optimizers, x_range, y_range, n_iters=5_000)

fig.show()
# %%
