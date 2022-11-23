#%%
import torch as t
import torch.nn as nn
from torch import optim as optim
from typing import Tuple, Iterable
import plotly.express as px
import plotly.graph_objects as go
# %%
def loss_fn(weights: t.Tensor, distinguish: Iterable[Tuple[int]]):
    loss = 0
    for (i, j) in distinguish:
        loss += 0.5 * (weights[i, j] - 1) ** 2
    loss += 0.5 * (weights.diff(dim=0) ** 2).sum()
    loss += 0.5 * (weights.diff(dim=1) ** 2).sum()
    return loss

#%%
class Net(nn.Module):

    def __init__(self, grid_size) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            t.zeros((grid_size, grid_size))
        )

    def forward(self):
        return self.weights

#%%
grid_size = 100
net = Net(grid_size=grid_size)
optimizer = optim.SGD(net.parameters(), lr=0.1)
distinguish_points = [(0,0), (70, 70)]
epochs = 1_000
#%%
heatmaps = []
for epoch in range(epochs):
    heatmaps.append(net.weights.detach().clone().numpy())
    loss = loss_fn(net(), distinguish_points)
    loss.backward()
    optimizer.step()

# %%
frames = [
    go.Frame(data=go.Heatmap(z=hm, zmin=0, zmax=1), name=f'frame{i+1}') 
    for i, hm in enumerate(heatmaps)
]
sliders = [dict(
    steps= [
        dict(
            method= 'animate',
            args= [
                [f'frame{k+1}'],
                dict(
                    mode= 'immediate',
                    frame= dict( duration=1, redraw= True ),
                    transition=dict( duration= 1)
                )
            ],
            label='Epoch : {}'.format(k)
        ) 
        for k in range(0, len(frames))
    ], 
    transition= dict(duration= 1),
    x=0,
    y=0,
    currentvalue=dict(font=dict(size=12), visible=True, xanchor= 'center'),
    len=1.0
)]
updatemenus = [
    dict(
        type="buttons", 
        visible=True,
        buttons=[
            {
                "args": [
                    None, 
                    {
                        "frame": {"duration": 1}, 
                        'transition': {'duration': 1}, 
                        'mode': 'immediate'
                    }
                ],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [
                    [None], 
                    {'mode': 'immediate'}
                ],
                "label": "Pause",
                "method": "animate",
            }
        ]
)]
fig = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=dict(
        updatemenus=updatemenus,
        sliders=sliders,
    )
)
fig.show()
# %%
fig.write_html("./colorization.html")
# %%
