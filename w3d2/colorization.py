#%%
import torch as t
import torch.nn as nn
from torch import optim as optim
from typing import Tuple, Iterable
import pandas as pd
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

    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            t.zeros((grid_size, grid_size))
        )

    def forward(self):
        return self.weights

#%%
grid_size = 10
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001)
distinguish_points = [(0,0), (7, 7)]
epochs = 100
#%%
pd.DataFrame(net.weights.detach().numpy()).style.background_gradient(cmap='Reds').format('{:.02f}')
#%%
heatmaps = []
for epoch in range(epochs):
    print(net.weights.detach()[0, 0])
    heatmaps.append(net.weights.detach().clone().numpy())
    # weight_df = pd.DataFrame(
    #     weight_matrix.detach()
    # ).unstack().reset_index()
    # weight_df.columns = ['x', 'y', 'z']
    # weight_df['t'] = epoch
    # heatmaps.append(weight_df)
    loss = loss_fn(net(), distinguish_points)
    loss.backward()
    optimizer.step()

#%%
pd.DataFrame(heatmaps[0]).style.background_gradient(cmap='Reds')
#%%
pd.DataFrame(heatmaps[-1]).style.background_gradient(cmap='Reds')
# %%
# epoch_df = pd.concat(heatmaps, ignore_index=True)
# %%
frames = [
    go.Frame(data=go.Heatmap(z=hm), name=f'frame{i+1}') for i, hm in enumerate(heatmaps)
]
fig = go.Figure(
    data=go.Heatmap(z=heatmaps[0]),
    frames=frames
)
sliders = [dict(
    steps= [
        dict(
            method= 'animate',
            args= [
                [f'frame{k+1}'],
                dict(
                    mode= 'immediate',
                    frame= dict( duration=600, redraw= True ),
                    transition=dict( duration= 200)
                )
            ],
            label='Epoch : {}'.format(k)
        ) 
        for k in range(0, len(frames))
    ], 
    transition= dict(duration= 100),
    x=0,
    y=0,
    currentvalue=dict(font=dict(size=12), visible=True, xanchor= 'center'),
    len=1.0
)]
menus = [
    dict(type="buttons", visible=True,
    buttons=[dict(label="Play", method="animate", args=[None])]
)]
fig.update_layout(
    updatemenus=menus,
    sliders=sliders
)
fig.show()
# %%
