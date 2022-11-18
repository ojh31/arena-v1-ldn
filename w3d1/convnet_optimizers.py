# %%
from einops import rearrange
import torch as t
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm_notebook
import wandb
import time
from typing import Callable

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.max2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


#%%
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
#%%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ================================= ConvNet training & testing =================================
#%%
loss_fn = nn.CrossEntropyLoss()

device = "cuda" if t.cuda.is_available() else "cpu"

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

#%%
def train_convnet(config=None):
    """
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    
    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    """
    wandb.init(config=config)
    config = wandb.config

    # wandb.init(
    #     project='convnet_optimizers',
    #     config = {
    #         'batch_size': 128,
    #         'lr': 0.001,
    #         'epochs': 3,
    #     }
    # ) 
    epochs = config.epochs
    lr = config.lr
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    optimizer_name = config.optimizer
    optimizer_class = getattr(t.optim, optimizer_name)
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    model = ConvNet().to(device).train()
    weight_params = [
        p_val for p_name, p_val in model.named_parameters() 
        if 'bias' not in p_name
    ]
    bias_params = [
        p_val for p_name, p_val in model.named_parameters() 
        if 'bias' in p_name
    ]
    optimizer = optimizer_class(
        [{'params': weight_params, 'weight_decay': weight_decay}, 
        {'params': bias_params, 'weight_decay': 0}], 
        lr=lr
    )
    loss_list = []
    accuracy_list = []
    start_time = time.time()
    examples_seen = 0
    
    for _ in tqdm_notebook(range(epochs)):
        
        for (x, y) in tqdm_notebook(trainloader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            
            examples_seen += len(y)
            loss_list.append(loss.item())
            wandb.log({
                "train_loss": loss, "elapsed": time.time() - start_time
            }, step=examples_seen)
        
        with t.inference_mode():
            accuracy = 0
            total = 0
            for (x, y) in testloader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            accuracy_list.append(accuracy/total)
            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)
            
    
    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

#%%
sweep_configuration = {
    'method': 'random',
    'name': 'convnet_sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'train_loss'
		},
    'parameters': {
        'batch_size': {'values': [128]},
        'epochs': {'values': [5]},
        'lr': {'values': [0.001, .01]},
        # 'lr': {'min': .0001, 'max': .01, 'distribution': 'loguniform'},
        'optimizer': {'values': ['Adam', 'RMSprop']},
        'weight_decay': {'values': [0.0, 0.1]},
     }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="convnet_optimizers")
wandb.agent(sweep_id, count=10, function=train_convnet)
#%% 
# Weight decay is bad
# Adam dominates
# 