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
epochs = 3
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

device = "cuda" if t.cuda.is_available() else "cpu"

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

#%%
def train_convnet(
    trainloader: DataLoader, testloader: DataLoader, epochs: int, loss_fn: Callable
) -> list:
    """
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    
    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    """

    # wandb.init(
    #     project='shakespeare',
    #     config = {
    #         'batch_size': 64,
    #         'hidden_size': 512,
    #         'lr': 0.001,
    #         'epochs': 1,
    #         'max_seq_len': 30,
    #         'dropout': 0.1,
    #         'num_layers': 6,
    #         'num_heads': 8,
    #     }
    # ) 
    
    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []
    accuracy_list = []
    start_time = time.time()
    examples_seen = 0
    
    for epoch in tqdm_notebook(range(epochs)):
        
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
            # wandb.log({
            #     "train_loss": loss, "elapsed": time.time() - start_time
            # }, step=examples_seen)
        
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
            # wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)
            
        # print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")
    
    # filename = f"{wandb.run.dir}/model_state_dict.pt"
    # print(f"Saving model to: {filename}")
    # t.save(model.state_dict(), filename)
    # wandb.save(filename)

#%%
train_convnet(
    trainloader, testloader, epochs, loss_fn
)
#%%