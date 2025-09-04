import torch
from torch import from_numpy
from torch.nn import Module, L1Loss, Linear
from torch.nn.functional import relu
from torch.optim import Adam
def make_network(n_in, n_out):
    class Net(Module):

        def __init__(self):
            super(Net, self).__init__()
            self.layer1 = Linear(n_in, 1024)
            self.layer2 = Linear(1024, 1024)
            self.layer3 = Linear(1024, 1024)
            self.layer4 = Linear(1024, n_out)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x

    return Net()


def main(inputs, outputs, learning_rate=0.05, niter=100):
    # inputs shape = nsample x nparam
    # outputs shape = nsample x ndata_vector
    print(inputs.shape, outputs.shape)
    n_in = inputs.shape[1]
    n_out = outputs.shape[1]
    inputs = from_numpy(inputs).float()
    outputs = from_numpy(outputs).float()
    loss_fn = L1Loss()
    model = make_network(n_in, n_out)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    for i in range(niter):
        print(f"Training step {i}")
        optimizer.zero_grad()
        predictions = model(inputs)
        print(predictions.shape, outputs.shape)
        loss = loss_fn(predictions, outputs)
        loss.backward()
        optimizer.step()
        print(loss)

    return model
