import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from CCRLDataset import CCRLDataset
from AlphaZeroNetwork import AlphaZeroNet
import matplotlib.pyplot as plt

# Training params
num_epochs = 2
num_blocks = 20
num_filters = 256
ccrl_dir = 'C:/Users/david/Documents/training data reformatted'
logmode = True
cuda = True


def train():
    train_ds = CCRLDataset(ccrl_dir)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=8)

    if cuda:
        alphaZeroNet = AlphaZeroNet(num_blocks, num_filters).cuda()
    else:
        alphaZeroNet = AlphaZeroNet(num_blocks, num_filters)
    optimizer = optim.Adam(alphaZeroNet.parameters())
    mseLoss = nn.MSELoss()

    print('Starting training')

    value_losses = []
    policy_losses = []
    total_losses = []

    for epoch in range(num_epochs):
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_total_loss = 0.0
        alphaZeroNet.train()
        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad()

            if cuda:
                position = data['position'].cuda()
                valueTarget = data['value'].cuda()
                policyTarget = data['policy'].cuda()
            else:
                position = data['position']
                valueTarget = data['value']
                policyTarget = data['policy']

            valueLoss, policyLoss = alphaZeroNet(position, valueTarget=valueTarget, policyTarget=policyTarget)
            loss = valueLoss + policyLoss

            loss.backward()
            optimizer.step()

            epoch_value_loss += float(valueLoss)
            epoch_policy_loss += float(policyLoss)
            epoch_total_loss += float(loss)

            message = 'Epoch {:03} | Step {:05} / {:05} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                epoch, iter_num, len(train_loader), float(valueLoss), float(policyLoss))

            if iter_num != 0 and not logmode:
                print(('\b' * len(message)), end='')
            print(message, end='', flush=True)
            if logmode:
                print('')

        print('')

        # Calculate average losses for the epoch
        avg_value_loss = epoch_value_loss / len(train_loader)
        avg_policy_loss = epoch_policy_loss / len(train_loader)
        avg_total_loss = epoch_total_loss / len(train_loader)

        value_losses.append(avg_value_loss)
        policy_losses.append(avg_policy_loss)
        total_losses.append(avg_total_loss)

        networkFileName = 'AlphaZeroNet_{}x{}.pt'.format(num_blocks, num_filters)
        torch.save(alphaZeroNet.state_dict(), networkFileName)
        print('Saved model to {}'.format(networkFileName))

    # Plot the losses
    plt.figure()
    plt.plot(range(num_epochs), value_losses, label='Value Loss')
    plt.plot(range(num_epochs), policy_losses, label='Policy Loss')
    plt.plot(range(num_epochs), total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses Over Epochs')
    plt.savefig('training_losses.png')
    plt.show()


if __name__ == '__main__':
    train()
