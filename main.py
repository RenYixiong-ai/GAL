import os
import sys

from models_struc.modelset import *
from loss.loss import *
from train.train import *
from utils import *
from loss.attack import attack

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import copy
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


# Record current working directory
local_path = os.getcwd()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64  
    tot_epoch = 10
    attack_num = 40
    data_name = "MNIST"

    epsilon_list = np.linspace(0.01, 0.9, attack_num)

    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Open log file
    log_path = "experiment_log.txt"
    log_file = open(log_path, "w")

    tot_NN = MultiLayerNetwork()
    tot_NN.add(nn.Flatten())

    input_size = 28 * 28
    alpha_list = [1.8, 1.05, 2.62]
    size_range = [1000, 1000, 1000]
    beta_list = [0.7, 0.6, 1.4]

    for num_layer, [output_size, alpha, beta] in enumerate(zip(size_range, alpha_list, beta_list)):
        print(f'start layer{num_layer}')
        log_file.write(f"===== Start Layer {num_layer} =====\n")

        Single_NN = SingleLayerNetwork(input_size, output_size, use_layernorm=True).to(device)
        optimizer = optim.Adam(Single_NN.parameters(), lr=0.001)
        criterion_cross = nn.CrossEntropyLoss()
        criterion_fbm = FDBLoss(alpha)
        readout_head = ReadoutHead(output_size, 10).to(device)

        for epoch in range(tot_epoch):
            Single_NN.train()
            loss, loss_fbm, loss_cross = train_with_readout(
                fixed_network=tot_NN,
                target_network=Single_NN,
                readout_head=readout_head,
                data_loader=trainloader,
                optimizer=optimizer,
                criterion_cross=criterion_cross,
                criterion_fbm=criterion_fbm,
                beta=beta,
                device=device
            )

            # Analyze learned features
            tot_NN.add(Single_NN.clone_self())
            accuracy = evaluate_accuracy(tot_NN, testloader, device, readout_head)

            tot_NN.eval()
            dis_fermi, dis_boson = 0, 0
            num_batches = len(trainloader)
            with torch.no_grad():
                for inputs, labels in trainloader:
                    inputs = inputs.view(inputs.shape[0], -1).to(device)
                    labels = labels.to(device)
                    output = tot_NN(inputs)
                    labels = F.one_hot(labels).float()
                    dis_f, dis_b = FB_smi_distance(output, labels, re_grad=False)
                    dis_fermi += dis_f.cpu().item() / num_batches
                    dis_boson += dis_b.cpu().item() / num_batches

            # Perform adversarial attacks (record accuracy only)
            tot_NN.add(readout_head)
            acc_f, _ = attack(tot_NN, device, testset, epsilon=0.15, attack_type='fgsm')
            acc_g, _ = attack(tot_NN, device, testset, epsilon=0.15, attack_type='gaussian')
            tot_NN.pop()

            # Write metrics to log
            log_file.write(
                f"[Epoch {epoch}] Layer {num_layer} | "
                f"Total Loss: {loss:.4f}, FBM Loss: {loss_fbm:.4f}, Cross Loss: {loss_cross:.4f}, "
                f"Fermi: {dis_fermi:.4f}, Boson: {dis_boson:.4f}, "
                f"Ratio: {dis_fermi / dis_boson:.4f}, "
                f"Acc: {accuracy:.4f}, "
                f"FGSM Robust Acc: {acc_f:.4f}, Gaussian Robust Acc: {acc_g:.4f}\n"
            )
            log_file.flush()
            tot_NN.pop()

        # Save layer-level summary metrics
        log_file.write(
            f"[Layer {num_layer} Summary] Total Loss: {loss:.4f}, FBM Loss: {loss_fbm:.4f}, "
            f"Cross Loss: {loss_cross:.4f}, Fermi: {dis_fermi:.4f}, "
            f"Boson: {dis_boson:.4f}, Ratio: {dis_fermi/dis_boson:.4f}, "
            f"Acc: {accuracy:.4f}\n"
        )
        log_file.flush()

        input_size = output_size
        tot_NN.add(Single_NN.clone_self())

        # Train the readout head
        readout_head = train_readout(tot_NN, trainloader, output_size, tot_epoch, device)

        # Scan over attack strength epsilon
        for step, epsilon in enumerate(epsilon_list):
            tot_NN.add(readout_head)
            acc_f, _ = attack(tot_NN, device, testset, epsilon, attack_type='fgsm')
            acc_g, _ = attack(tot_NN, device, testset, epsilon, attack_type='gaussian')
            tot_NN.pop()
            log_file.write(
                f"[Attack] Layer {num_layer}, Eps={epsilon:.3f}, "
                f"FGSM Acc={acc_f:.4f}, Gaussian Acc={acc_g:.4f}\n"
            )
            log_file.flush()

        # Save readout model for this layer
        os.makedirs("model", exist_ok=True)
        model_path = f"model/readout_head{num_layer}.pth"
        torch.save(readout_head.state_dict(), model_path)

    # Save the complete network
    tot_NN.add(readout_head)
    os.makedirs("model", exist_ok=True)
    model_path = "model/tot_NN.pth"
    torch.save(tot_NN.state_dict(), model_path)

    final_eval = evaluate_accuracy(tot_NN, testloader, device=device)
    log_file.write(f"Final Eval Accuracy: {final_eval:.4f}\n")
    log_file.close()
    print("Training completed, log written to experiment_log.txt")

