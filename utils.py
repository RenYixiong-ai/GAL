import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt

####################################################
#################### Random Seeds ##################
####################################################


def set_seed(seed):
    """Set random seeds for reproducibility."""

    # Python's built-in random
    random.seed(seed)

    # NumPy seed
    np.random.seed(seed)

    # PyTorch seed
    torch.manual_seed(seed)

    # Ensure deterministic behavior on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Configure CuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


####################################################
#################### Dataset Loading ################
####################################################


def load_MNIST():
    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training set
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Download and load the test set
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Quick sanity check
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    
    return train_loader, test_loader


def load_small_MNIST(loaad_size=100):
    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the full MNIST training set
    full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # Download and load the test set
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize index list for each class
    class_indices = {i: [] for i in range(10)}

    # Iterate over dataset to record indices for each class
    for idx, (_, label) in enumerate(full_train_dataset):
        if len(class_indices[label]) < loaad_size:  # cap number of samples per class
            class_indices[label].append(idx)
        if all(len(class_indices[i]) == loaad_size for i in range(10)):
            break  # stop once each class has enough samples

    # Merge indices of all classes to form a subset
    selected_indices = [idx for indices in class_indices.values() for idx in indices]

    # Create the reduced training dataset
    small_train_dataset = Subset(full_train_dataset, selected_indices)
    train_loader = DataLoader(small_train_dataset, batch_size=64, shuffle=True)

    # Quick sanity check
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Sample labels: {labels}")
    return train_loader, test_loader


def load_FashionMNIST():
    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training set
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Download and load the test set
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Quick sanity check
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")

    return train_loader, test_loader

def load_cifar10():
    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize RGB channels individually
    ])

    # Download and load the training set
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Download and load the test set
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Quick sanity check
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")

    return train_loader, test_loader

def load_small_cifar10(loaad_size=100):
    # Define transforms: convert to grayscale and normalize
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # convert RGB to single-channel grayscale
        transforms.ToTensor(),                        # to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))          # normalize grayscale
    ])

    # Load the full CIFAR-10 training set
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    # Download and load the test set
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize index list for each class
    class_indices = {i: [] for i in range(10)}

    # Record indices for each class
    for idx, (_, label) in enumerate(full_train_dataset):
        if len(class_indices[label]) < loaad_size:  # cap samples per class
            class_indices[label].append(idx)
        if all(len(class_indices[i]) == loaad_size for i in range(10)):
            break  # stop once each class has enough samples

    # Merge indices into subset
    selected_indices = [idx for indices in class_indices.values() for idx in indices]

    # Create the reduced training dataset
    small_train_dataset = Subset(full_train_dataset, selected_indices)
    train_loader = DataLoader(small_train_dataset, batch_size=64, shuffle=True)

    # Quick sanity check
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Sample labels: {labels}")
    return train_loader, test_loader

def get_dataloader(dataset_name, batch_size=64, root='./data', train=True, selected_classes=None, class_counts=None):
    """Load a dataset with appropriate normalization.

    Parameters
    ----------
    dataset_name : str
        One of 'MNIST', 'FashionMNIST', 'KMNIST', 'CIFAR10', 'CIFAR100'.
    batch_size : int
        Batch size for the loader.
    root : str
        Path for downloading/loading the dataset.
    train : bool
        Whether to load the training split (``False`` loads the test set).
    selected_classes : list, optional
        Subset of classes to include; ``None`` keeps all classes.
    class_counts : list, optional
        Number of samples per class corresponding to ``selected_classes``.

    Returns
    -------
    DataLoader
        Configured data loader.
    """

    # Normalize parameters for each dataset
    normalize_params = {
        'MNIST': ((0.5,), (0.5,)),
        'FashionMNIST': ((0.2860,), (0.3530,)),
        'KMNIST': ((0.1904,), (0.3475,)),
        'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'CIFAR100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    }

    # Validate dataset name
    if dataset_name not in normalize_params:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    # Fetch mean and std for normalization
    mean, std = normalize_params[dataset_name]

    # Build transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Select dataset according to name
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    elif dataset_name == 'FashionMNIST':
        dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=True)
    elif dataset_name == 'KMNIST':
        dataset = datasets.KMNIST(root=root, train=train, transform=transform, download=True)
    elif dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    elif dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    # Filter specific classes and counts
    if selected_classes is not None and class_counts is not None:
        # Prepare list of indices matching the requirements
        selected_indices = []
        class_counts_dict = {cls: count for cls, count in zip(selected_classes, class_counts)}

        # Iterate over dataset to gather desired samples
        for idx, (_, label) in enumerate(dataset):
            if label in selected_classes and class_counts_dict[label] > 0:
                selected_indices.append(idx)
                class_counts_dict[label] -= 1

            # Stop when all class counts are satisfied
            if all(count == 0 for count in class_counts_dict.values()):
                break

        # Create subset with selected indices
        dataset = Subset(dataset, selected_indices)

    # If classes aren't specified but counts per class are
    elif selected_classes is None and class_counts is not None:
        # Prepare list of indices matching the requirements
        selected_indices = []
        class_counts_dict = {i: count for i, count in zip(range(10), class_counts)}  # assume 10 classes

        # Iterate over dataset to gather desired samples
        for idx, (_, label) in enumerate(dataset):
            if class_counts_dict[label] > 0:
                selected_indices.append(idx)
                class_counts_dict[label] -= 1

            # Stop when all class counts are satisfied
            if all(count == 0 for count in class_counts_dict.values()):
                break

        # Create subset with selected indices
        dataset = Subset(dataset, selected_indices)

    # Construct DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

####################################################
################### Dataset Processing #############
####################################################


def deal_dataloader(input_loader, model, device, batch_size=64):
    """Process a dataloader through a frozen model.

    After training a layer, the next layer uses the previous layer's output
    as input. This helper runs the dataset through ``model`` and returns a
    new dataloader of features and labels.
    """

    processed_data = []
    labels_data = []

    with torch.no_grad():  # disable gradients for efficiency
        for images, labels in input_loader:
            images = images.to(device)

            # Forward images through the model
            features = model(images)

            # Save outputs for later
            processed_data.append(features.cpu())
            labels_data.append(labels.cpu())

    processed_data = torch.cat(processed_data, dim=0)
    labels_data = torch.cat(labels_data, dim=0)

    output_dataset = TensorDataset(processed_data, labels_data)
    output_data_loader = DataLoader(output_dataset, batch_size=batch_size, shuffle=True)

    return output_data_loader

####################################################
#################### Label Processing ##############
####################################################


def efface_label(selected_labels, labels, num_classes):
    """Convert specified labels to one-hot form while zeroing others."""

    one_hot_labels = torch.zeros((labels.size(0), num_classes))

    for i, label in enumerate(labels):
        if label.item() in selected_labels:
            one_hot_labels[i] = F.one_hot(label, num_classes=num_classes).float()

    return one_hot_labels

####################################################
################ Fermi-Bose Distance ###############
####################################################


def fast_FBDistance(features, labels):
    """Return the average distance matrix between classes."""

    labels_list = torch.unique(labels)
    labels_count = len(labels_list)

    _, dim = features.shape

    features_list = [[] for _ in range(labels_count)]
    for i in range(labels_count):
        idx_label = labels == i
        features_list[i] = features[idx_label]

    labels_matrix = torch.zeros(labels_count, labels_count)

    for i in range(labels_count):
        for j in range(i, labels_count):  # only compute upper triangle
            sample_diff = (
                features_list[i].unsqueeze(0) - features_list[j].unsqueeze(1)
            )

            D_matrix = torch.norm(sample_diff, dim=2) / dim

            labels_matrix[i, j] = torch.sum(D_matrix) / (
                len(features_list[i]) * len(features_list[j])
            )

    return labels_matrix

####################################################
##################### 2D Visualization #############
####################################################

# Visualize data by projecting to 2D
def visualize_2d(test_loader, model=None, device=None):
    """Use PCA to project features to 2D for visualization."""
    if model is None:
        deal_test_loader = test_loader
    else:
        deal_test_loader = deal_dataloader(test_loader, model, device, batch_size = 64)

    features, labels = next(iter(deal_test_loader))
    dims = torch.prod(torch.tensor(features.shape[1:]))

    # Gather all data from loader
    deal_test_features = []
    deal_test_labels = []

    for batch_data, batch_labels in deal_test_loader:
        deal_test_features.append(batch_data.view(-1, dims))
        deal_test_labels.append(batch_labels)

    # Merge all batches
    deal_test_features = torch.cat(deal_test_features, dim=0)
    deal_test_labels = torch.cat(deal_test_labels, dim=0)

    features_np = deal_test_features.numpy()  # flatten
    labels_np = deal_test_labels.numpy()

    # Reduce dimensionality to 2D using PCA (t-SNE optional)
    _, dims = features.shape

    if dims > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_np)
        explained_variance_ratio = pca.explained_variance_ratio_
    else:
        features_2d = features_np
        explained_variance_ratio = None

    # Plot 2D scatter
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        mask = labels_np == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            label=f"Class {label}",
            alpha=0.6
        )
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Visualization of High-Dimensional Data")
    plt.legend()
    plt.grid(True)

    # Display PCA variance ratio
    if explained_variance_ratio is not None:
        info_text = f"Explained Variance:\nDim 1: {explained_variance_ratio[0]:.2%}\nDim 2: {explained_variance_ratio[1]:.2%}"
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))

    plt.show()

# Visualization function: reduce data to 3D and plot
def visualize_3d(test_loader, model=None, device=None):
    """Use PCA to project data to 3D and visualize."""
    if model is None:
        deal_test_loader = test_loader
    else:
        deal_test_loader = deal_dataloader(test_loader, model, device, batch_size=64)

    features, labels = next(iter(deal_test_loader))
    dims = torch.prod(torch.tensor(features.shape[1:]))

    # Gather all data from loader
    deal_test_features = []
    deal_test_labels = []

    for batch_data, batch_labels in deal_test_loader:
        deal_test_features.append(batch_data.view(-1, dims))
        deal_test_labels.append(batch_labels)

    # Merge all batches
    deal_test_features = torch.cat(deal_test_features, dim=0)
    deal_test_labels = torch.cat(deal_test_labels, dim=0)

    features_np = deal_test_features.numpy()
    labels_np = deal_test_labels.numpy()

    # Reduce dimensionality to 3D
    _, dims = features_np.shape

    if dims > 3:
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features_np)
        explained_variance_ratio = pca.explained_variance_ratio_
    else:
        features_3d = features_np
        explained_variance_ratio = None

    # Plot 3D scatter
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(labels_np):
        mask = labels_np == label
        ax.scatter(
            features_3d[mask, 0],
            features_3d[mask, 1],
            features_3d[mask, 2],
            label=f"Class {label}",
            alpha=0.6
        )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_title("3D Visualization of High-Dimensional Data")
    ax.legend()

    # Display PCA variance ratio
    if explained_variance_ratio is not None:
        info_text = (
            f"Explained Variance:\n"
            f"Dim 1: {explained_variance_ratio[0]:.2%}\n"
            f"Dim 2: {explained_variance_ratio[1]:.2%}\n"
            f"Dim 3: {explained_variance_ratio[2]:.2%}"
        )
        plt.figtext(0.02, 0.95, info_text, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))

    plt.show()

####################################################
################## Train Readout Head ##############
####################################################


def train_readout(tot_NN, trainloader, input_size, tot_epoch, device):
    """Train a linear readout on top of the frozen network."""

    new_readout_head = nn.Linear(input_size, 10)
    new_readout_head.to(device)

    optimizer = optim.Adam(new_readout_head.parameters(), lr=0.001)
    criterion_cross = nn.CrossEntropyLoss()
    lambda_l2 = 1e-4  # regularization strength

    deal_trainloader = deal_dataloader(trainloader, tot_NN, device, batch_size=64)

    for epoch in range(tot_epoch):
        for inputs, labels in deal_trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = new_readout_head(inputs)
            loss = criterion_cross(outputs, labels)
            
            # Add L2 regularization
            l2_reg = 0.
            for param in new_readout_head.parameters():
                l2_reg += torch.norm(param, p=2)**2
            loss += lambda_l2 * l2_reg

            loss.backward()
            optimizer.step()

    return new_readout_head


####################################################
#################### Accuracy Utilities ############
####################################################


def KMeans_accuracy(test_loader, model=None, device=None):
    """Compute clustering accuracy using KMeans and Hungarian matching."""
    if model is None:
        deal_test_loader = test_loader
    else:
        deal_test_loader = deal_dataloader(test_loader, model, device, batch_size = 64)

    features, labels = next(iter(deal_test_loader))
    dims = torch.prod(torch.tensor(features.shape[1:]))

    # Gather all data from loader
    deal_test_features = []
    deal_test_labels = []

    for batch_data, batch_labels in deal_test_loader:
        deal_test_features.append(batch_data.view(-1, dims))
        deal_test_labels.append(batch_labels)

    # Merge all batches
    deal_test_features = torch.cat(deal_test_features, dim=0)
    deal_test_labels = torch.cat(deal_test_labels, dim=0)

    data = deal_test_features.numpy()  # flatten
    labels = deal_test_labels.numpy()

    # Apply KMeans clustering
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(data)

    # Build confusion matrix
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int32)
    for cluster in range(n_clusters):
        cluster_labels = labels[cluster_assignments == cluster]
        for label in range(n_clusters):
            confusion_matrix[cluster, label] = np.sum(cluster_labels == label)

    #print("Confusion matrix:")
    #print(confusion_matrix)

    # Use Hungarian algorithm to find best label assignment
    cost_matrix = -confusion_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping from clusters to labels
    cluster_to_label = {cluster: label for cluster, label in zip(row_ind, col_ind)}

    # Map clusters to labels and compute accuracy
    predicted_labels = np.array([cluster_to_label[cluster] for cluster in cluster_assignments])
    accuracy = accuracy_score(labels, predicted_labels)
    #print(f"\nAccuracy: {accuracy * 100:.2f}%")
    return accuracy


def MLP_accuracy(train_loader, test_loader, model=None, device=None, get_model=False):
    """Train and evaluate a simple MLP classifier."""

    class MLP(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.linear = nn.Linear(input_size, num_classes)

        def forward(self, x):
            x = x.view(-1, self.input_size)  # flatten image
            out = self.linear(x)
            #out = F.softmax(out)
            return out

    if model is not None:
        train_loader = deal_dataloader(train_loader, model, device, batch_size = 64)
        test_loader = deal_dataloader(test_loader, model, device, batch_size = 64)
    learning_rate = 0.001
    num_classes = 10
    # Define loss and optimizer
    features, labels = next(iter(test_loader))
    images_size = torch.prod(torch.tensor(features.shape[1:]))
    model0 = MLP(images_size, 10).to(device)
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model0.parameters(), lr=learning_rate)

    model0.train()
    # Train the model
    epochs = 30
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)

            outputs = model0(images)
            loss = criterion2(outputs, labels_one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model0.eval()

    # Accuracy counting
    correct = 0
    total = 0

    # Disable gradients for faster evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to device
            images = images.view(-1, images_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model0(images)
            
            # Get predicted labels
            _, predicted = torch.max(outputs, 1)
            
            # Update counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute accuracy
    accuracy = 1.0 * correct / total
    if get_model :
        return accuracy, model0
    else:
        return accuracy


def evaluate_accuracy(target_network, data_loader, device, readout_head=None):
    """Evaluate classification accuracy of the (optionally extended) network."""

    target_network.eval()
    if readout_head is not None:
        readout_head.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            #inputs = inputs.view(inputs.shape[0], -1)  # flatten if needed
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward through target network
            target_outputs = target_network(inputs)

            if readout_head is not None:
                # Pass through readout head
                target_outputs = readout_head(target_outputs)

            # Predicted class
            _, predicted = torch.max(target_outputs, dim=1)

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute accuracy
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    set_seed(42)

    # Example usage: select 100 samples each from CIFAR-10 classes 0 and 1
    selected_classes = [0, 1]
    class_counts = [100, 100]

    train_loader = get_dataloader(
        'CIFAR10', batch_size=64, train=True,
        selected_classes=selected_classes, class_counts=class_counts,
    )
    test_loader = get_dataloader('CIFAR10', batch_size=64, train=False)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")

