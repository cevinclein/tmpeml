from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import numpy as np

def save_histogram(weights, epoch, l2_reg):
    """Save the normalized histogram of weights."""
    plt.figure()
    plt.hist(weights.cpu().detach().numpy().flatten(), bins=50, color='blue', alpha=0.7, density=True)
    plt.xlabel("Weight Values")
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of Weights (L2: {l2_reg}, Epoch: {epoch})")
    if not os.path.exists('img_02'):
        os.makedirs('img_02')
    plt.savefig(f"img_02/histogram_L2_{l2_reg}_epoch_{epoch}.png")
    plt.close()


def save_plots(train_loss, test_loss, test_accuracy, epochs, appendd = ""):
    # Create directory for plots if it doesn't exist
    if not os.path.exists('img_02'):
        os.makedirs('img_02')

    # Plotting Training Loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss {appendd}')
    plt.savefig('img_02/training_loss.png')
    plt.close()

    # Plotting Test Loss
    plt.figure()
    plt.plot(range(1, epochs + 1), test_loss, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Test Loss {appendd}')
    plt.savefig('img_02/test_loss.png')
    plt.close()

    # Plotting Test Accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), test_accuracy, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Test Accuracy {appendd}')
    plt.savefig('img_02/test_accuracy.png')
    plt.close()

    # Plotting All Metrics
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss, label='Training Loss', color='blue')
    plt.plot(range(1, epochs + 1), test_loss, label='Test Loss', color='red')
    plt.plot(range(1, epochs + 1), test_accuracy, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title(f'Training and Test Metrics')
    plt.legend()
    plt.savefig('img_02/all_metrics.png')
    plt.close()

class VGG11(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
           
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)  

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.time(),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Current time: {:.4f}; Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        time.time(),
        epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Example with L2 Regularization")
    parser.add_argument("--batch-size", type=int, default=128, help="Input batch size for training")
    parser.add_argument("--test-batch-size", type=int, default=1024, help="Input batch size for testing")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--l2-regs", type=str, default="0.001,0.0001,0.000001",
                        help="Comma-separated list of L2 regularization strengths")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=200, help="Logging interval")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(device)

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    l2_values = [float(val) for val in args.l2_regs.split(",")]
    results = {"l2": [], "test_accuracy": []}
    last_cv_layer = 18
    
    for l2_reg in l2_values:
        print(f"\nTraining with L2 Regularization: {l2_reg}")
        model = VGG11().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2_reg)

        train_loss, test_loss, test_accuracy = [], [], []
        for epoch in range(1, args.epochs + 1):
            train_loss.append(train(args, model, device, train_loader, optimizer, epoch))
            tl, acc = test(model, device, test_loader, epoch)
            test_loss.append(tl)
            test_accuracy.append(acc)

            # Save weight histogram for the last convolutional layer
            if epoch == args.epochs:
                last_layer_weights = model.features[last_cv_layer].weight
                save_histogram(last_layer_weights, epoch, l2_reg)

        results["l2"].append(l2_reg)
        results["test_accuracy"].append(np.mean(test_accuracy))

    # Plot L2 Regularization vs Test Accuracy
    plt.figure()
    plt.plot(results["l2"], results["test_accuracy"], marker="o")
    plt.xscale("log")
    plt.xlabel("L2 Regularization Strength")
    plt.ylabel("Test Accuracy (%)")
    plt.title("L2 Regularization vs Test Accuracy")
    plt.grid(True)
    plt.savefig("img_02/l2_vs_accuracy.png")
    plt.close()

if __name__ == '__main__':
    main()
