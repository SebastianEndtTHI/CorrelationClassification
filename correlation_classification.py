import argparse
import sys
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import sklearn.metrics as metrics
import corr_class_visualization as visualization


class CNN(nn.Module):
    def __init__(self, dropout_rate, params):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=0, stride=1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, params['max_no_comp'])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(F.max_pool2d(x, 2))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)


def train(model, device, train_loader, optimizer, train_losses):
    model.train()
    train_loss = 0
    for batch, (input_data, ref_data) in enumerate(train_loader):
        input_data, ref_data = input_data.to(device), ref_data.to(device)
        optimizer.zero_grad()
        output = model(input_data)
        loss = F.cross_entropy(output, ref_data - 1)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader.dataset)/train_loader.batch_size
    print('Training: \tAverage loss: {:.6f}'.format(train_loss))
    train_losses.append(train_loss)


def validate(model, device, val_loader, val_losses, val_correct_prct, best_val_loss, models, ep_wo_improv):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for input_data, ref_data in val_loader:
            input_data, ref_data = input_data.to(device), ref_data.to(device)
            output = model(input_data)
            val_loss += F.cross_entropy(output, ref_data - 1, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) + 1  # get the index of the max log-probability
            correct += pred.eq(ref_data.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_correct_prct.append(100. * correct / len(val_loader.dataset))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        models.append(model)
        ep_wo_improv = 0
    else:
        ep_wo_improv += 1
        print('No improvement in validation since {} epochs'.format(ep_wo_improv))

    print('Validation: Average loss: {:.6f}\tAccuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))
    print('Validation: Best loss: {:.6f}'.format(best_val_loss))

    return best_val_loss, ep_wo_improv


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_data, ref_data in test_loader:
            input_data, ref_data = input_data.to(device), ref_data.to(device)
            output = model(input_data)
            test_loss += F.cross_entropy(output, ref_data - 1, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) + 1  # get the index of the max log-probability
            correct += pred.eq(ref_data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_correct_prct = 100. * correct / len(test_loader.dataset)

    print('\nTest: Average loss: {:.6f}\tAccuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_correct_prct))

    return ref_data, pred, output, test_loss, test_correct_prct


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Settings
    parser = argparse.ArgumentParser(description='Correlation MRI classifier')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs (default: 3000)')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size (default: 10000)')
    parser.add_argument('--learning_rate', type=float, default=1, help='learning rate (default: 1)')
    parser.add_argument('--rel_end_learning_rate', type=float, default=0.01,
                        help='learning rate at end of training relative to learning_rate (default: 0.01)')
    parser.add_argument('--stepLR_steps', type=int, default=5, help='number of steps in StepLR scheduler (default: 5)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (default: 0.2)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD optimizer momentum (default: 0.9)')
    parser.add_argument('--save_model', default=False, help='save the model (default: False)')
    args = parser.parse_args()

    # Init device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('using device: ', device)

    # Load data
    dataset = sio.loadmat('demo_data.mat')

    train_data = dataset['input_train'].astype(np.float32)
    val_data = dataset['input_val'].astype(np.float32)
    test_data = dataset['input_test'].astype(np.float32)
    train_labels = dataset['label_train'].astype(np.int64).squeeze()
    val_labels = dataset['label_val'].astype(np.int64).squeeze()
    test_labels = dataset['label_test'].astype(np.int64).squeeze()
    test_noiselevels = dataset['noiselevel_test'].astype(np.int64).squeeze()

    params = {'P': train_data.shape[1], 'max_no_comp': np.max(train_labels)}
    
    train_dataset = TensorDataset(torch.tensor(train_data).to(device), torch.tensor(train_labels).to(device))
    val_dataset = TensorDataset(torch.tensor(val_data).to(device), torch.tensor(val_labels).to(device))
    test_dataset = TensorDataset(torch.tensor(test_data).to(device), torch.tensor(test_labels).to(device))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_labels), shuffle=False)

    # Initialization
    model = CNN(args.dropout, params).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs/args.stepLR_steps,
                                    gamma=args.rel_end_learning_rate ** (1./args.stepLR_steps))

    # Training loop
    train_losses = []
    val_losses = []
    val_correct_prct = []
    best_val_loss = sys.float_info.max
    models = [model]
    ep_wo_improv = 0
    for ep in range(1, args.epochs + 1):
        print('\nEpoch {}'.format(ep))
        train(model, device, train_loader, optimizer, train_losses)
        best_val_loss, ep_wo_improv = validate(model, device, val_loader, val_losses, val_correct_prct, best_val_loss,
                                               models, ep_wo_improv)
        scheduler.step()

    # Save best model
    best_model = models[-1]
    if args.save_model:
        torch.save(best_model.state_dict(), "model.pt")

    # Test
    test_ref, test_pred, test_output, test_loss, test_correct_prct = test(best_model, device, test_loader)
    test_ref = test_ref.cpu().numpy().astype(np.int64).squeeze()
    test_pred = test_pred.cpu().numpy().astype(np.int64).squeeze()
    test_output = test_output.cpu().numpy()

    test_ref_binary = test_ref.copy()
    test_ref_binary[test_ref_binary >= 2] = 2
    test_pred_binary = test_pred.copy()
    test_pred_binary[test_pred_binary >= 2] = 2

    # Noise dependency - binary classification
    acc_per_noise_binary = []
    for jj in range(5):
        acc_per_noise_binary.append(np.sum(test_pred_binary[test_noiselevels == (jj + 1)] ==
                                           test_ref_binary[test_noiselevels == (jj + 1)]) *
                                    100. / np.sum([test_noiselevels == (jj + 1)]))
    # Noise dependency - multiclass classification
    acc_per_noise = []
    for jj in range(5):
        acc_per_noise.append(np.sum(test_pred[test_noiselevels == (jj + 1)] == test_ref[test_noiselevels == (jj + 1)]) *
                             100. / np.sum([test_noiselevels == (jj + 1)]))

    print('\nFinal scores:\n')
    # Metrics - binary classification
    print('\nBinary:')
    print('accuracy binary: {:.4f}'.format(metrics.accuracy_score(test_ref_binary, test_pred_binary)))
    print('precision binary: {:.4f}'.format(metrics.precision_score(test_ref_binary, test_pred_binary)))
    print('recall binary: {:.4f}'.format(metrics.recall_score(test_ref_binary, test_pred_binary)))
    print('F1 binary: {:.4f}'.format(metrics.f1_score(test_ref_binary, test_pred_binary)))

    # Metrics - multiclass classification
    print('\nMulticlass:')
    print('Accuracy: {:.4f}'.format(metrics.accuracy_score(test_ref, test_pred)))
    print('macro precision: {:.4f}'.format(metrics.precision_score(test_ref, test_pred, average='macro')))
    print('micro precision: {:.4f}'.format(metrics.precision_score(test_ref, test_pred, average='micro')))
    print('macro recall: {:.4f}'.format(metrics.recall_score(test_ref, test_pred, average='macro')))
    print('micro recall: {:.4f}'.format(metrics.recall_score(test_ref, test_pred, average='micro')))
    print('macro F1: {:.4f}'.format(metrics.f1_score(test_ref, test_pred, average='macro')))
    print('micro F1: {:.4f}'.format(metrics.f1_score(test_ref, test_pred, average='micro')))

    # Plots
    visualization.plot_noise_dependency(np.unique(test_noiselevels), acc_per_noise_binary, binary=True)
    visualization.plot_noise_dependency(np.unique(test_noiselevels), acc_per_noise, binary=False)
    visualization.plot_loss_curve(train_losses, val_losses, test_loss)
    visualization.plot_corrects(val_correct_prct, test_correct_prct)
    visualization.plot_confusion_matrix_binary(test_ref_binary, test_pred_binary)
    visualization.plot_confusion_matrix(test_ref, test_pred, params['max_no_comp'])
    visualization.plot_multiclass_roc(test_ref, test_output)


if __name__ == '__main__':
    main()
