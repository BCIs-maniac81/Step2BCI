import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from deepproc.neuroDevice import DeviceSwitcher


class NetPytorch(torch.nn.Module):
    def __init__(self, input_size, output_size, *hidden_sizes, dropout_prob=.2):
        super(NetPytorch, self).__init__()
        # Input and output size of the first and last linear layers
        self.input_size = input_size
        self.output_size = output_size

        # Define fully connected layers with variable hidden sizes
        self.fc_layers = torch.nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.fc_layers.append(torch.nn.Linear(prev_size, size))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.Dropout(p=dropout_prob))
            prev_size = size
        self.fc_layers.append(torch.nn.Linear(prev_size, output_size))

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x


class TorchTrainer:
    def __init__(self, model, train_data, batch_size=32, max_epochs=10, learning_rate=0.01, momentum=0.9,
                 weight_decay=0.0, device=None, criterion=torch.nn.CrossEntropyLoss, optimizer=optim.SGD,
                 eval_train=True):
        self.early_stopping_epoch = None
        self.net = model
        self.train_data = train_data
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_accuracy = []
        self.epoch_losses = []
        self.eval_losses = []

        # Define device to use for training and move model to device
        self.device_switcher = DeviceSwitcher(force_device=device)
        self.net.to(self.device_switcher.device)

        # Define loss function and optimizer
        self.criterion = criterion()
        self.optimizer = optimizer(self.net.parameters(), lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.learning_rate, epochs=self.max_epochs,
                                    steps_per_epoch=len(self.train_data))

        # Define train data loader
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def train(self, eval_train=False, patience=15, save_model=True):
        if eval_train:
            evaluator = TorchEvaluator(self.net, self.criterion, batch_size=self.batch_size)
            best_val_loss = float('inf')
            best_model_params = None
        for epoch in range(1, self.max_epochs + 1):
            train_loss = 0
            # Set model to train mode
            self.net.train()
            # Train for one epoch
            for i, (inputs, targets) in enumerate(self.train_loader):
                # Move batch items to device
                inputs = self.device_switcher.to_device(inputs)
                targets = self.device_switcher.to_device(targets)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                self.optimizer.step()

                self.scheduler.step()
                train_loss += loss.item()
            epoch_loss = train_loss / len(self.train_loader)
            self.epoch_losses.append(epoch_loss)

            if eval_train:
                print(f'Epoch[{epoch}/{self.max_epochs}] .......... epoch loss: {epoch_loss}', end=' ... ')
            else:
                print(f'Epoch[{epoch}/{self.max_epochs}] .......... epoch loss: {epoch_loss}')
            if eval_train:
                accuracy, eval_loss = evaluator.evaluate(self.train_data, None, True)
                print(f' ....... validation loss: {eval_loss}')
                self.train_accuracy.append(accuracy)
                self.eval_losses.append(eval_loss)
                # Early stopping
                if eval_loss < best_val_loss:
                    best_val_loss = eval_loss
                    best_model_params = self.net.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Arrêt précoce après {epoch - 1} epochs')
                    self.early_stopping_epoch = epoch - 1
                    break
        if best_model_params is not None:
            self.net.load_state_dict(best_model_params)
            if save_model:
                current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path = f"../deepproc/saved_models/best_model_{current_date}.pt"
                torch.save(self.net.state_dict(), save_path)
        return self.net, self.train_accuracy[-1], self.epoch_losses, self.eval_losses


class TorchEvaluator:
    def __init__(self, model, criterion=None, batch_size=32, device=None):
        self.val_loader = None
        self.net = model
        self.batch_size = batch_size
        self.device_switcher = DeviceSwitcher(force_device=device)
        self.net.to(self.device_switcher.device)
        self.criterion = criterion

    def evaluate(self, val_data, criterion=torch.nn.CrossEntropyLoss, eval_train=False):
        if not eval_train:
            self.criterion = criterion()
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        # Set model to eval mode
        self.net.eval()
        eval_loss = 0
        correct = 0
        total = 0
        # Disable gradients for evaluation
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                inputs = self.device_switcher.to_device(inputs)
                targets = self.device_switcher.to_device(targets)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                eval_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            eval_loss = eval_loss / len(self.val_loader)
        accuracy = correct / total

        if not eval_train:
            print(f'Précision sur l\'ensemble de validation: {accuracy:.5f}')
        else:
            print(f'Précision sur l\'ensemble d\'entraînement: {accuracy:.5f}')
        return accuracy, eval_loss


class TorchPredictor(object):
    def __init__(self, model, input_size, device='cpu'):
        self.net = model
        self.device = device
        self.input_size = input_size

    def predict_single_sample(self, sample):
        self.net.eval()
        if isinstance(sample, (list, tuple)):
            sample = (torch.tensor(sample)).reshape(-1, self.input_size)
        elif isinstance(sample, np.ndarray):
            sample = (torch.from_numpy(sample)).reshape(-1, self.input_size)
        elif isinstance(sample, torch.Tensor):
            sample = sample.reshape(-1, self.input_size)
        else:
            print('Unknown Type of data !!')
        with torch.no_grad():
            sample = sample.to(self.device)
            output = self.net(sample)
            _, predicted_label = torch.max(output, dim=1)
            return predicted_label

    def predict_batch(self, batch):
        self.net.eval()
        if isinstance(batch, (list, tuple)):
            batch = torch.tensor(batch).reshape(-1, self.input_size)
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch).reshape(-1, self.input_size)
        elif isinstance(batch, torch.Tensor):
            batch = batch.reshape(-1, self.input_size)
        else:
            print('Type de données inconnu !!')
        with torch.no_grad():
            batch = batch.to(self.device, non_blocking=True)
            outputs = self.net(batch)
            _, predicted_labels = torch.max(outputs, dim=1)
            return predicted_labels
