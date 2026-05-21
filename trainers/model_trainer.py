import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, model, lr, epochs, batch_size=32, optimizer_cls=torch.optim.Adam, device=None, criterion=None):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    def fit(self, dataset):
        self.model.to(self.device)
        self.model.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {running_loss / len(loader):.4f}")

    def evaluate(self, dataset):
        self.model.eval()
        self.model.to(self.device)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        correct = total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
