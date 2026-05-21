import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_classification(teacher_features, head, dataset, epochs=5, lr=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    teacher_features.to(device)
    head.to(device)
    
    teacher_features.eval()
    for param in teacher_features.parameters():
        param.requires_grad = False
        
    head.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = teacher_features(images)
            
            outputs = head(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")

    return