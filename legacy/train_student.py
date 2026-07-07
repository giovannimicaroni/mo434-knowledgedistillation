import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_student(student, teacher, dataset, epochs, lr, device='cuda'):
    criterion = nn.MSELoss() 

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    student.to(device)
    teacher.to(device)

    student.train() 
    teacher.eval()

    optimizer = optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        
        # Nested loop: Iterate over the batches
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            # Step A: Forward Pass
            predictions = student(batch_X)

            with torch.no_grad(): # Disable gradient calculation to save memory and speed up inference
                teacher_output = teacher(batch_X)
            
            # Step B: Loss
            loss = criterion(predictions, teacher_output)
            
            # Step C: Backward Pass
            optimizer.zero_grad()
            loss.backward()
            
            # Step D: Update
            optimizer.step()
            
            # Track the loss for the whole epoch
            epoch_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.4f}')

    print("\nTraining complete!")

    return

def evaluate_student(student, rest_of_teacher, dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    student.eval()
    rest_of_teacher.eval()
    
    student.to(device)
    rest_of_teacher.to(device)

    correct = 0
    total = 0

    with torch.no_grad(): 
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            features = student(images)
            outputs = rest_of_teacher(features)

            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy
                
