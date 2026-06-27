import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


@torch.no_grad()
def _distill_accuracy(model, teacher, dataset, batch_size, device):
    """Student accuracy: student features passed through the frozen teacher classifier."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = teacher.forward_classifier(model(images))
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total


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

        history = {"loss": []}
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(loader)
            history["loss"].append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")

        return history

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

class ModelTrainerCombined:
    def __init__(self, model, lr, epochs, batch_size=32, optimizer_cls=torch.optim.Adam, device=None, criterion=None):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    def fit(self, dataset, train_eval_dataset=None, val_dataset=None):
        self.model.to(self.device)
        self.model.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        track_acc = train_eval_dataset is not None or val_dataset is not None
        history = {"total": [], "mse": [], "ce": []}
        if track_acc:
            history["train_acc"] = []
            history["val_acc"] = []

        for epoch in range(self.epochs):
            # Initialize trackers for all three losses
            running_loss = 0.0
            running_mse = 0.0
            running_ce = 0.0

            # 1. FIX: Unpack all 3 items returned by your DistillationDataset
            for images, teacher_features, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False):
                images = images.to(self.device)
                teacher_features = teacher_features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                student_features = self.model(images)

                # 2. FIX: Unpack the 3 losses returned by CombinedDistillationLoss
                loss, loss_mse, loss_ce = self.criterion(student_features, teacher_features, labels)

                loss.backward()
                self.optimizer.step()

                # 3. Accumulate all losses using .item()
                running_loss += loss.item()
                running_mse += loss_mse.item()
                running_ce += loss_ce.item()

            # Calculate the averages over the entire epoch
            avg_total = running_loss / len(loader)
            avg_mse = running_mse / len(loader)
            avg_ce = running_ce / len(loader)

            history["total"].append(avg_total)
            history["mse"].append(avg_mse)
            history["ce"].append(avg_ce)

            msg = (
                f"Epoch {epoch + 1}/{self.epochs} -> "
                f"Total Loss: {avg_total:.4f} | "
                f"MSE (Feature): {avg_mse:.4f} | "
                f"CE (Classification): {avg_ce:.4f}"
            )

            if track_acc:
                teacher = self.criterion.teacher
                train_acc = _distill_accuracy(self.model, teacher, train_eval_dataset, self.batch_size, self.device) if train_eval_dataset is not None else float("nan")
                val_acc = _distill_accuracy(self.model, teacher, val_dataset, self.batch_size, self.device) if val_dataset is not None else float("nan")
                self.model.train()  # _distill_accuracy left the model in eval mode
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)
                msg += f" | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"

            print(msg)

        return history

    # def fit(self, dataset):
    #     self.model.to(self.device)
    #     self.model.train()
    #     loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    #     for epoch in range(self.epochs):
    #         running_loss = 0.0
    #         for images, teacher_features, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False):
    #             images = images.to(self.device)
    #             teacher_features = teacher_features.to(self.device)
    #             labels = labels.to(self.device)

    #             self.optimizer.zero_grad()
    #             student_features = self.model(images)
    #             total_loss, mse, ce = self.criterion(student_features, teacher_features, labels)

    #             total_loss.backward()
    #             self.optimizer.step()

    #             running_loss += total_loss.item()

    #         print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {running_loss / len(loader):.4f}")

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
    
class RkdDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, student, teacher):
        """
        Args:
            student (torch.Tensor): Student embeddings of shape [Batch_Size, Student_Dim]
            teacher (torch.Tensor): Teacher embeddings of shape [Batch_Size, Teacher_Dim]
        """
        B = student.size(0)
        if B < 2:
            return torch.tensor(0.0, device=student.device, requires_grad=True)
        
        student = torch.mean(student, dim=[2, 3])
        teacher = torch.mean(teacher, dim=[2, 3])
        # Create a mask to exclude diagonal entries (self-distances)
        mask = ~torch.eye(B, dtype=torch.bool, device=student.device)
        
        # 1. Compute normalized pairwise distances for the Teacher (detached)
        with torch.no_grad():
            t_diff = teacher.unsqueeze(1) - teacher.unsqueeze(0)  # [B, B, Teacher_Dim]
            # Add 1e-8 inside sqrt to prevent NaN gradients at zero-distance diagonals
            t_dist = torch.sqrt(torch.sum(t_diff ** 2, dim=-1) + 1e-8)  # [B, B]
            t_dist_flat = t_dist[mask]
            
            mu_t = t_dist_flat.mean() + 1e-7
            psi_t = t_dist_flat / mu_t
            
        # 2. Compute normalized pairwise distances for the Student
        s_diff = student.unsqueeze(1) - student.unsqueeze(0)  # [B, B, Student_Dim]
        s_dist = torch.sqrt(torch.sum(s_diff ** 2, dim=-1) + 1e-8)  # [B, B]
        s_dist_flat = s_dist[mask]
        
        mu_s = s_dist_flat.mean() + 1e-7
        psi_s = s_dist_flat / mu_s
        
        loss = F.huber_loss(psi_s, psi_t, reduction='mean')
        return loss

class RkdAngleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, student, teacher):
        """
        Args:
            student (torch.Tensor): Student embeddings of shape [Batch_Size, Student_Dim]
            teacher (torch.Tensor): Teacher embeddings of shape [Batch_Size, Teacher_Dim]
        """
        B = student.size(0)
        if B < 3:
            return torch.tensor(0.0, device=student.device, requires_grad=True)
        
        student = torch.mean(student, dim=[2, 3])
        teacher = torch.mean(teacher, dim=[2, 3])

        # Create a 3D mask to filter valid triplets where (i != j) and (k != j) and (i != k)
        # Dimensions represent: [j (vertex), i (neighbor 1), k (neighbor 2)]
        idx = torch.arange(B, device=student.device)
        j_idx = idx.view(B, 1, 1)
        i_idx = idx.view(1, B, 1)
        k_idx = idx.view(1, 1, B)
        mask = (i_idx != j_idx) & (k_idx != j_idx) & (i_idx != k_idx)
        
        # 1. Compute angle potentials for the Teacher (detached)
        with torch.no_grad():
            t_diff = teacher.unsqueeze(1) - teacher.unsqueeze(0)  # [B, B, Teacher_Dim] (t_i - t_j)
            t_norm = torch.sqrt(torch.sum(t_diff ** 2, dim=-1, keepdim=True) + 1e-8)
            e_t = t_diff / t_norm  # Normalized directional vectors e_ij
            
            # Permute to make vertex 'j' the batch dimension: [B_j, B_i, Teacher_Dim]
            e_t = e_t.permute(1, 0, 2)
            # Batch Matrix Multiply to get dot products <e_ij, e_kj> for all i, k pairs at each j
            psi_t = torch.bmm(e_t, e_t.permute(0, 2, 1))  # [B, B, B]
            psi_t_flat = psi_t[mask]
            
        # 2. Compute angle potentials for the Student
        s_diff = student.unsqueeze(1) - student.unsqueeze(0)  # [B, B, Student_Dim]
        s_norm = torch.sqrt(torch.sum(s_diff ** 2, dim=-1, keepdim=True) + 1e-8)
        e_s = s_diff / s_norm
        
        e_s = e_s.permute(1, 0, 2)
        psi_s = torch.bmm(e_s, e_s.permute(0, 2, 1))  # [B, B, B]
        psi_s_flat = psi_s[mask]
        
        # 3. Huber Loss
        loss = F.huber_loss(psi_s_flat, psi_t_flat, reduction='mean')
        return loss

class ReletionalModelTrainer:
    def __init__(self, model, lr, epochs, batch_size=32, optimizer_cls=torch.optim.Adam, device=None, criterion=None):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    def fit(self, dataset, train_eval_dataset=None, val_dataset=None):
        self.model.to(self.device)
        self.model.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        rkd_distance = RkdDistanceLoss()
        rkd_angle = RkdAngleLoss()

        track_acc = train_eval_dataset is not None or val_dataset is not None
        history = {"total": [], "angle": [], "combined": []}
        if track_acc:
            history["train_acc"] = []
            history["val_acc"] = []

        for epoch in range(self.epochs):
            # Initialize trackers for all three losses
            running_loss = 0.0
            running_angle = 0.0
            running_combined = 0.0

            for images, teacher_features, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False):
                images = images.to(self.device)
                teacher_features = teacher_features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                student_features = self.model(images)

                loss_combined, loss_mse, loss_ce = self.criterion(student_features, teacher_features, labels)

                clf_features = student_features
                if clf_features.dim() == 2:
                    clf_features = clf_features.unsqueeze(-1).unsqueeze(-1)
                loss_rkd_dist = rkd_distance(clf_features, teacher_features)
                loss_rkd_angle = rkd_angle(clf_features, teacher_features)

                loss = loss_combined + loss_rkd_angle + loss_rkd_dist

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_angle += loss_rkd_angle.item()
                running_combined += loss_combined.item()

            # Calculate the averages over the entire epoch
            avg_total = running_loss / len(loader)
            avg_angle = running_angle / len(loader)
            avg_combined = running_combined / len(loader)

            history["total"].append(avg_total)
            history["angle"].append(avg_angle)
            history["combined"].append(avg_combined)

            msg = (
                f"Epoch {epoch + 1}/{self.epochs} -> "
                f"Total Loss: {avg_total:.4f} | "
                f"Angle (Feature): {avg_angle:.4f} | "
                f"Combined (Classification): {avg_combined:.4f}"
            )

            if track_acc:
                teacher = self.criterion.teacher
                train_acc = _distill_accuracy(self.model, teacher, train_eval_dataset, self.batch_size, self.device) if train_eval_dataset is not None else float("nan")
                val_acc = _distill_accuracy(self.model, teacher, val_dataset, self.batch_size, self.device) if val_dataset is not None else float("nan")
                self.model.train()  # _distill_accuracy left the model in eval mode
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)
                msg += f" | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"

            print(msg)

        return history

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