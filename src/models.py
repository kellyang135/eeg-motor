"""
Classification models 
Implements:
- Classical ML: LDA, SVM with CSP features
- Deep Learning: EEGNet (compact CNN for EEG)
"""

from typing import Optional
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Classical ML Models
# ==============================================================================

def create_lda_pipeline(shrinkage: str = 'auto') -> LinearDiscriminantAnalysis:
    """
    Create LDA classifier with automatic shrinkage.

    LDA is the most common baseline for MI-BCI with CSP features.
    """
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)


def create_svm_pipeline(
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale'
) -> SVC:
    """
    Create SVM classifier.

    RBF kernel often works well with CSP features.
    """
    return SVC(kernel=kernel, C=C, gamma=gamma, probability=True)


def create_rf_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = None
) -> RandomForestClassifier:
    """Create Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )


def evaluate_classifier(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5
) -> dict:
    """
    Train and evaluate a classifier.

    Returns
    -------
    results : dict
        Contains accuracy, kappa, confusion matrix, and CV scores
    """
    # Cross-validation on training data
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

    # Fit on full training set and evaluate on test
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_kappa': cohen_kappa_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'cv_scores': cv_scores
    }


# ==============================================================================
# Deep Learning: EEGNet
# ==============================================================================

class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.

    Reference: Lawhern et al., 2018
    https://arxiv.org/abs/1611.08024

    Parameters
    ----------
    n_channels : int
        Number of EEG channels
    n_times : int
        Number of time samples per epoch
    n_classes : int
        Number of output classes
    F1 : int
        Number of temporal filters
    D : int
        Depth multiplier (spatial filters per temporal filter)
    F2 : int
        Number of pointwise filters
    kernel_length : int
        Length of temporal convolution kernel
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 875,  # 3.5s at 250 Hz
        n_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Calculate the size after convolutions and pooling
        self._to_linear = self._get_conv_output_size()

        # Classification head
        self.fc = nn.Linear(self._to_linear, n_classes)

    def _get_conv_output_size(self) -> int:
        """Calculate output size after conv layers."""
        x = torch.zeros(1, 1, self.n_channels, self.n_times)
        x = self._forward_features(x)
        return x.numel()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG data, shape (batch, channels, times) or (batch, 1, channels, times)

        Returns
        -------
        out : torch.Tensor
            Class logits, shape (batch, n_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, channels, times)

        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


class EEGNetTrainer:
    """
    Training wrapper for EEGNet.

    Parameters
    ----------
    model : EEGNet
        The model to train
    device : str
        'cuda' or 'cpu'
    learning_rate : float
        Initial learning rate
    """

    def __init__(
        self,
        model: EEGNet,
        device: str = 'cpu',
        learning_rate: float = 1e-3
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, dataloader) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, dataloader) -> tuple[float, float]:
        """Evaluate on validation/test data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)

        return total_loss / total, correct / total

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> dict:
        """
        Full training loop with early stopping.

        Returns
        -------
        history : dict
            Training history
        """
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train: {train_acc:.3f} - Val: {val_acc:.3f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        outputs = self.model(X_tensor)
        _, predicted = outputs.max(1)
        return predicted.cpu().numpy()


def create_eeg_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
):
    """Create PyTorch DataLoaders for EEG data."""
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
