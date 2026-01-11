"""
Visualization utilities
- Topographic maps
- CSP pattern visualization
- Spectrograms
- Classification results
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.viz import plot_topomap


# Class names for labeling
CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']


def set_style():
    """Set consistent plotting style for publication."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })


def plot_raw_segment(
    raw: mne.io.Raw,
    duration: float = 10.0,
    start: float = 0.0,
    n_channels: int = 10,
    title: str = "Raw EEG Segment",
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot a segment of raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    duration : float
        Duration to plot (seconds)
    start : float
        Start time (seconds)
    n_channels : int
        Number of channels to show
    """
    fig = raw.plot(
        duration=duration,
        start=start,
        n_channels=n_channels,
        title=title,
        scalings='auto',
        show=False
    )
    return fig


def plot_epoch_average(
    epochs: mne.Epochs,
    class_name: str,
    channels: Optional[list] = None,
    figsize: tuple = (12, 5)
) -> plt.Figure:
    """Plot average evoked response for a class."""
    evoked = epochs[class_name].average()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Time series
    evoked.plot(axes=axes[0], show=False, spatial_colors=True)
    axes[0].set_title(f'{class_name} - Average ERP')

    # Topomap at peak
    times = [0.5, 1.0, 2.0, 3.0]  # Sample times
    evoked.plot_topomap(times=times, axes=axes[1:], show=False)

    plt.tight_layout()
    return fig


def plot_psd_by_class(
    epochs: mne.Epochs,
    fmin: float = 4.0,
    fmax: float = 40.0,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot power spectral density for each class.

    Shows mu (8-12 Hz) and beta (13-30 Hz) bands.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, class_name in enumerate(CLASS_NAMES):
        try:
            class_epochs = epochs[class_name.lower().replace(' ', '_')]
            spectrum = class_epochs.compute_psd(fmin=fmin, fmax=fmax)
            spectrum.plot(axes=axes[idx], show=False, average=True)
            axes[idx].set_title(class_name)
            axes[idx].axvspan(8, 12, alpha=0.2, color='blue', label='mu')
            axes[idx].axvspan(13, 30, alpha=0.2, color='red', label='beta')
        except KeyError:
            axes[idx].text(0.5, 0.5, f'No {class_name} data',
                          ha='center', va='center', transform=axes[idx].transAxes)

    plt.tight_layout()
    return fig


def plot_csp_patterns(
    csp,
    info: mne.Info,
    n_components: int = 4,
    figsize: tuple = (14, 4)
) -> plt.Figure:
    """
    Plot CSP spatial patterns as topographic maps.

    Parameters
    ----------
    csp : CSPFeatures or mne.decoding.CSP
        Fitted CSP object
    info : mne.Info
        MNE Info object with channel locations
    n_components : int
        Number of components to plot
    """
    patterns = csp.patterns_[:n_components]

    fig, axes = plt.subplots(1, n_components, figsize=figsize)
    if n_components == 1:
        axes = [axes]

    for idx, (pattern, ax) in enumerate(zip(patterns, axes)):
        plot_topomap(pattern, info, axes=ax, show=False)
        ax.set_title(f'CSP {idx + 1}')

    fig.suptitle('CSP Spatial Patterns', y=1.02)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    figsize: tuple = (8, 6),
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Plot confusion matrix with nice formatting.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    normalize : bool
        If True, show percentages instead of counts
    """
    from sklearn.metrics import confusion_matrix as compute_cm

    cm = compute_cm(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%' if normalize else 'd',
        cmap=cmap,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    return fig


def plot_training_history(
    history: dict,
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Plot training curves for deep learning model.

    Parameters
    ----------
    history : dict
        Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_subject_comparison(
    results: dict,
    metric: str = 'accuracy',
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Compare performance across subjects.

    Parameters
    ----------
    results : dict
        Dictionary mapping subject ID to results dict
    metric : str
        Metric to plot ('accuracy' or 'kappa')
    """
    subjects = list(results.keys())
    values = [results[s].get(f'test_{metric}', 0) for s in subjects]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(subjects, values, color='steelblue', edgecolor='black')
    ax.axhline(np.mean(values), color='red', linestyle='--',
               label=f'Mean: {np.mean(values):.2%}')

    ax.set_xlabel('Subject')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} by Subject')
    ax.set_ylim(0, 1)
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_model_comparison(
    results: dict,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Compare performance across models.

    Parameters
    ----------
    results : dict
        Dictionary mapping model name to (mean, std) accuracy
    """
    models = list(results.keys())
    means = [results[m][0] for m in models]
    stds = [results[m][1] for m in models]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color='steelblue', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_ylim(0, 1)

    # Chance level for 4 classes
    ax.axhline(0.25, color='gray', linestyle=':', label='Chance (25%)')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_time_frequency(
    epochs: mne.Epochs,
    picks: str = 'C3',
    fmin: float = 4.0,
    fmax: float = 40.0,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot time-frequency decomposition for each class.

    Shows event-related spectral perturbation (ERSP).
    """
    freqs = np.arange(fmin, fmax, 1)
    n_cycles = freqs / 2

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, class_name in enumerate(CLASS_NAMES):
        try:
            class_epochs = epochs[class_name.lower().replace(' ', '_')]
            power = mne.time_frequency.tfr_morlet(
                class_epochs, freqs=freqs, n_cycles=n_cycles,
                picks=picks, return_itc=False, average=True
            )
            power.plot([0], axes=axes[idx], show=False,
                      title=f'{class_name} - {picks}')
        except (KeyError, IndexError):
            axes[idx].text(0.5, 0.5, f'No {class_name} data',
                          ha='center', va='center')

    plt.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    filename: str,
    figures_dir: str = 'figures',
    dpi: int = 300,
    formats: list = ['png', 'pdf']
):
    """Save figure in multiple formats."""
    from pathlib import Path

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = figures_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")
