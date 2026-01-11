"""
Preprocessing pipeline
- Loading GDF files from BCI Competition IV Dataset 2a
- Bandpass filtering (8-30 Hz for mu/beta rhythms)
- Epoching around motor imagery cues
- Artifact rejection
"""

from pathlib import Path
from typing import Optional
import numpy as np
import mne
from mne.io import read_raw_gdf


# Channel mapping for BCI Competition IV Dataset 2a
# 22 EEG channels in 10-20 system
CHANNEL_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

# Event codes in the GDF files
EVENT_CODES = {
    769: 'left_hand',   # Class 1
    770: 'right_hand',  # Class 2
    771: 'feet',        # Class 3
    772: 'tongue',      # Class 4
    783: 'unknown',     # Cue onset (evaluation data)
}

# Mapping to integer labels
CLASS_LABELS = {
    'left_hand': 0,
    'right_hand': 1,
    'feet': 2,
    'tongue': 3,
}


def load_raw_gdf(
    filepath: str | Path,
    preload: bool = True
) -> mne.io.Raw:
    """
    Load a GDF file from BCI Competition IV Dataset 2a.

    Parameters
    ----------
    filepath : str or Path
        Path to the GDF file (e.g., 'A01T.gdf')
    preload : bool
        Whether to preload data into memory

    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data with proper channel names and montage
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GDF file not found: {filepath}")

    # Load raw data
    raw = read_raw_gdf(str(filepath), preload=preload, verbose='WARNING')

    # Select only EEG channels (first 22)
    raw.pick_channels(raw.ch_names[:22])

    # Rename channels to standard 10-20 names
    rename_dict = {old: new for old, new in zip(raw.ch_names, CHANNEL_NAMES)}
    raw.rename_channels(rename_dict)

    # Set channel types
    raw.set_channel_types({ch: 'eeg' for ch in CHANNEL_NAMES})

    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')

    return raw


def apply_bandpass_filter(
    raw: mne.io.Raw,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    method: str = 'iir',
    copy: bool = True
) -> mne.io.Raw:
    """
    Apply bandpass filter to capture mu (8-12 Hz) and beta (13-30 Hz) rhythms.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low frequency cutoff (Hz)
    h_freq : float
        High frequency cutoff (Hz)
    method : str
        Filter method ('fir' or 'iir')
    copy : bool
        Whether to return a copy

    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered raw data
    """
    if copy:
        raw = raw.copy()

    raw.filter(l_freq=l_freq, h_freq=h_freq, method=method, verbose='WARNING')

    return raw


def extract_events(raw: mne.io.Raw) -> tuple[np.ndarray, dict]:
    """
    Extract motor imagery events from the raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with annotations

    Returns
    -------
    events : np.ndarray
        MNE events array (n_events, 3)
    event_id : dict
        Mapping from event names to codes
    """
    events, _ = mne.events_from_annotations(raw, verbose='WARNING')

    # Filter to only motor imagery events (769-772)
    mi_event_codes = [769, 770, 771, 772]
    mask = np.isin(events[:, 2], mi_event_codes)
    events = events[mask]

    # Create event_id dict
    event_id = {
        'left_hand': 769,
        'right_hand': 770,
        'feet': 771,
        'tongue': 772,
    }

    return events, event_id


def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    tmin: float = 0.5,
    tmax: float = 4.0,
    baseline: Optional[tuple] = None,
    reject: Optional[dict] = None,
    preload: bool = True
) -> mne.Epochs:
    """
    Create epochs around motor imagery events.

    Parameters
    ----------
    raw : mne.io.Raw
        Filtered raw EEG data
    events : np.ndarray
        Events array from extract_events()
    event_id : dict
        Event ID mapping
    tmin : float
        Start time relative to event (seconds)
    tmax : float
        End time relative to event (seconds)
    baseline : tuple or None
        Baseline correction interval, None for no correction
    reject : dict or None
        Rejection thresholds (e.g., {'eeg': 100e-6})
    preload : bool
        Whether to preload epoch data

    Returns
    -------
    epochs : mne.Epochs
        Epoched EEG data
    """
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=preload,
        verbose='WARNING'
    )

    return epochs


def preprocess_subject(
    gdf_path: str | Path,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = 0.5,
    tmax: float = 4.0,
    reject_threshold: float = 100e-6
) -> tuple[mne.Epochs, np.ndarray]:
    """
    Full preprocessing pipeline for a single subject.

    Parameters
    ----------
    gdf_path : str or Path
        Path to subject's GDF file
    l_freq, h_freq : float
        Bandpass filter frequencies
    tmin, tmax : float
        Epoch boundaries relative to cue
    reject_threshold : float
        Amplitude threshold for artifact rejection (V)

    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epochs
    labels : np.ndarray
        Integer class labels (0-3)
    """
    # Load and filter
    raw = load_raw_gdf(gdf_path)
    raw = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)

    # Extract events and create epochs
    events, event_id = extract_events(raw)

    reject = {'eeg': reject_threshold} if reject_threshold else None
    epochs = create_epochs(
        raw, events, event_id,
        tmin=tmin, tmax=tmax,
        reject=reject
    )

    # Get labels as integers
    labels = np.array([CLASS_LABELS[epochs.events[i, 2]]
                       for i in range(len(epochs))
                       if epochs.events[i, 2] in CLASS_LABELS.values()])

    # Actually use event codes to get labels
    event_to_label = {v: CLASS_LABELS[k] for k, v in event_id.items()}
    labels = np.array([event_to_label[e] for e in epochs.events[:, 2]])

    return epochs, labels


def load_all_subjects(
    data_dir: str | Path,
    session: str = 'T',  # 'T' for training, 'E' for evaluation
    subjects: Optional[list[int]] = None,
    **preprocess_kwargs
) -> tuple[list[mne.Epochs], list[np.ndarray]]:
    """
    Load and preprocess data for multiple subjects.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing GDF files
    session : str
        'T' for training session, 'E' for evaluation
    subjects : list of int or None
        Subject numbers (1-9), None for all
    **preprocess_kwargs
        Arguments passed to preprocess_subject()

    Returns
    -------
    all_epochs : list of mne.Epochs
        Epochs for each subject
    all_labels : list of np.ndarray
        Labels for each subject
    """
    data_dir = Path(data_dir)
    subjects = subjects or list(range(1, 10))

    all_epochs = []
    all_labels = []

    for subj in subjects:
        gdf_path = data_dir / f"A0{subj}{session}.gdf"
        if not gdf_path.exists():
            print(f"Warning: {gdf_path} not found, skipping subject {subj}")
            continue

        print(f"Processing subject {subj}...")
        epochs, labels = preprocess_subject(gdf_path, **preprocess_kwargs)
        all_epochs.append(epochs)
        all_labels.append(labels)
        print(f"  {len(epochs)} epochs, {len(np.unique(labels))} classes")

    return all_epochs, all_labels
