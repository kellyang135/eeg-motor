"""
Feature extraction
Implements:
- Common Spatial Patterns (CSP) - the gold standard for MI-BCI
- Power Spectral Density (PSD) features
- Time-frequency features
"""

from typing import Optional
import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
import mne
from mne.decoding import CSP


class CSPFeatures(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns feature extractor.

    CSP finds spatial filters that maximize variance for one class
    while minimizing it for another. For multi-class problems,
    we use One-vs-Rest CSP.

    Parameters
    ----------
    n_components : int
        Number of CSP components per class pair
    reg : str or float
        Regularization method ('ledoit_wolf', 'oas', 'shrunk', or float)
    log : bool
        Whether to log-transform the CSP features
    transform_into : str
        'average_power' for var(X_csp), 'csp_space' for raw projection
    """

    def __init__(
        self,
        n_components: int = 6,
        reg: str = 'ledoit_wolf',
        log: bool = True,
        transform_into: str = 'average_power'
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.transform_into = transform_into
        self.csp_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit CSP on training data.

        Parameters
        ----------
        X : np.ndarray
            EEG data, shape (n_epochs, n_channels, n_times)
        y : np.ndarray
            Class labels

        Returns
        -------
        self
        """
        self.csp_ = CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
            transform_into=self.transform_into,
            norm_trace=True
        )
        self.csp_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted CSP.

        Parameters
        ----------
        X : np.ndarray
            EEG data, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        X_csp : np.ndarray
            CSP features, shape (n_epochs, n_components)
        """
        if self.csp_ is None:
            raise ValueError("CSP not fitted. Call fit() first.")
        return self.csp_.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    @property
    def patterns_(self) -> np.ndarray:
        """Get CSP patterns for visualization."""
        if self.csp_ is None:
            raise ValueError("CSP not fitted.")
        return self.csp_.patterns_

    @property
    def filters_(self) -> np.ndarray:
        """Get CSP filters."""
        if self.csp_ is None:
            raise ValueError("CSP not fitted.")
        return self.csp_.filters_


class PSDFeatures(BaseEstimator, TransformerMixin):
    """
    Power Spectral Density feature extractor.

    Computes band power in specified frequency bands.

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz)
    fmin : float
        Minimum frequency for PSD computation
    fmax : float
        Maximum frequency for PSD computation
    n_fft : int
        FFT length
    bands : dict or None
        Frequency bands for averaging, e.g. {'mu': (8, 12), 'beta': (13, 30)}
        If None, returns raw PSD
    """

    def __init__(
        self,
        sfreq: float = 250.0,
        fmin: float = 8.0,
        fmax: float = 30.0,
        n_fft: int = 256,
        bands: Optional[dict] = None
    ):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.n_fft = n_fft
        self.bands = bands or {'mu': (8, 12), 'beta': (13, 30)}

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """No fitting required for PSD."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute PSD features.

        Parameters
        ----------
        X : np.ndarray
            EEG data, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        psd_features : np.ndarray
            PSD features, shape (n_epochs, n_channels * n_bands)
        """
        n_epochs, n_channels, n_times = X.shape
        n_bands = len(self.bands)

        features = np.zeros((n_epochs, n_channels * n_bands))

        for i in range(n_epochs):
            for ch in range(n_channels):
                # Compute PSD using Welch's method
                freqs, psd = signal.welch(
                    X[i, ch, :],
                    fs=self.sfreq,
                    nperseg=min(self.n_fft, n_times),
                    noverlap=self.n_fft // 2
                )

                # Extract band powers
                for b, (band_name, (fmin, fmax)) in enumerate(self.bands.items()):
                    mask = (freqs >= fmin) & (freqs <= fmax)
                    band_power = np.mean(psd[mask])
                    features[i, ch * n_bands + b] = np.log(band_power + 1e-10)

        return features


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Filter Bank Common Spatial Patterns (FBCSP).

    Applies CSP to multiple frequency bands and selects the best features.

    Parameters
    ----------
    bands : list of tuples
        Frequency bands, e.g. [(4, 8), (8, 12), (12, 16), ...]
    n_components : int
        CSP components per band
    sfreq : float
        Sampling frequency
    """

    def __init__(
        self,
        bands: Optional[list] = None,
        n_components: int = 4,
        sfreq: float = 250.0
    ):
        # Default: 4 Hz wide bands from 4-40 Hz
        self.bands = bands or [(4*i, 4*(i+1)) for i in range(1, 10)]
        self.n_components = n_components
        self.sfreq = sfreq
        self.csps_ = None
        self.filters_ = None

    def _bandpass_filter(
        self,
        X: np.ndarray,
        low: float,
        high: float
    ) -> np.ndarray:
        """Apply bandpass filter to data."""
        nyq = self.sfreq / 2
        low_norm = low / nyq
        high_norm = min(high / nyq, 0.99)

        b, a = signal.butter(5, [low_norm, high_norm], btype='band')
        return signal.filtfilt(b, a, X, axis=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit CSP for each frequency band."""
        self.csps_ = []
        self.filters_ = []

        for low, high in self.bands:
            # Filter data
            X_filt = self._bandpass_filter(X, low, high)

            # Fit CSP
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True)
            csp.fit(X_filt, y)
            self.csps_.append(csp)
            self.filters_.append((low, high))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using all filter bank CSPs."""
        if self.csps_ is None:
            raise ValueError("FBCSP not fitted.")

        features = []
        for (low, high), csp in zip(self.filters_, self.csps_):
            X_filt = self._bandpass_filter(X, low, high)
            features.append(csp.transform(X_filt))

        return np.hstack(features)


def extract_epochs_data(
    epochs: mne.Epochs,
    picks: Optional[str] = 'eeg'
) -> np.ndarray:
    """
    Extract numpy array from MNE Epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object
    picks : str or None
        Channel types to include

    Returns
    -------
    X : np.ndarray
        Data array, shape (n_epochs, n_channels, n_times)
    """
    return epochs.get_data(picks=picks)


def compute_csp_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 6
) -> tuple[np.ndarray, np.ndarray, CSPFeatures]:
    """
    Convenience function to compute CSP features for train/test data.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and test EEG data
    y_train : np.ndarray
        Training labels
    n_components : int
        Number of CSP components

    Returns
    -------
    X_train_csp, X_test_csp : np.ndarray
        CSP-transformed features
    csp : CSPFeatures
        Fitted CSP object (for visualization)
    """
    csp = CSPFeatures(n_components=n_components)
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

    return X_train_csp, X_test_csp, csp
