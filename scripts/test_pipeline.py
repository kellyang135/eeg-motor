#!/usr/bin/env python3
"""
Uses MOABB to load BCI Competition IV Dataset 2a and tests
1. Data loading
2. Preprocessing
3. CSP feature extraction
4. Classification (LDA, SVM)
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP

print("=" * 60)
print("EEG Motor Imagery Classification - Pipeline Test")
print("=" * 60)

# 1. Load dataset
print("\n[1/4] Loading BCI Competition IV Dataset 2a...")
dataset = BNCI2014_001()
paradigm = MotorImagery(
    n_classes=4,
    fmin=8, fmax=30,  # Bandpass filter
    tmin=0.5, tmax=4.0,  # Epoch window
    resample=250
)

# Load just subject 1 for testing
subjects = [1]
X, y, metadata = paradigm.get_data(dataset, subjects=subjects)

print(f"   Data shape: {X.shape} (trials, channels, samples)")
print(f"   Labels: {np.unique(y)}")
print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# 2. Test CSP
print("\n[2/4] Testing CSP feature extraction...")
csp = CSP(n_components=6, reg='ledoit_wolf', log=True, norm_trace=True)
X_csp = csp.fit_transform(X, y)
print(f"   CSP output shape: {X_csp.shape} (trials, components)")

# 3. Test LDA classifier
print("\n[3/4] Testing CSP + LDA pipeline...")
pipe_lda = Pipeline([
    ('csp', CSP(n_components=6, reg='ledoit_wolf', log=True)),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
])

scores_lda = cross_val_score(pipe_lda, X, y, cv=5, scoring='accuracy')
print(f"   5-fold CV accuracy: {scores_lda.mean():.1%} (+/- {scores_lda.std():.1%})")

# 4. Test SVM classifier
print("\n[4/4] Testing CSP + SVM pipeline...")
pipe_svm = Pipeline([
    ('csp', CSP(n_components=6, reg='ledoit_wolf', log=True)),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

scores_svm = cross_val_score(pipe_svm, X, y, cv=5, scoring='accuracy')
print(f"   5-fold CV accuracy: {scores_svm.mean():.1%} (+/- {scores_svm.std():.1%})")

# Summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Subject 1 - 4-class Motor Imagery Classification")
print(f"  CSP + LDA: {scores_lda.mean():.1%}")
print(f"  CSP + SVM: {scores_svm.mean():.1%}")
print(f"  Chance:    25.0%")
print("=" * 60)

if scores_lda.mean() > 0.35 and scores_svm.mean() > 0.35:
    print("\n✓ Pipeline test passed. Classification well above chance.")
else:
    print("\n✗ Pipeline test may have issues - accuracy near chance level.")
