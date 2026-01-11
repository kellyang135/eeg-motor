# EEG Motor Imagery Classification 

Classify imagined movements (left hand, right hand, feet, tongue) from brain signals using the BCI Competition IV Dataset 2a.

## Project Structure

```
eeg-motor-imagery/
├── configs/              # Experiment configurations (YAML)
│   └── default.yaml      # Default hyperparameters
├── data/
│   ├── raw/              # Downloaded GDF files
│   └── processed/        # Preprocessed epochs
├── figures/              # Generated visualizations
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/              # Model outputs and metrics
├── scripts/
│   └── download_data.py  # Dataset downloader
├── src/
│   ├── preprocessing.py  # Signal processing pipeline
│   ├── features.py       # CSP, PSD feature extraction
│   ├── models.py         # LDA, SVM, EEGNet
│   └── visualization.py  # Plotting utilities
├── requirements.txt
└── README.md
```

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
python scripts/download_data.py

# Or download specific subjects
python scripts/download_data.py --subjects 1 2 3
```

### 3. Run Exploration Notebook

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Dataset

**BCI Competition IV Dataset 2a**
- 9 subjects
- 22 EEG channels (10-20 system)
- 250 Hz sampling rate
- 4 classes: left hand, right hand, feet, tongue
- ~288 trials per subject (144 train + 144 test)

## Methods

### Preprocessing
1. Bandpass filter (8-30 Hz) for mu/beta rhythms
2. Epoch extraction (0.5-4s post-cue)
3. Artifact rejection (>100 µV threshold)

### Features
- **CSP**: Common Spatial Patterns (6 components)
- **PSD**: Power spectral density in mu/beta bands
- **FBCSP**: Filter Bank CSP (multiple frequency bands)

### Models
| Model | Description | Expected Accuracy |
|-------|-------------|-------------------|
| LDA + CSP | Linear baseline | ~65-70% |
| SVM + CSP | RBF kernel | ~70-75% |
| EEGNet | Compact CNN | ~75-80% |

## Usage

```python
from src.preprocessing import preprocess_subject
from src.features import CSPFeatures
from src.models import create_lda_pipeline, evaluate_classifier

# Load and preprocess
epochs, labels = preprocess_subject('data/raw/A01T.gdf')
X = epochs.get_data()

# Extract CSP features
csp = CSPFeatures(n_components=6)
X_csp = csp.fit_transform(X, labels)

# Train and evaluate
clf = create_lda_pipeline()
results = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
print(f"Accuracy: {results['test_accuracy']:.2%}")
```

## Configuration

Experiments are configured via YAML files in `configs/`:

```yaml
preprocessing:
  l_freq: 8.0
  h_freq: 30.0
  tmin: 0.5
  tmax: 4.0

features:
  csp:
    n_components: 6
    reg: "ledoit_wolf"
```

## References

1. Tangermann et al. (2012). Review of the BCI Competition IV. Frontiers in Neuroscience.
2. Lawhern et al. (2018). EEGNet: A Compact CNN for EEG-based BCIs. Journal of Neural Engineering.
3. Blankertz et al. (2008). The BCI Competition 2003: Progress and perspectives in detection and discrimination of EEG single trials. IEEE TBME.

## License

MIT License - See LICENSE file for details.
