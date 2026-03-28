# Voice Detector Project

## Quick Start
```bash
# Run demo
python demo.py

# Train model
python train_model.py

# Validate performance
python validate.py

# Run tests
python run_tests.py
```

## Project Structure
```
voice_detector/          # Main package
├── api/                 # REST API
├── audio/               # Audio processing
├── features/            # Feature extraction
├── models/              # ML models
├── services/            # Business logic
└── utils/               # Utilities

dataset/                 # Training data
├── Real/Real/           # Human voices
└── Fake/Fake/           # Synthetic voices

models/production/       # Trained model
docs/                    # Documentation
tests/                   # Test suite
```

## Model Info
- Algorithm: XGBoost
- Features: 78 acoustic features
- Accuracy: 95.9%
- Languages: Tamil, English, Hindi, Malayalam, Telugu

## Development
- Python 3.9+
- See requirements.txt for dependencies
- Use pytest for testing
- Docker support available