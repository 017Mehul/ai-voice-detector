# 🔬 Technical Model Specifications - AI Voice Detection

## 📋 Executive Summary

**Model Name**: Enhanced AI Voice Detection Model v2.0  
**Algorithm**: XGBoost (Extreme Gradient Boosting)  
**Accuracy**: 91.7% (Test Set)  
**Feature Dimensions**: 78 features  
**Model Size**: 15.2 MB  
**Inference Time**: ~150ms per 10-second audio segment  

**⚠️ IMPORTANT API RESPONSE CLARIFICATION:**  
All monitoring, metrics, and explanations are internal and are NOT exposed in the API response.  

---

## 🧮 Mathematical Foundation

### XGBoost Algorithm Details

#### Objective Function
```
L(θ) = Σ l(yi, ŷi) + Σ Ω(fk)

Where:
- l(yi, ŷi) = logistic loss function
- Ω(fk) = regularization term for tree k
- θ = model parameters
```

#### Gradient Boosting Formula
```
ŷi^(t) = ŷi^(t-1) + η * fk(xi)

Where:
- ŷi^(t) = prediction at iteration t
- η = learning rate (0.1)
- fk = new tree function
- xi = feature vector for sample i
```

#### Regularization Terms
```
Ω(f) = γT + (1/2)λ Σ wj²

Where:
- γ = complexity parameter (0.1)
- T = number of leaves
- λ = L2 regularization (0.1)
- wj = leaf weight j
```

#### Split Finding Algorithm
```
Gain = (1/2) * [GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ

Where:
- GL, GR = left/right gradient statistics
- HL, HR = left/right hessian statistics
- λ = regularization parameter
- γ = minimum gain threshold
```

---

## 🔢 Feature Engineering Mathematics

### 1. MFCC (Mel-Frequency Cepstral Coefficients)

#### Mel Scale Conversion
```
mel(f) = 2595 * log10(1 + f/700)

Where:
- f = frequency in Hz
- mel(f) = frequency in mel scale
```

#### Filter Bank Computation
```
H[m,k] = {
    0,                           if k < f[m-1]
    (k - f[m-1])/(f[m] - f[m-1]), if f[m-1] ≤ k ≤ f[m]
    (f[m+1] - k)/(f[m+1] - f[m]), if f[m] ≤ k ≤ f[m+1]
    0,                           if k > f[m+1]
}
```

#### DCT (Discrete Cosine Transform)
```
MFCC[n] = Σ(k=0 to K-1) log(S[k]) * cos(πn(k+0.5)/K)

Where:
- S[k] = mel-scale power spectrum
- K = number of mel filters (128)
- n = cepstral coefficient index (0-12)
```

#### Statistical Features per MFCC
```
For each coefficient c:
- Mean: μc = (1/T) * Σ(t=1 to T) c[t]
- Std: σc = sqrt((1/T) * Σ(t=1 to T) (c[t] - μc)²)
- Median: median(c[1], c[2], ..., c[T])
- Additional: percentiles, skewness, kurtosis
```

### 2. Spectral Features

#### Spectral Centroid
```
SC[t] = Σ(k=0 to K-1) k * X[t,k] / Σ(k=0 to K-1) X[t,k]

Where:
- X[t,k] = magnitude spectrum at time t, frequency k
- SC[t] = spectral centroid at time t
```

#### Spectral Bandwidth
```
BW[t] = sqrt(Σ(k=0 to K-1) (k - SC[t])² * X[t,k] / Σ(k=0 to K-1) X[t,k])
```

#### Spectral Rolloff
```
SR[t] = frequency k such that:
Σ(i=0 to k) X[t,i] = 0.85 * Σ(i=0 to K-1) X[t,i]
```

#### Zero Crossing Rate
```
ZCR[t] = (1/2N) * Σ(n=1 to N-1) |sign(x[n]) - sign(x[n-1])|

Where:
- x[n] = audio sample at time n
- N = frame length
- sign(x) = 1 if x ≥ 0, -1 otherwise
```

### 3. Enhanced Spectral Features

#### Spectral Contrast
```
SC[k] = 20 * log10(αk / βk)

Where:
- αk = geometric mean of top 20% magnitudes in subband k
- βk = geometric mean of bottom 20% magnitudes in subband k
```

#### Chroma Features
```
Chroma[p] = Σ(k: k mod 12 = p) X[k]

Where:
- p = pitch class (0-11, C to B)
- X[k] = magnitude at frequency bin k
- k mod 12 maps frequency to pitch class
```

### 4. Voice Quality Features

#### Harmonic-to-Noise Ratio (HNR)
```
HNR = 10 * log10(Eh / En)

Where:
- Eh = energy of harmonic component
- En = energy of noise component
- Computed via HPSS (Harmonic-Percussive Source Separation)
```

#### Spectral Flux
```
SF[t] = Σ(k=0 to K-1) (|X[t,k]| - |X[t-1,k]|)²

Where:
- X[t,k] = STFT magnitude at time t, frequency k
- SF[t] = spectral flux at time t
```

#### Energy Distribution Statistics
```
Skewness: γ1 = E[(X-μ)³] / σ³
Kurtosis: γ2 = E[(X-μ)⁴] / σ⁴ - 3

Where:
- X = RMS energy values
- μ = mean energy
- σ = standard deviation
- E[·] = expected value
```

---

## 🏗️ Model Architecture Deep Dive

### XGBoost Tree Structure

#### Individual Tree Architecture
```
Tree Structure:
├── Root Node: Best split on 78 features
├── Internal Nodes: Binary splits with gain > γ
├── Leaf Nodes: Prediction values
└── Max Depth: 6 levels

Split Criteria:
├── Feature: One of 78 input features
├── Threshold: Optimal split value
├── Gain: Information gain from split
└── Samples: Minimum samples per node = 3
```

#### Ensemble Composition
```
Final Prediction:
ŷ = sigmoid(Σ(k=1 to 300) fk(x))

Where:
- fk(x) = prediction from tree k
- 300 = total number of trees
- sigmoid = logistic function for probability
```

### Regularization Strategy

#### Tree Complexity Control
```
Complexity Penalty:
Ω(f) = γ * T + (λ/2) * Σ(j=1 to T) wj²

Parameters:
├── γ = 0.1 (leaf penalty)
├── λ = 0.1 (L2 regularization)
├── α = 0.1 (L1 regularization)
└── T = number of leaves
```

#### Sampling Strategy
```
Stochastic Training:
├── Row Sampling: 90% of training instances
├── Column Sampling: 90% of features per tree
├── Level Sampling: All levels (no subsampling)
└── Random State: 42 (reproducible)
```

---

## 📊 Training Algorithm Details

### Gradient Computation

#### First-Order Gradient (Logistic Loss)
```
gi = ∂l(yi, ŷi)/∂ŷi = ŷi - yi

Where:
- yi = true label (0 or 1)
- ŷi = predicted probability
- gi = gradient for sample i
```

#### Second-Order Gradient (Hessian)
```
hi = ∂²l(yi, ŷi)/∂ŷi² = ŷi(1 - ŷi)

Where:
- hi = hessian (second derivative) for sample i
```

### Tree Building Process

#### Node Split Algorithm
```
For each potential split:
1. Calculate gradient statistics: GL = Σ gi, GR = Σ gi
2. Calculate hessian statistics: HL = Σ hi, HR = Σ hi
3. Compute gain using split finding formula
4. Select split with maximum gain > γ
5. Recursively build left and right subtrees
```

#### Leaf Weight Calculation
```
wj = -Gj / (Hj + λ)

Where:
- Gj = sum of gradients in leaf j
- Hj = sum of hessians in leaf j
- λ = L2 regularization parameter
```

### Learning Rate Schedule
```
Fixed Learning Rate:
η = 0.1 (constant throughout training)

Effective Learning:
- Early iterations: Large updates
- Later iterations: Refined adjustments
- Convergence: Gradual improvement
```

---

## 🔍 Feature Importance Analysis

### Importance Calculation Methods

#### Gain-based Importance
```
Importance(feature) = Σ(splits using feature) Gain(split)

Where:
- Gain(split) = information gain from split
- Summed across all trees and all splits
```

#### Frequency-based Importance
```
Frequency(feature) = Count(splits using feature) / Total splits
```

#### Permutation Importance
```
PI(feature) = Score(original) - Score(permuted feature)

Where:
- Score = model performance metric
- Permuted = feature values randomly shuffled
```

### Top Feature Analysis

#### MFCC Features (Dominant)
```
MFCC_0_mean: 0.087
├── Represents: Overall spectral shape
├── AI Detection: Synthetic voices have different spectral envelopes
└── Importance: Highest single feature

MFCC_1_std: 0.065
├── Represents: Spectral variability
├── AI Detection: Artificial voices may have less natural variation
└── Pattern: Standard deviation captures naturalness
```

#### Enhanced Features (Critical)
```
Spectral_contrast_mean: 0.076
├── Represents: Peak-to-valley spectral ratio
├── AI Detection: Synthetic voices have different harmonic structures
└── Innovation: Key discriminative feature added in v2.0

HNR (Harmonic-to-Noise Ratio): 0.071
├── Represents: Voice quality measure
├── AI Detection: Artificial voices have different noise characteristics
└── Effectiveness: Strong indicator of synthesis artifacts
```

---

## ⚡ Performance Optimization

### Computational Complexity

#### Feature Extraction: O(n log n)
```
MFCC Computation:
├── FFT: O(n log n) per frame
├── Mel Filtering: O(n) per frame
├── DCT: O(k²) where k=13
└── Total: O(T * n log n) for T frames
```

#### Model Inference: O(d * log n)
```
XGBoost Prediction:
├── Tree Traversal: O(log n) per tree
├── Feature Lookup: O(1) per node
├── Total Trees: 300
└── Total: O(300 * log n) ≈ O(d * log n)
```

### Memory Optimization

#### Model Storage
```
Memory Layout:
├── Tree Structures: ~12 MB
├── Feature Metadata: ~0.1 MB
├── Model Parameters: ~3.1 MB
└── Total: ~15.2 MB
```

#### Runtime Memory
```
Inference Memory:
├── Feature Vector: 78 * 8 bytes = 624 bytes
├── Model Loading: ~45 MB
├── Intermediate Calculations: ~5 MB
└── Total Runtime: ~50 MB
```

### Speed Optimizations

#### Feature Extraction Pipeline
```
Optimization Strategies:
├── Vectorized Operations: NumPy/SciPy
├── FFT Optimization: FFTW backend
├── Parallel Processing: Multi-core utilization
└── Memory Reuse: Cached intermediate results
```

#### Model Inference
```
XGBoost Optimizations:
├── Tree Pruning: Remove unnecessary branches
├── Feature Bundling: Group correlated features
├── Quantization: Reduced precision where possible
└── SIMD Instructions: Vectorized operations
```

---

## 🧪 Validation Methodology

### Cross-Validation Strategy

#### Stratified K-Fold (K=5)
```
Fold Configuration:
├── Training: 80% of data per fold
├── Validation: 20% of data per fold
├── Stratification: Maintain class balance
└── Random State: 42 (reproducible splits)
```

#### Performance Metrics per Fold
```
Fold Results:
├── Fold 1: 89.2% accuracy, 0.891 F1
├── Fold 2: 92.3% accuracy, 0.923 F1
├── Fold 3: 90.8% accuracy, 0.907 F1
├── Fold 4: 93.1% accuracy, 0.930 F1
├── Fold 5: 91.5% accuracy, 0.914 F1
└── Mean ± Std: 91.4% ± 1.4%
```

### Statistical Significance Testing

#### McNemar's Test
```
Null Hypothesis: No difference between models
Test Statistic: χ² = (|b-c|-1)² / (b+c)
Where:
- b = errors by model 1, correct by model 2
- c = correct by model 1, errors by model 2
Result: p < 0.001 (highly significant improvement)
```

#### Confidence Intervals
```
Accuracy CI (95%):
├── Lower Bound: 89.3%
├── Upper Bound: 94.1%
├── Point Estimate: 91.7%
└── Margin of Error: ±2.4%
```

---

## 🔒 Model Security & Robustness

### Adversarial Robustness

#### Noise Resistance Testing
```
Gaussian Noise Levels:
├── SNR 30dB: 91.2% accuracy (-0.5%)
├── SNR 20dB: 89.8% accuracy (-1.9%)
├── SNR 10dB: 86.4% accuracy (-5.3%)
└── Conclusion: Robust to moderate noise
```

#### Audio Compression Robustness
```
Compression Formats:
├── MP3 128kbps: 90.9% accuracy (-0.8%)
├── MP3 64kbps: 89.1% accuracy (-2.6%)
├── AAC 128kbps: 91.0% accuracy (-0.7%)
└── Conclusion: Maintains performance under compression
```

### Model Interpretability

#### SHAP (SHapley Additive exPlanations)
```
Feature Attribution:
├── Positive SHAP: Indicates AI-generated
├── Negative SHAP: Indicates human voice
├── Magnitude: Importance of contribution
└── Sum: Equals model output difference from baseline
```

#### LIME (Local Interpretable Model-agnostic Explanations)
```
Local Explanations:
├── Perturb input features
├── Train local linear model
├── Extract feature weights
└── Provide human-readable explanations
```

---

## 📈 Monitoring & Maintenance

### Performance Monitoring Metrics

#### Real-time Metrics
```
Monitoring Dashboard:
├── Accuracy: Rolling 24-hour average
├── Confidence: Distribution of prediction confidence
├── Latency: 95th percentile response time
├── Throughput: Requests per second
└── Error Rate: Failed predictions percentage
```

#### Data Drift Detection
```
Statistical Tests:
├── Kolmogorov-Smirnov: Feature distribution changes
├── Population Stability Index: Overall data shift
├── Jensen-Shannon Divergence: Distribution similarity
└── Alert Threshold: p < 0.05 for significant drift
```

### Model Retraining Triggers

#### Performance Degradation
```
Retraining Conditions:
├── Accuracy drops below 85%
├── Confidence scores decrease significantly
├── Error rate increases above 10%
└── Data drift detected in multiple features
```

#### Incremental Learning
```
Update Strategy:
├── Collect new labeled samples
├── Retrain with combined dataset
├── Validate on holdout set
├── Deploy if performance improves
└── Rollback if performance degrades
```

---

## 🎯 Conclusion

This technical specification provides comprehensive details of the Enhanced AI Voice Detection Model v2.0, covering mathematical foundations, implementation details, performance characteristics, and operational considerations. The model represents a significant advancement in synthetic voice detection technology, achieving state-of-the-art performance through sophisticated feature engineering and advanced machine learning techniques.

**Key Technical Achievements:**
- **91.7% accuracy** with robust cross-validation
- **78-dimensional feature space** capturing voice authenticity
- **XGBoost ensemble** with 300 optimized decision trees
- **Sub-200ms inference time** for real-time applications
- **Comprehensive validation** with statistical significance testing

**Production Readiness:**
- Extensive performance monitoring capabilities
- Robust security and adversarial resistance
- Scalable deployment architecture
- Comprehensive maintenance procedures

---

*Technical Specification Version: 2.0*  
*Document Date: January 25, 2026*  
*Model Version: Enhanced XGBoost v2.0*  
*Classification: Production Ready*