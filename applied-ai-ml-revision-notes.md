---
tags: [htb-academy, ai, machine-learning, revision, applied]
module: Applying AI/ML (Spam, Network Anomaly, Malware Classification)
---

# Applied AI/ML: Revision Notes

> **Legend:** Unmarked = from your module text. **(Extra)** = my own added explanation or example. **⚠ Verify** = unclear or inconsistent in the source.
> **Format note:** This module is hands-on and code-heavy (builds 3 models end-to-end). Section 3 is organized by **workflow stage** (environment → data → model 1 → model 2 → model 3) rather than by isolated commands, since that's how the pipeline actually runs.
> **No flags or assessment answers included.** Section 6 gives approach only.

---

## 1. Main Concepts: What, Why, How

### 1.1 The Three Models Built in This Module

| Model | Task | Algorithm | Input |
|---|---|---|---|
| Spam Classifier | SMS spam vs. ham (binary) | Multinomial Naive Bayes | Text (SMS messages) |
| Network Anomaly Detector | Normal vs. attack type (multi-class) | Random Forest | Tabular network logs |
| Malware Classifier | Malware family (multi-class) | CNN (ResNet50, transfer learning) | Images (byteplots of binaries) |

- **Why this progression:** Moves from simplest (text + Naive Bayes) → tabular (Random Forest) → most complex (images + deep learning/CNN), covering the end-to-end AI workflow: explore data → preprocess → train → evaluate → save/deploy.

### 1.2 The End-to-End ML Workflow (applies to all three models)

1. **Set up environment** (Miniconda, packages, Jupyter).
2. **Load the dataset** into a structure you can inspect (pandas DataFrame, image folders).
3. **Explore** it (`head()`, `info()`, `isnull().sum()`, class distribution plots).
4. **Clean/preprocess** (handle invalid/missing values, standardize categories).
5. **Transform** (encode categoricals, scale/transform skewed numerics, text preprocessing, image resizing/normalization).
6. **Extract features** (vectorize text, select numeric columns, apply image transforms).
7. **Split** into train/validation/test.
8. **Train** the model (optionally with hyperparameter tuning).
9. **Evaluate** (accuracy, precision, recall, F1, confusion matrix).
10. **Save** the model (`joblib` for scikit-learn, `torch.jit.script` for PyTorch) for reuse/deployment.

### 1.3 What Makes a "Good" Dataset

| Attribute | Why it matters | Example from source |
|---|---|---|
| Relevance | Irrelevant data adds noise | Social media text is more relevant to sentiment analysis than stock prices |
| Completeness | Missing data biases models | Imputation helps, but starting complete is better |
| Consistency | Inconsistent formats break preprocessing | Uniform date formats (`YYYY-MM-DD`) |
| Quality | Errors from collection/entry/transmission reduce reliability | Validation/verification processes |
| Representativeness | A biased dataset creates a biased model | Facial recognition needs diverse faces |
| Balance | Imbalanced classes bias predictions toward the majority | Oversampling/undersampling/synthetic data |
| Size | Too small = not enough signal; too large = computationally expensive | — |

- **Why data quality matters overall:** Affects model accuracy, generalization (avoiding overfitting), efficiency (training time/compute), and reliability of downstream decisions.

### 1.4 Data Preprocessing Concepts

- **What:** Transforming raw data into a format algorithms can use effectively.
- **Four key techniques (from source):**
  - **Data Cleaning:** missing values, duplicates, noisy data.
  - **Data Transformation:** normalizing, encoding, scaling, reducing.
  - **Data Integration:** merging/aggregating multiple sources.
  - **Data Formatting:** converting types, reshaping structures.
- **Why:** Improves accuracy, efficiency, and robustness by addressing inconsistencies, missing values, outliers, noise, and feature scaling.
- **Two strategies for invalid data:**
  - **Dropping:** simplest; preferred when data accuracy matters most and losing some rows doesn't hurt the analysis. Not always feasible (small datasets, large proportion of bad data).
  - **Imputing:** replacing missing/invalid values with estimates, to retain as much data as possible.
    - Standardize invalid markers (`MISSING_IP`, `?`, etc.) to `NaN` first, for uniform downstream handling.
    - Simple imputation: median/mean for numeric, most-frequent for categorical (`SimpleImputer`).
    - Advanced imputation: `KNNImputer`/`IterativeImputer`, which consider relationships between features.
    - Apply **domain knowledge** afterward (e.g. default `source_ip` to `0.0.0.0`, clip ports to 0–65535, map invalid protocols to the mode).
    - Always do a **final verification** of distributions and categorical validity after imputing.

### 1.5 Data Transformation Concepts

- **Encoding categorical features (why):** ML algorithms need numeric input.
  - **OneHotEncoder:** binary indicator column per category; avoids implying false ordinal relationships; increases feature count with high-cardinality categories.
  - **LabelEncoder:** integer codes; can imply unintended order.
  - **HashingEncoder / frequency-based:** for high-cardinality features, to control feature space size.
- **Handling skewed data:** Uneven distributions (most values clustered, a few extreme) hurt models sensitive to outliers or that assume normal-like distributions.
  - **Log transform** (`log1p`, adds 1 before taking the log so it's defined at/near zero) compresses large values more than small ones, balancing the distribution without losing information.
- **Data Splitting:**

| Set | Purpose | Typical size |
|---|---|---|
| Training | Fit the model | ~60–80% |
| Validation | Tune hyperparameters, model selection | ~10–20% |
| Test | Final, untouched evaluation | ~10–20% |

  - **Two-step split pattern:** First split train/test (e.g. 80/20), then split the 80% training portion again (e.g. 75/25) to carve out a validation set, so the overall proportions work out to the targets (e.g. 0.8 × 0.25 = 0.2 → 60/20/20 overall).

### 1.6 Evaluation Metrics (recap + context)

> Covered more deeply in the Fundamentals of AI module; this section adds the **practical framing**.

| Metric | Formula | What it tells you | Risk if used alone |
|---|---|---|---|
| **Accuracy** | (TP+TN) / all | Overall correctness | Misleading under class imbalance — a model that always predicts the majority class can still score high |
| **Precision** | TP / (TP+FP) | Quality of positive predictions (how often "positive" calls are right) | High precision alone doesn't mean the model catches all positives |
| **Recall** | TP / (TP+FN) | Completeness of positive detection | High recall alone can mean many false positives (noisy) |
| **F1-score** | 2·(P·R)/(P+R) | Balance of precision and recall | Still just one number; check it's not hiding per-class weaknesses |

- **Other measures mentioned:** Specificity (how well negatives are identified), AUC (ROC curve area, discriminative capability across thresholds), Matthews Correlation Coefficient (good for highly imbalanced data), Confusion Matrix (full breakdown of predictions vs. truth).
- **Contextualizing metrics — questions to ask:**
  - Are metrics consistent across data segments?
  - Does the dataset reflect real-world conditions (including class imbalance)?
  - Are costs of false positives vs. false negatives accounted for?
- **Trade-offs by context (source examples):**
  - **Threat detection:** favor **recall** — missing a real threat is worse than a false alarm.
  - **Resource-constrained environments:** favor **precision** — reduce wasted effort chasing false alarms.

### 1.7 Naive Bayes for Spam Detection (theory)

- **What:** Applies Bayes' theorem with a "naive" independence assumption between features, given the class.
- **Bayes' theorem:** `P(A|B) = (P(B|A) * P(A)) / P(B)`
  - In spam terms: `P(Spam|Features) = (P(Features|Spam) * P(Spam)) / P(Features)`.
- **Naive simplification:** `P(Features|Spam) = P(feature1|Spam) * P(feature2|Spam) * ...` (features treated as conditionally independent).
- **Decision rule:** Compute the posterior for each class; pick the class with the **higher posterior probability**.
- **Worked example from source:**
  - Priors: P(Spam)=0.3, P(Not Spam)=0.7.
  - Likelihoods: P(F1|Spam)=0.4, P(F2|Spam)=0.5, P(F1|¬Spam)=0.2, P(F2|¬Spam)=0.3.
  - `P(F1,F2|Spam) = 0.4×0.5 = 0.2`; `P(F1,F2|¬Spam) = 0.2×0.3 = 0.06`.
  - `P(F1,F2) = 0.2×0.3 + 0.06×0.7 = 0.102` (law of total probability).
  - `P(Spam|F1,F2) = (0.2×0.3)/0.102 ≈ 0.588`; `P(¬Spam|F1,F2) ≈ 0.412` → classified **Spam** (higher posterior).

### 1.8 Text Preprocessing Pipeline (for spam/NLP)

Order used in the source, each step building on the last:

1. **Lowercasing** — treats "Free" and "free" as the same token.
2. **Removing punctuation/numbers** (but keep context-bearing symbols like `$` and `!`) — reduces noise while preserving spam-relevant signals.
3. **Tokenization** — splits text into words (`word_tokenize`), needed before stop-word removal/stemming.
4. **Stop word removal** — drops common low-information words (`and`, `the`, `is`), reducing noise.
5. **Stemming** — reduces words to a base form (`running` → `run`), shrinking vocabulary and improving generalization.
6. **Rejoining tokens into a string** — most vectorizers (e.g. TF-IDF, CountVectorizer) expect raw text strings, not token lists.

### 1.9 Feature Extraction for Text: Bag-of-Words / CountVectorizer

- **What:** Represents each document as a vector of term counts from a vocabulary built across the dataset.
- **Why bigrams in addition to unigrams:** Unigrams alone lose word order entirely. Bigrams (consecutive word pairs) recover a little local ordering (e.g. "free prize" is more spam-indicative than "free" alone) — though overall sentence structure is still lost.
- **Key `CountVectorizer` parameters:**
  - `min_df`: minimum documents a term must appear in to be kept (filters overly rare terms).
  - `max_df`: maximum document frequency (fraction) before a term is excluded as too common/uninformative.
  - `ngram_range=(1,2)`: capture unigrams and bigrams together.
- **How it works (3 stages):** Tokenization (per `ngram_range`) → Build vocabulary (apply `min_df`/`max_df`) → Vectorization (count occurrences per document).

### 1.10 Random Forests (for Network Anomaly Detection)

- **What:** An ensemble of decision trees; classification uses majority vote across trees, regression averages outputs.
- **Why it generalizes well:** Combining many trees reduces overfitting and correlation compared to a single tree, giving robust performance on high-dimensional data.
- **Three construction concepts:**
  - **Bootstrapping:** sample subsets of training data (with replacement) per tree.
  - **Tree construction:** each split considers a random subset of features, adding diversity and reducing inter-tree correlation.
  - **Voting:** majority vote (classification) or averaging (regression) across trees.
- **For anomaly detection specifically:** Train on data representing **normal** conditions; new points that don't fit well or get low-confidence predictions are flagged as potential anomalies. **(Extra note)** In this module's actual exercise, the model is trained as a **multi-class classifier** over labeled attack categories, not a pure one-class/normal-only detector — see §1.11.

### 1.11 NSL-KDD Dataset and Targets

- **What it is:** A refined version of KDD Cup 1999, removing redundant entries and correcting class imbalance; a standard benchmark for intrusion detection research. Supports both binary (normal vs. abnormal) and multi-class (attack-type) tasks.
- **Binary target (`attack_flag`):** 0 = normal, 1 = any attack.
- **Multi-class target (`attack_map`):** 0 = normal, 1 = DoS, 2 = Probe, 3 = Privilege Escalation, 4 = Access.
  - **DoS examples:** neptune, smurf, back, land, mailbomb, pod, teardrop, worm, etc.
  - **Probe examples:** satan, ipsweep, nmap, portsweep, mscan, saint.
  - **Privilege examples:** buffer_overflow, rootkit, perl, ps, sqlattack, xterm.
  - **Access examples:** guess_passwd, ftp_write, imap, phf, warezclient, spy, etc.
- **Feature categories used:**
  - **Categorical:** `protocol_type`, `service` → one-hot encoded.
  - **Numeric:** a large set of connection statistics (`duration`, `src_bytes`, `dst_bytes`, `serror_rate`, `dst_host_*` rates, etc.).

### 1.12 Malware Image Classification (concept)

- **What:** Classifying malware **families** from grayscale images generated from the raw bytes of PE (Windows executable) binaries, rather than from the binaries directly.
- **Why images:**
  - Safer for a learning environment — you never handle live malicious binaries, only images.
  - **Lossless representation:** every byte maps to one pixel (0=black, 255=white, values in between = gray), so the image can exactly reconstruct the binary.
  - Visual patterns in the image can correspond to structural patterns in the binary — different families can look visibly distinct.
- **Why ML speeds this up:** Manual malware classification normally needs static + dynamic analysis and time-consuming reverse engineering; an ML classifier can automate much of the family-identification step.
- **Malimg dataset:** 9,339 images across 25 malware families, one folder per family (e.g. Adialer.C, Allaple.A, Fakerean, Yuner.A).

### 1.13 CNNs for Malware Classification: Transfer Learning with ResNet50

- **What:** Using a **pre-trained** CNN (ResNet50 — 50 layers, ~23M parameters, from a 2015 paper) as a starting point instead of training from scratch.
- **Why:** Training from random weights could take days/weeks; starting from pre-trained weights and fine-tuning saves enormous time.
- **How (the approach used in this module):**
  1. Load ResNet50 with pre-trained ("DEFAULT") weights.
  2. **Freeze** all layers except the final one (`param.requires_grad = False` for everything but the replaced head) — only the last layer's weights get updated during training.
  3. **Replace the final layer** with a small head sized to the number of classes (25 malware families), via a `Linear → ReLU → Linear` sequence.
  4. Train only that head on the malware image dataset.
- **Trade-off acknowledged in source:** Freezing most of the network may reduce peak achievable performance, but it's a good trade-off for training-time savings in a proof-of-concept.
- **Image preprocessing for the CNN:**
  - **Resize** all images to a fixed size (e.g. 75×75) — CNNs need consistent input dimensions.
  - **Normalize** pixel values (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` — standard ImageNet stats used because ResNet50 was pre-trained on ImageNet) — standardized input is easier to train on.
  - **(Extra)** Note from source: after resize+normalize, fine detail is visibly reduced compared to the raw image, but enough signal remains for classification.

---

## 2. Important Terminology

| Term | Short definition |
|---|---|
| Miniconda | Minimal installer providing the `conda` package/environment manager without the full Anaconda bundle |
| Conda environment | Isolated space for a project's packages/dependencies |
| JupyterLab | Web-based interactive coding/notebook environment |
| Kernel (Jupyter) | The process executing notebook code and holding its state |
| Stateful notebook | Variables/imports persist across cells once run, in execution order, not necessarily top-to-bottom file order |
| DataFrame | pandas' 2D labeled tabular data structure |
| Imputation | Replacing missing/invalid values with estimated ones |
| One-hot encoding | Binary indicator column per category value |
| Label encoding | Integer code per category (can imply false order) |
| Log transform (`log1p`) | Compresses large values more than small ones to reduce skew |
| Train/Validation/Test split | Fit / tune / final-evaluate data partitions |
| Accuracy / Precision / Recall / F1 | See §1.6 |
| Confusion matrix | Table of predicted vs. actual classes |
| Naive Bayes | Probabilistic classifier assuming conditional feature independence given the class |
| Bag-of-words | Representing text as term-count vectors, ignoring most word order |
| CountVectorizer | scikit-learn tool building a bag-of-words feature matrix |
| Unigram / Bigram | Single word / pair of consecutive words as a feature |
| `min_df` / `max_df` | Minimum/maximum document frequency thresholds for keeping a term |
| Pipeline (scikit-learn) | Chains preprocessing + model steps into one consistent object |
| GridSearchCV | Exhaustive search over hyperparameter combinations with cross-validation |
| Hyperparameter (`alpha` in Naive Bayes) | Smoothing factor controlling how unseen words are handled |
| joblib | Library for serializing/deserializing Python objects (e.g. trained models) to disk |
| Random Forest | Ensemble of decision trees combined by voting/averaging |
| Bootstrapping (RF) | Sampling training subsets with replacement per tree |
| NSL-KDD | Refined KDD Cup 1999 intrusion-detection benchmark dataset |
| Malware family | A category of malware sharing characteristics (e.g. behavior, propagation) |
| Byteplot | Image where each pixel's brightness encodes one byte of a binary |
| PE file | Portable Executable — Windows executable file format |
| CNN | Convolutional Neural Network, suited to grid-like (image) data |
| Transfer learning | Reusing a pre-trained model's learned weights for a new, related task |
| Freezing layers | Preventing a layer's weights from updating during training |
| ResNet50 | A 50-layer, ~23M-parameter CNN architecture from 2015, used here pre-trained |
| Epoch | One full pass through the training dataset |
| DataLoader (PyTorch) | Iterates over a dataset in batches, optionally shuffled/parallelized |
| Normalization (images) | Scaling pixel values using a fixed mean/std so inputs are standardized |

---

## 3. Commands and Code Reference (by workflow stage)

### 3.1 Environment Setup

**Install Miniconda**

Windows (via Scoop):
```powershell
Set-ExecutionPolicy RemoteSigned -scope CurrentUser
irm get.scoop.sh | iex
scoop bucket add extras
scoop install miniconda3
conda --version
```

macOS (via Homebrew):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install --cask miniconda
conda --version
```

Linux:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -u
eval "$(/home/$USER/miniconda3/bin/conda shell.$(ps -p $$ -o comm=) hook)"
conda --version
```
- **Purpose:** Install the `conda` package/environment manager with a minimal Python base.
- **Expected result:** `conda --version` prints a version (e.g. `conda 24.9.2`).

**Initialize conda for your shell**
```bash
conda init
conda config --add channels defaults
conda config --add channels conda-forge
conda config --add channels nvidia   # only if you have an Nvidia GPU
conda config --add channels pytorch
conda config --set channel_priority strict
```
- **Purpose:** Lets your shell use `conda activate`/`conda` commands; adds package channels and sets strict channel priority to avoid conflicts.
- **Note:** Modifies shell config files (`.bashrc`/`.zshrc`); reopen your terminal afterward.

**Stop auto-activating the base environment**
```bash
conda config --set auto_activate_base false
```
- **Purpose:** Prevents the `(base)` prefix from appearing in every new terminal; you activate environments explicitly instead.

**Create and manage a project environment**
```bash
conda create -n ai python=3.11
conda activate ai
conda deactivate
```
- **Purpose:** `create -n ai python=3.11` makes an isolated environment named `ai` with Python 3.11; `activate`/`deactivate` switch in and out of it.
- **Why isolate:** Dependency isolation, clean project structure, reproducibility, and system stability (per source).

**Install core AI/ML packages**
```bash
conda install -y numpy scipy pandas scikit-learn matplotlib seaborn transformers datasets tokenizers accelerate evaluate optimum huggingface_hub nltk category_encoders
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install requests requests_toolbelt
```
- **Purpose:** Installs the scientific/ML stack via conda (preferred for performance and dependency resolution) and anything conda doesn't cover via `pip` inside the same environment.
- **⚠ Verify:** `pytorch-cuda=12.4` assumes a matching CUDA setup; adjust per your own GPU/driver versions if following along outside the module's prescribed environment.

**Update conda-managed packages**
```bash
conda update --all
```
- **Note:** Only updates conda-installed packages in the active environment; pip-installed packages must be managed separately. Mixing pip/conda installs raises the risk of dependency conflicts.

### 3.2 JupyterLab

**Install and launch**
```bash
conda install -y jupyter jupyterlab notebook ipykernel
jupyter lab
```
- **Purpose:** Installs and starts the JupyterLab web interface (opens a browser tab).
- **When to use:** Run this from within your activated `ai` environment so the kernel has your installed packages.

**Notebook basics (from source):**
- Create a notebook: click "Python 3" under Notebook in the Launcher.
- Run a cell: `Shift + Enter`.
- Save: `Ctrl + S` (rename via right-click on the tab/file browser entry).
- **Stateful behavior:** variables/functions/imports persist across cells as long as the kernel runs; **running cells out of order** can produce results that don't match the file's visual top-to-bottom order.
- **Restart kernel:** Kernel menu → "Restart Kernel" (keeps outputs) or "Restart Kernel and Clear All Outputs" (also clears outputs). Clears all memory state — re-run needed cells afterward.

### 3.3 Dataset 1 — Network Log Demo Dataset (Loading, Exploring, Cleaning)

**Load**
```python
import pandas as pd
data = pd.read_csv("./demo_dataset.csv")
```

**Explore**
```python
print(data.head())
print(data.info())
print(data.isnull().sum())
```
- **Purpose:** Quick look at sample rows, column types/null counts, and per-column missing-value counts.

**Validity checks (custom functions)**
```python
import re

def is_valid_ip(ip):
    pattern = re.compile(r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    return bool(pattern.match(ip))

invalid_ips = data[~data['source_ip'].astype(str).apply(is_valid_ip)]

def is_valid_port(port):
    try:
        port = int(port)
        return 0 <= port <= 65535
    except ValueError:
        return False

invalid_ports = data[~data['destination_port'].apply(is_valid_port)]

valid_protocols = ['TCP', 'TLS', 'SSH', 'POP3', 'DNS', 'HTTPS', 'SMTP', 'FTP', 'UDP', 'HTTP']
invalid_protocols = data[~data['protocol'].isin(valid_protocols)]

def is_valid_bytes(bytes):
    try:
        bytes = int(bytes)
        return bytes >= 0
    except ValueError:
        return False

invalid_bytes = data[~data['bytes_transferred'].apply(is_valid_bytes)]

def is_valid_threat_level(threat_level):
    try:
        threat_level = int(threat_level)
        return 0 <= threat_level <= 2
    except ValueError:
        return False

invalid_threat_levels = data[~data['threat_level'].apply(is_valid_threat_level)]
```
- **Purpose:** Flag rows with invalid IPs, out-of-range ports, unrecognized protocols, negative/non-numeric byte counts, or out-of-range threat levels.

**Option A — Drop invalid rows**
```python
data = data.drop(invalid_ips.index, errors='ignore')
data = data.drop(invalid_ports.index, errors='ignore')
data = data.drop(invalid_protocols.index, errors='ignore')
data = data.drop(invalid_bytes.index, errors='ignore')
data = data.drop(invalid_threat_levels.index, errors='ignore')
print(data.describe(include='all'))
```
- `errors='ignore'` handles overlap where one row matches multiple invalid criteria.
- **Expected result (per source):** Only 77 clean entries remained in the demo dataset after dropping.

**Option B — Impute instead of dropping**
```python
import pandas as pd
import numpy as np
import re
from ipaddress import ip_address

df = pd.read_csv('demo_dataset.csv')

invalid_ips = ['INVALID_IP', 'MISSING_IP']
invalid_ports = ['STRING_PORT', 'UNUSED_PORT']
invalid_bytes = ['NON_NUMERIC', 'NEGATIVE']
invalid_threat = ['?']

df.replace(invalid_ips + invalid_ports + invalid_bytes + invalid_threat, np.nan, inplace=True)

df['destination_port'] = pd.to_numeric(df['destination_port'], errors='coerce')
df['bytes_transferred'] = pd.to_numeric(df['bytes_transferred'], errors='coerce')
df['threat_level'] = pd.to_numeric(df['threat_level'], errors='coerce')

def is_valid_ip(ip):
    pattern = re.compile(r'^((25[0-5]|2[0-4][0-9]|[01]?\d?\d)\.){3}(25[0-5]|2[0-4]\d|[01]?\d?\d)$')
    if pd.isna(ip) or not pattern.match(str(ip)):
        return np.nan
    return ip

df['source_ip'] = df['source_ip'].apply(is_valid_ip)
```
```python
from sklearn.impute import SimpleImputer

numeric_cols = ['destination_port', 'bytes_transferred', 'threat_level']
categorical_cols = ['protocol']

num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
```
```python
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
```
```python
valid_protocols = ['TCP', 'TLS', 'SSH', 'POP3', 'DNS', 'HTTPS', 'SMTP', 'FTP', 'UDP', 'HTTP']
df.loc[~df['protocol'].isin(valid_protocols), 'protocol'] = df['protocol'].mode()[0]

df['source_ip'] = df['source_ip'].fillna('0.0.0.0')
df['destination_port'] = df['destination_port'].clip(lower=0, upper=65535)

print(df.describe(include='all'))
```
- **Purpose/flow:** Standardize invalid markers to `NaN` → simple impute (median/most-frequent) → optionally refine with `KNNImputer` (considers feature relationships) → apply domain-knowledge fixes → final verification with `describe()`.

### 3.4 Data Transformation (shared techniques)

**One-hot encode a categorical column**
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(df[['protocol']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['protocol']))
df = pd.concat([df.drop('protocol', axis=1), encoded_df], axis=1)
```
- `handle_unknown='ignore'`: unseen categories at inference time don't error out.

**Log-transform a skewed numeric column**
```python
import numpy as np
df["bytes_transferred"] = np.log1p(df["bytes_transferred"])
```

**Train/validation/test split (two-step pattern)**
```python
from sklearn.model_selection import train_test_split

X = df.drop("threat_level", axis=1)
y = df["threat_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1337)
```
- **Note:** `random_state` fixes the split for reproducibility. The second `test_size=0.25` is relative to the already-80%-sized training set, giving ~60/20/20 overall.

### 3.5 Model 1 — Spam Classifier (Naive Bayes)

**Download and extract the SMS Spam Collection dataset**
```python
import requests, zipfile, io

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
response = requests.get(url)
if response.status_code == 200:
    print("Download successful")
else:
    print("Failed to download the dataset")

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")

import os
print("Extracted files:", os.listdir("sms_spam_collection"))
```

**Load it**
```python
import pandas as pd
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)
print(df.head()); print(df.describe()); print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Duplicate entries:", df.duplicated().sum())
df = df.drop_duplicates()
```
- **Dataset facts:** 5,574 SMS messages labeled `ham` (legitimate) or `spam`; from UCI, built by Almeida, Yamakami & Gómez Hidalgo (2011).

**Preprocess text (NLTK pipeline)**
```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

df["message"] = df["message"].str.lower()

import re
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))

from nltk.tokenize import word_tokenize
df["message"] = df["message"].apply(word_tokenize)

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])

df["message"] = df["message"].apply(lambda x: " ".join(x))
```
- **Order matters:** lowercase → strip punctuation/numbers (keep `$`/`!`) → tokenize → remove stop words → stem → rejoin to string.

**Feature extraction (bag-of-words with bigrams)**
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["message"])
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)
```

**Build pipeline and tune hyperparameters**
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])

param_grid = {"classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
grid_search.fit(df["message"], y)

best_model = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params_)
```
- **Why `Pipeline`:** Chains vectorization + classification so the same transform is applied consistently (training and inference).
- **Why `GridSearchCV` with `cv=5` on `alpha`:** Systematically searches Naive Bayes' smoothing parameter via 5-fold cross-validation, scoring by F1 (balances precision/recall — relevant given spam/ham imbalance).

**Evaluate on new, unseen messages**
```python
new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]

import re
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

processed_messages = [preprocess_message(msg) for msg in new_messages]
X_new = best_model.named_steps["vectorizer"].transform(processed_messages)

predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_probabilities = best_model.named_steps["classifier"].predict_proba(X_new)

for i, msg in enumerate(new_messages):
    prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
    spam_probability = prediction_probabilities[i][1]
    ham_probability = prediction_probabilities[i][0]
    print(f"Message: {msg}")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Not-Spam Probability: {ham_probability:.2f}")
    print("-" * 50)
```
- **Critical point:** New data must go through the **exact same preprocessing** used in training, or the model receives mismatched input.

**Save and reload with joblib**
```python
import joblib

model_filename = 'spam_detection_model.joblib'
joblib.dump(best_model, model_filename)

loaded_model = joblib.load(model_filename)
new_data_processed = [preprocess_message(msg) for msg in new_messages]
predictions = loaded_model.predict(new_data_processed)
```
- **Why joblib:** Efficiently serializes Python objects with large arrays (NumPy arrays, scikit-learn models); avoids retraining from scratch each time the app starts.

**Evaluation portal upload (Playground VM)**
```python
import requests, json

url = "http://localhost:8000/api/upload"
model_file_path = "spam_detection_model.joblib"

with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

print(json.dumps(response.json(), indent=4))
```
- **Purpose:** Submits the saved model to the local evaluation API (port 8000 for this model) to check it meets the required performance bar.
- **Alternative:** Browser upload at `http://<VM-IP>:8000/` if not using Jupyter directly.

### 3.6 Model 2 — Network Anomaly Detector (Random Forest, NSL-KDD)

**Download and extract**
```python
import requests, zipfile, io

url = "https://cdn.services-k8s.prod.aws.htb.systems/content/modules/292/KDD_dataset.zip"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.')
```

**Imports**
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
```

**Load with column names**
```python
file_path = r'KDD+.txt'

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]

df = pd.read_csv(file_path, names=columns)
print(df.head())
```

**Create binary and multi-class targets**
```python
df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)

dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod',
               'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps',
                     'rootkit', 'sqlattack', 'xterm']
access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap',
                  'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack',
                  'snmpguess', 'spy', 'warezclient', 'warezmaster',
                  'xclock', 'xsnoop']

def map_attack(attack):
    if attack in dos_attacks: return 1
    elif attack in probe_attacks: return 2
    elif attack in privilege_attacks: return 3
    elif attack in access_attacks: return 4
    else: return 0

df['attack_map'] = df['attack'].apply(map_attack)
```

**Encode categoricals, select numeric features**
```python
features_to_encode = ['protocol_type', 'service']
encoded = pd.get_dummies(df[features_to_encode])

numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

train_set = encoded.join(df[numeric_features])
multi_y = df['attack_map']
```

**Split (train/test, then train/validation)**
```python
train_X, test_X, train_y, test_y = train_test_split(train_set, multi_y, test_size=0.2, random_state=1337)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1337)
```

**Train**
```python
rf_model_multi = RandomForestClassifier(random_state=1337)
rf_model_multi.fit(multi_train_X, multi_train_y)
```

**Evaluate on validation set**
```python
multi_predictions = rf_model_multi.predict(multi_val_X)
accuracy = accuracy_score(multi_val_y, multi_predictions)
precision = precision_score(multi_val_y, multi_predictions, average='weighted')
recall = recall_score(multi_val_y, multi_predictions, average='weighted')
f1 = f1_score(multi_val_y, multi_predictions, average='weighted')
print(f"Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1-Score: {f1:.4f}")

conf_matrix = confusion_matrix(multi_val_y, multi_predictions)
class_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Network Anomaly Detection - Validation Set'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

print(classification_report(multi_val_y, multi_predictions, target_names=class_labels))
```
- `average='weighted'`: accounts for class imbalance across the multi-class labels when aggregating precision/recall/F1.

**Final evaluation on the held-out test set**
```python
test_multi_predictions = rf_model_multi.predict(test_X)
test_accuracy = accuracy_score(test_y, test_multi_predictions)
test_precision = precision_score(test_y, test_multi_predictions, average='weighted')
test_recall = recall_score(test_y, test_multi_predictions, average='weighted')
test_f1 = f1_score(test_y, test_multi_predictions, average='weighted')
print(f"Accuracy: {test_accuracy:.4f}  Precision: {test_precision:.4f}  Recall: {test_recall:.4f}  F1-Score: {test_f1:.4f}")

test_conf_matrix = confusion_matrix(test_y, test_multi_predictions)
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Network Anomaly Detection'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

print(classification_report(test_y, test_multi_predictions, target_names=class_labels))
```

**Save**
```python
import joblib
model_filename = 'network_anomaly_detection_model.joblib'
joblib.dump(rf_model_multi, model_filename)
```

**Upload for evaluation**
```python
import requests, json

url = "http://localhost:8001/api/upload"
model_file_path = "network_anomaly_detection_model.joblib"

with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

print(json.dumps(response.json(), indent=4))
```
- **Note:** This model's evaluation portal uses **port 8001** (vs. 8000 for the spam model).

### 3.7 Model 3 — Malware Classifier (CNN / ResNet50, Malimg)

**Download and unpack the dataset**
```bash
wget https://www.kaggle.com/api/v1/datasets/download/ikrambenabd/malimg-original -O malimg.zip
unzip malimg.zip
ls malimg_paper_dataset_imgs
```
- **Dataset facts:** 9,339 grayscale PNG images, 25 malware-family folders (e.g. `Adialer.C`, `Allaple.A`, `Fakerean`, `Yuner.A`).

**Explore class distribution**
```python
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_BASE_PATH = "./malimg_paper_dataset_imgs/"

dist = {}
for mlw_class in os.listdir(DATA_BASE_PATH):
    mlw_dir = os.path.join(DATA_BASE_PATH, mlw_class)
    dist[mlw_class] = len(os.listdir(mlw_dir))

htb_green = "#9FEF00"; node_black = "#141D2B"; hacker_grey = "#A4B1CD"
classes = list(dist.keys()); frequencies = list(dist.values())

plt.figure(facecolor=node_black)
sns.barplot(y=classes, x=frequencies, edgecolor="black", orient='h', color=htb_green)
plt.title("Malware Class Distribution", color=htb_green)
plt.xlabel("Malware Class Frequency", color=htb_green)
plt.ylabel("Malware Class", color=htb_green)
plt.show()
```
- **Purpose:** Spot over-/under-represented families that could skew training.

**Split into train/val/test folders**
```bash
pip3 install split-folders
```
```python
import splitfolders

DATA_BASE_PATH = "./malimg_paper_dataset_imgs/"
TARGET_BASE_PATH = "./newdata/"
TRAINING_RATIO = 0.8
TEST_RATIO = 1 - TRAINING_RATIO

splitfolders.ratio(input=DATA_BASE_PATH, output=TARGET_BASE_PATH, ratio=(TRAINING_RATIO, 0, TEST_RATIO))
```
- **Expected result (per source counts):** `./newdata/train` 7,459 files, `./newdata/test` 1,880 files, `./newdata/val` 0 files (no validation split was used in this exercise).

**Preprocessing transforms + DataLoaders**
```python
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

def load_datasets(base_path, train_batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.Resize((75, 75)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=os.path.join(base_path, "train"), transform=transform)
    test_dataset = ImageFolder(root=os.path.join(base_path, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    n_classes = len(train_dataset.classes)
    return train_loader, test_loader, n_classes
```
- **Why read `n_classes` dynamically:** Keeps the code valid even if classes are added/removed later, rather than hardcoding 25.

**Model definition (frozen ResNet50 + custom head)**
```python
import torch.nn as nn
import torchvision.models as models

HIDDEN_LAYER_SIZE = 1000

class MalwareClassifier(nn.Module):
    def __init__(self, n_classes):
        super(MalwareClassifier, self).__init__()
        self.resnet = models.resnet50(weights='DEFAULT')

        for param in self.resnet.parameters():
            param.requires_grad = False

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, n_classes)
        )

    def forward(self, x):
        return self.resnet(x)
```
```python
DATA_PATH = "./newdata/"
TRAINING_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024

train_loader, test_loader, n_classes = load_datasets(DATA_PATH, TRAINING_BATCH_SIZE, TEST_BATCH_SIZE)
model = MalwareClassifier(n_classes)
```

**Training function**
```python
import torch, time

def compute_accuracy(n_correct, n_total):
    return round(100 * n_correct / n_total, 2)

def train(model, train_loader, n_epochs, verbose=False):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    training_data = {"accuracy": [], "loss": []}

    for epoch in range(n_epochs):
        running_loss = 0; n_total = 0; n_correct = 0
        checkpoint = time.time() * 1000

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            n_total += labels.size(0)
            n_correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_duration = int(time.time() * 1000 - checkpoint)
        epoch_accuracy = compute_accuracy(n_correct, n_total)

        training_data["accuracy"].append(epoch_accuracy)
        training_data["loss"].append(epoch_loss)

        if verbose:
            print(f"[i] Epoch {epoch+1} of {n_epochs}: Acc: {epoch_accuracy:.2f}% Loss: {epoch_loss:.4f} (Took {epoch_duration} ms).")

    return training_data
```
- **Loss/optimizer choice:** `CrossEntropyLoss` for multi-class classification; `Adam` optimizer (defaults, no explicit LR shown in source).

**Save model (TorchScript)**
```python
def save_model(model, path):
    model_scripted = torch.jit.script(model)
    model_scripted.save(path)
```

**Predict / evaluate**
```python
def predict(model, test_data):
    model.eval()
    with torch.no_grad():
        output = model(test_data)
        _, predicted = torch.max(output.data, 1)
    return predicted

def evaluate(model, test_loader):
    model.eval()
    n_correct = 0; n_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            predicted = predict(model, data)
            n_total += target.size(0)
            n_correct += (predicted == target).sum().item()
    return compute_accuracy(n_correct, n_total)
```
- `model.eval()` + `torch.no_grad()`: disables training-only behavior (e.g. dropout) and gradient tracking for faster, memory-light inference.

**Plot helpers**
```python
import matplotlib.pyplot as plt

def plot(data, title, label, xlabel, ylabel):
    htb_green = "#9FEF00"; node_black = "#141D2B"; hacker_grey = "#A4B1CD"
    plt.figure(figsize=(10, 6), facecolor=node_black)
    plt.plot(range(1, len(data)+1), data, label=label, color=htb_green)
    plt.title(title, color=htb_green); plt.xlabel(xlabel, color=htb_green); plt.ylabel(ylabel, color=htb_green)
    plt.show()

def plot_training_accuracy(training_data):
    plot(training_data['accuracy'], "Training Accuracy", "Accuracy", "Epoch", "Accuracy (%)")

def plot_training_loss(training_data):
    plot(training_data['loss'], "Training Loss", "Loss", "Epoch", "Loss")
```

**Full run script**
```python
DATA_PATH = "./newdata/"
N_EPOCHS = 10
TRAINING_BATCH_SIZE = 512
TEST_BATCH_SIZE = 1024
HIDDEN_LAYER_SIZE = 1000
MODEL_FILE = "malware_classifier.pth"

train_loader, test_loader, n_classes = load_datasets(DATA_PATH, TRAINING_BATCH_SIZE, TEST_BATCH_SIZE)
model = MalwareClassifier(n_classes)

print("[i] Starting Training...")
training_information = train(model, train_loader, N_EPOCHS, verbose=True)

save_model(model, MODEL_FILE)

accuracy = evaluate(model, test_loader)
print(f"[i] Inference accuracy: {accuracy}%.")

plot_training_accuracy(training_information)
plot_training_loss(training_information)
```
- **Expected result (per source run):** Training accuracy climbed from ~57% (epoch 1) to ~96% (epoch 10); final **test/inference accuracy ≈ 88.54%**. The module notes this gap is acceptable given the simplified setup (frozen backbone, few epochs) favoring training speed over peak performance, and that results vary with the random train/test split.

**Upload for evaluation**
```python
import requests, json

url = "http://localhost:8002/api/upload"
model_file_path = "malware_classifier.pth"

with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

print(json.dumps(response.json(), indent=4))
```
- **Note:** Port **8002** for this model's evaluation portal. Per source: in the Playground VM, one epoch can take up to ~10 minutes, and ~3 epochs is stated as sufficient to reach the required accuracy there; evaluating an uploaded model may take up to 2 minutes.

### 3.8 Reference: Scikit-learn and PyTorch API Overview

> From the module's high-level library primer — not code you necessarily run verbatim, but the APIs worth recognizing.

**Scikit-learn — common preprocessing**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- `StandardScaler` (zero mean, unit variance), `MinMaxScaler` (fixed range, typically 0–1), `RobustScaler` (robust to outliers).

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
```

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

**Scikit-learn — model selection/evaluation**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```

**Scikit-learn — generic train/predict pattern**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**PyTorch — tensors**
```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
if torch.cuda.is_available():
    x = x.to('cuda')
```

**PyTorch — Sequential model**
```python
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)
```

**PyTorch — custom Module subclass**
```python
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

model = CustomModel()
```

**PyTorch — optimizers / loss functions**
```python
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()  # multi-class
# nn.BCEWithLogitsLoss()  # binary
# nn.MSELoss()  # regression
```

**PyTorch — custom Dataset/DataLoader**
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**PyTorch — save/load (state_dict style, note: differs from the `torch.jit.script` style used for the malware model)**
```python
torch.save(model.state_dict(), 'model.pth')

model = CustomModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```
- **⚠ Verify:** This save/load pattern (`state_dict`) is from the general PyTorch primer in §4. The malware classifier build in §3.7 instead uses `torch.jit.script(...).save(...)` — a different serialization approach (TorchScript). Use whichever matches how the model was saved when loading it back.

---

## 4. Practical Workflow (Generalized Across All Three Models)

1. **Set up the environment** — Miniconda, create/activate a dedicated environment, install the package set, launch JupyterLab.
2. **Acquire the dataset** — download/extract programmatically (`requests` + `zipfile`/`io`, or `wget`/`unzip`), or load from a provided file.
3. **Explore** — `head()`, `info()`, `describe()`, null counts, duplicate counts, class-distribution plots (for images: count files per class folder).
4. **Clean** — decide drop vs. impute for invalid/missing values; standardize invalid markers to `NaN` first if imputing.
5. **Transform** — encode categoricals (one-hot for nominal data), log-transform skewed numerics, text-specific preprocessing (lowercase → clean → tokenize → stopword-remove → stem → rejoin) or image-specific preprocessing (resize → tensor → normalize).
6. **Extract features** — vectorize text (`CountVectorizer` with chosen `ngram_range`/`min_df`/`max_df`), or select/assemble numeric+encoded columns, or build `ImageFolder` + `DataLoader`.
7. **Split** — train/validation/test (commonly 60/20/20 via two sequential `train_test_split` calls), using a fixed `random_state` for reproducibility.
8. **Train** — fit directly, or wrap in a `Pipeline` + `GridSearchCV` for hyperparameter tuning (scikit-learn models); write an explicit epoch loop with an optimizer and loss function (PyTorch models). For transfer learning, freeze pretrained layers and only train the new head.
9. **Evaluate** — accuracy, precision, recall, F1 (use `average='weighted'` for multi-class with imbalance); confusion matrix and classification report for a full picture; evaluate on validation first, then a final, untouched test set.
10. **Save** — `joblib.dump()` for scikit-learn pipelines/models; `torch.jit.script(...).save()` (or `state_dict`, depending on your deployment pattern) for PyTorch models.
11. **(This module specifically) Submit for automated evaluation** — POST the saved model file to the Playground VM's local API endpoint (port varies per model) to receive a pass/fail style result.

### Common Mistakes
- Forgetting that new/inference data must go through **identical preprocessing** to training data (tokenization, stemming, encoding, scaling, image transforms) — mismatched input breaks predictions silently.
- Judging a model by accuracy alone on an imbalanced dataset (see the Fundamentals-module spam example: 99% "legitimate" baseline accuracy while catching zero spam).
- Running Jupyter cells out of order and trusting stale variable state.
- Mixing `pip` and `conda` installs carelessly, risking dependency conflicts; prefer `conda install` first, `pip` only as a fallback.
- Dropping too much data when invalid entries are a large fraction of a small dataset — consider imputing instead.
- Treating a label-encoded categorical as if it had a real numeric order.
- Forgetting `model.eval()` and `torch.no_grad()` during inference (wastes memory/compute and can change some layers' behavior).
- Hardcoding the number of classes in image/multi-class problems instead of reading it from the data.
- Mismatching the save/load method for PyTorch models (`torch.jit.script` vs. `state_dict`).

### Troubleshooting
- **`conda` not found after install:** reopen your terminal; confirm `conda init` ran and shell config files were sourced.
- **Dependency conflicts:** avoid mixing pip/conda where possible; `conda update --all` only touches conda-installed packages.
- **Unexpected variable values in a notebook:** check the actual **execution order** of cells, not their visual order; consider "Restart Kernel and Clear All Outputs" to start clean.
- **Model inference gives nonsensical predictions:** verify the new data was preprocessed identically to the training data.
- **Imbalanced multi-class metrics look inconsistent:** check `average='weighted'` (or review per-class results in the classification report) rather than relying on overall accuracy.
- **Low malware-classifier accuracy:** remember this module intentionally trades performance for training-time speed (frozen backbone, modest epoch counts); more epochs, unfreezing more layers, or a cleaner class balance could improve it, at the cost of time.

---

## 5. Quick Revision Sheet

### 5.1 Top 10 Things to Remember
1. Workflow: environment → load → explore → clean → transform → extract features → split → train → evaluate → save.
2. A good dataset is relevant, complete, consistent, high-quality, representative, balanced, and appropriately sized.
3. For invalid/missing data: drop (simpler, costs data) or impute (keeps data; standardize invalid markers to `NaN` first, then simple → KNN/Iterative → domain-knowledge fixes).
4. One-hot encoding avoids false ordinal relationships; label encoding can introduce them.
5. Log-transform (`log1p`) tames right-skewed numeric features without losing information.
6. Split data into train/validation/test (commonly ~60/20/20) with a fixed `random_state`.
7. Accuracy alone misleads on imbalanced classes; use precision/recall/F1/confusion matrix together, and weight the trade-off by context (recall-favoring for threat detection, precision-favoring under limited resources).
8. Spam model: NLTK text pipeline (lowercase → clean → tokenize → stopword-remove → stem → rejoin) → `CountVectorizer` (unigrams+bigrams) → `MultinomialNB` in a `Pipeline`, tuned via `GridSearchCV` on `alpha`.
9. Network anomaly model: NSL-KDD, one-hot encode `protocol_type`/`service`, keep numeric traffic-stat features, `RandomForestClassifier` for multi-class attack categorization (DoS/Probe/Privilege/Access/Normal).
10. Malware model: convert PE binaries to grayscale byteplot images (lossless, safe to handle), transfer-learn on a frozen, pretrained ResNet50 with a replaced classification head.

### 5.2 Compact Command/Code Reference

| Task | Command/Code |
|---|---|
| Create conda env | `conda create -n ai python=3.11` |
| Activate / deactivate | `conda activate ai` / `conda deactivate` |
| Install core packages | `conda install -y numpy scipy pandas scikit-learn matplotlib seaborn ...` |
| Launch JupyterLab | `jupyter lab` |
| Load CSV | `pd.read_csv("file.csv")` |
| Inspect | `.head()`, `.info()`, `.isnull().sum()`, `.describe()` |
| Drop rows by index | `df.drop(idx, errors='ignore')` |
| Simple impute | `SimpleImputer(strategy='median'/'most_frequent')` |
| KNN impute | `KNNImputer(n_neighbors=5)` |
| One-hot encode | `OneHotEncoder()` / `pd.get_dummies()` |
| Log transform | `np.log1p(col)` |
| Split data | `train_test_split(X, y, test_size=..., random_state=...)` |
| Vectorize text | `CountVectorizer(min_df=, max_df=, ngram_range=(1,2))` |
| Naive Bayes | `MultinomialNB()` |
| Grid search | `GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")` |
| Random Forest | `RandomForestClassifier(random_state=...)` |
| Evaluation metrics | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report` |
| Save sklearn model | `joblib.dump(model, "file.joblib")` |
| Load sklearn model | `joblib.load("file.joblib")` |
| PyTorch dataset/loader | `ImageFolder(...)`, `DataLoader(dataset, batch_size=, shuffle=, num_workers=)` |
| Image transforms | `transforms.Compose([Resize, ToTensor, Normalize])` |
| Freeze pretrained layers | `param.requires_grad = False` |
| Save TorchScript model | `torch.jit.script(model).save("file.pth")` |
| Upload model for evaluation | `requests.post(url, files={"model": open(path,"rb")})` |

### 5.3 Common Mistakes to Avoid
- Skipping identical preprocessing between training and inference data.
- Trusting accuracy alone on imbalanced classes.
- Running notebook cells out of order and misreading stale state.
- Mixing pip/conda installs without care.
- Hardcoding class counts instead of reading them from the data.
- Forgetting `model.eval()` + `torch.no_grad()` at inference time.

### 5.4 Revision Questions

1. **What are the 10 stages of the generalized ML workflow used across all three models in this module?**
   Environment setup → load → explore → clean → transform → extract features → split → train → evaluate → save (see §4).
2. **Name the 7 attributes of a "good" dataset.**
   Relevance, Completeness, Consistency, Quality, Representativeness, Balance, Size.
3. **Why standardize invalid markers to `NaN` before imputing?**
   It gives a single, uniform representation of "missing," enabling consistent downstream imputation steps.
4. **Why might one-hot encoding be preferred over label encoding for a feature like `protocol`?**
   Label encoding would imply a false numeric order between categories; one-hot encoding avoids that by giving each category its own independent binary column.
5. **Why apply a log transform to `bytes_transferred`?**
   It reduces skew from a few very large values, producing a more balanced distribution that helps the model generalize and resist being dominated by outliers.
6. **Why use `GridSearchCV` with `scoring="f1"` for the Naive Bayes spam model?**
   To systematically find the smoothing parameter (`alpha`) that gives the best balance of precision and recall via cross-validation, which matters more than raw accuracy on an imbalanced spam/ham dataset.
7. **Why are bigrams added to the bag-of-words model alongside unigrams?**
   To recover some local word-order information (e.g. "free prize" vs. "free" alone), which unigrams alone discard.
8. **What are the three core ideas behind how a Random Forest is built?**
   Bootstrapping (sampling with replacement per tree), random feature subsets per split, and majority voting/averaging across trees.
9. **Why convert malware binaries into images rather than analyzing the binaries directly in this module?**
   It's safer for a learning environment (no live malicious binaries handled) and the byteplot image is a lossless, exact representation of the binary that can still reveal visual family-specific patterns.
10. **Why freeze most of ResNet50's layers when training the malware classifier?**
    To drastically cut training time by only updating the new final layer's weights, accepting some reduction in peak performance as a worthwhile trade-off for a proof-of-concept.

### 5.5 Two-Minute Final Summary

This module walks the full applied ML pipeline through three projects. Setup uses Miniconda for isolated, reproducible environments and JupyterLab (a stateful, cell-based notebook interface) for iterative development.

Every dataset goes through the same arc: load into pandas (or image folders), explore for shape/nulls/duplicates/class balance, clean (drop or impute invalid values, standardizing markers to `NaN` first), transform (one-hot encode categoricals, log-transform skewed numerics, or run text/image-specific preprocessing), extract features, and split into train/validation/test.

The **Spam Classifier** turns SMS text into a cleaned, tokenized, stemmed string, vectorizes it with `CountVectorizer` (unigrams+bigrams), and trains a `MultinomialNB` inside a `Pipeline`, tuned via `GridSearchCV` on the smoothing parameter `alpha`, scored by F1 to stay robust to spam/ham imbalance.

The **Network Anomaly Detector** uses the NSL-KDD dataset, builds both binary and multi-class attack targets (Normal/DoS/Probe/Privilege/Access), one-hot encodes `protocol_type`/`service`, keeps a wide set of numeric traffic statistics, and trains a `RandomForestClassifier`, evaluated with weighted precision/recall/F1 and a confusion matrix.

The **Malware Classifier** converts PE binaries into lossless grayscale byteplot images (Malimg dataset, 25 families), resizes/normalizes them for CNN input, and fine-tunes a frozen, pretrained **ResNet50** by replacing and training only its final classification layer — trading some accuracy for drastically faster training.

Across all three, evaluation always goes beyond accuracy (precision/recall/F1/confusion matrix, weighted for multi-class), models are serialized for reuse (`joblib` for scikit-learn, `torch.jit.script` for PyTorch), and in this module specifically, finished models are POSTed to a local evaluation API running on the Playground VM to confirm they meet a performance bar.

---

## 6. Skills Assessment: Approach Only

The assessment asks you to train a sentiment classifier (positive/negative) on the IMDB movie review dataset and upload it to the local evaluation portal for a flag. These notes contain no answers. The workflow is the same as the Spam Classifier build in §3.5: load and explore the text dataset, preprocess it (NLTK-style cleaning/tokenizing/stemming), extract features (e.g. `CountVectorizer`), train and tune a classifier, evaluate it, save it (matching the expected filename/format for the upload script), and submit it to the evaluation endpoint.
