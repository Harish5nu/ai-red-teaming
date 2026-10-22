---
tags: [htb-academy, ai, machine-learning, applied, revision]
module: Applications of AI in InfoSec
---

# Applications of AI in InfoSec: Revision Notes

> **Legend:** Unmarked = from your module text. **(Extra)** = my own added explanation or example. **⚠ Verify** = unclear or inconsistent in the source.
> **Format note:** This is a hands-on, code-heavy module building three real models (spam classifier, network anomaly detector, malware classifier). Section 3 holds the commands/code, organized by model pipeline rather than by single CLI tools.
> **No flags or assessment answers are included.** Section 6 gives approach only for the three evaluation checkpoints and the skills assessment.

---

## 1. Main Concepts: What, Why, How

### 1.1 Module Overview

- **What:** A practical follow-up to Fundamentals of AI — building and evaluating three real models end-to-end:
  1. **Spam Classifier** (SMS messages, Naive Bayes).
  2. **Network Anomaly Detection Model** (NSL-KDD dataset, Random Forest).
  3. **Malware Classifier** (byteplot images of binaries, CNN/ResNet50).
- **Why:** Moves from theory to the full workflow: explore data → preprocess → train → evaluate → save/deploy.
- **Environment options:** The hosted **Playground VM** (Jupyter at `http://<VM-IP>:8888`) or your **own environment** (min. 4GB RAM, 4 CPU cores recommended).

### 1.2 Environment Setup — Why Conda/Miniconda

- **What:** Miniconda = minimal installer for the conda package manager + core Python, without Anaconda's full bundled library suite.
- **Why Miniconda over plain Python:**
  - **Performance:** optimized packages for data science/ML.
  - **Package management:** conda resolves complex dependency webs (important in deep learning).
  - **Environment isolation:** separate environments per project avoid conflicts.
- **Conda vs pip:** conda's curated channels may not cover every package; pip can be used *inside* a conda environment for anything conda lacks.
- **Updates:** `conda update --all` only updates conda-installed packages, not pip-installed ones — mixing increases dependency conflict risk.

### 1.3 JupyterLab

- **What:** Web-based interactive coding/data/visualization environment; the primary workspace for this module.
- **Why:** Lets you run code in individual cells (fast iteration), integrates with plotting libraries, and supports markdown/LaTeX documentation.
- **Cell types:** Code, Markdown, Raw.
- **Key concept — statefulness:** Variables/functions/imports from one executed cell persist for all later cells as long as the kernel runs. Running cells **out of order** can produce unexpected results because earlier state may not match what you'd expect from reading top-to-bottom.
  - **Example from source:** Cell 1: `x = 1`. Later cell: `print(x)` → prints `1`. If you then edit cell 1 to `x = 2` and re-run *only* cell 1 (not the print cell), the stored value of `x` becomes `2`, so the next time you run the print cell it shows `2` — illustrating how edits and re-run order silently change results.
- **Restarting the kernel:** Clears all variables/functions/imports from memory. Use when the environment becomes cluttered or behaves unexpectedly. "Restart Kernel" keeps prior outputs displayed; "Restart Kernel and Clear All Outputs" also wipes displayed outputs. After restarting, **re-run** cells with definitions/imports to restore state.

### 1.4 Key Python Libraries — Purpose Only (not line-by-line code)

| Library | Role in this module |
|---|---|
| **Scikit-learn** | Classic ML: Naive Bayes, Random Forest, preprocessing (scalers, encoders, imputers), model selection (`train_test_split`, `GridSearchCV`, `cross_val_score`), metrics |
| **PyTorch** | Deep learning: tensors, `nn.Module`/`Sequential` models, optimizers, loss functions, training loops, `Dataset`/`DataLoader` |
| **pandas** | Loading/inspecting/cleaning tabular data (DataFrames) |
| **NLTK** | Text preprocessing: tokenization, stopword removal, stemming |
| **joblib** | Serializing/deserializing scikit-learn models (and other large-array objects) to disk |
| **torchvision** | Pretrained CNN models (ResNet50), image transforms, `ImageFolder` dataset loader |
| **splitfolders** | Splitting an image dataset into train/val/test folders |
| **seaborn / matplotlib** | Visualization (class distributions, confusion matrices, training curves) |

- **PyTorch's key differentiator (source):** **Dynamic computational graphs** — built on the fly during the forward pass (vs. static graphs in some other frameworks), giving more flexible/intuitive model building and debugging. Also: GPU support, `autograd` for automatic differentiation, TorchVision integration.

### 1.5 Datasets — What Makes One "Good"

- **What:** Structured collections of data for training/evaluation. Forms covered: tabular, image, text, time series.
- **Why quality matters:** Directly drives model accuracy, generalization, training efficiency, and reliability.
- **Good-dataset attributes (table from source):**

| Attribute | Why it matters |
|---|---|
| Relevance | Irrelevant data adds noise, lowers performance |
| Completeness | Missing data biases models (imputation helps, but starting complete is better) |
| Consistency | Inconsistent formats cause preprocessing/training errors |
| Quality | Errors from collection/entry/transmission reduce accuracy |
| Representativeness | An unrepresentative dataset biases the model |
| Balance | Imbalanced classes bias models against minority classes |
| Size | Too small → not enough signal; too large → computationally expensive |

### 1.6 Data Preprocessing — General Workflow

- **What:** Transforming raw data into a format algorithms can use effectively.
- **Four key techniques (source):**
  1. **Data Cleaning:** missing values, duplicates, noise.
  2. **Data Transformation:** normalizing, encoding, scaling, reducing.
  3. **Data Integration:** merging/aggregating multiple sources.
  4. **Data Formatting:** converting types, reshaping structures.
- **Handling invalid entries — two strategies:**
  - **Dropping:** simplest; keeps the dataset clean but can lose a lot of data (source example: dropping bad rows from the demo network-log dataset left only **77 clean entries**).
  - **Imputing:** replace missing/invalid values with estimates instead of discarding them, preserving more information.
    - Standardize all known "invalid" markers (`MISSING_IP`, `?`, etc.) to `NaN` first.
    - Simple imputation: median/mean for numeric columns, most-frequent for categorical.
    - Advanced imputation: **KNNImputer** or **IterativeImputer** — consider relationships among features for more contextually meaningful fills.
    - Apply domain knowledge afterward (e.g., default missing IPs to `0.0.0.0`, clip ports to the valid range, map unknown protocols to the mode).

### 1.7 Data Transformation — Encoding and Skew

- **Encoding categorical features — why:** ML algorithms need numeric input.
  - **OneHotEncoder:** one binary column per category; avoids implying false ordinal relationships. Can blow up feature count with high-cardinality features.
  - **LabelEncoder:** integer codes; risk of implying unintended order.
  - **HashingEncoder / frequency-based methods:** for high-cardinality features, to control feature space size.
- **Handling skewed numeric data:**
  - **What:** Uneven value distribution, often with a long tail of extreme values.
  - **Why it matters:** Hurts models sensitive to outliers or that assume more uniform/normal distributions.
  - **How:** Apply a **log transform** (`log1p`, i.e. `log(1+x)`) to compress large values more than small ones, avoiding `log(0)` issues and producing a more balanced distribution.
- **Data splitting:**

| Set | Purpose | Typical share |
|---|---|---|
| Training | Fit the model | ~60-80% |
| Validation | Tune hyperparameters, model selection | ~10-20% |
| Test | Final, untouched evaluation | ~10-20% |

- **Nested-split math (source example):** First split 80% train / 20% test. Then split the 80% training portion again with `test_size=0.25`, which is 25% *of that 80%*, i.e. `0.8 × 0.25 = 0.2` → 20% of the whole dataset becomes validation, leaving 60% for final training.

### 1.8 Evaluation Metrics (Recap + Context)

- **Accuracy:** correct predictions / all predictions. Can be **misleading under class imbalance** (source's spam example: predicting "legitimate" for every email with a 99% legitimate dataset yields 0.99 accuracy while catching **zero** spam).
- **Precision:** of predicted positives, how many are truly positive. High precision → fewer false alarms, but doesn't guarantee catching all positives.
- **Recall:** of actual positives, how many were found. High recall → fewer missed cases, but can flood results with false positives if precision is low.
- **F1-score:** harmonic mean of precision and recall; balances both, useful under class imbalance.
- **Other metrics mentioned:** Specificity, AUC (ROC), Matthews Correlation Coefficient, Confusion Matrix.
- **Context matters (source's framing):**
  - Check metric consistency across data segments.
  - Does the dataset reflect real-world conditions (including imbalance)?
  - Are the costs of false positives vs. false negatives accounted for?
  - **Trade-off examples:** threat detection often favors **recall** (don't miss real threats, tolerate some false alarms); resource-constrained environments may favor **precision** (fewer false alarms to chase).

### 1.9 Naive Bayes for Spam Detection (Theory Recap + Worked Example)

- **What:** Applies Bayes' theorem with the "naive" assumption that features (words) are conditionally independent given the class.
- **Formula:** `P(A|B) = (P(B|A) * P(A)) / P(B)`
  - In spam terms: `P(Spam|Features) = (P(Features|Spam) * P(Spam)) / P(Features)`
- **Naive independence simplification:** `P(Features|Spam) = P(feature1|Spam) * P(feature2|Spam) * ...`
- **Worked example from source:**
  - Priors: P(Spam)=0.3, P(Not Spam)=0.7.
  - Likelihoods: P(F1|Spam)=0.4, P(F2|Spam)=0.5, P(F1|Not Spam)=0.2, P(F2|Not Spam)=0.3.
  - `P(F1,F2|Spam) = 0.4 × 0.5 = 0.2`
  - `P(F1,F2|Not Spam) = 0.2 × 0.3 = 0.06`
  - `P(F1,F2) = (0.2×0.3) + (0.06×0.7) = 0.06 + 0.042 = 0.102`
  - `P(Spam|F1,F2) = 0.06 / 0.102 ≈ 0.588`
  - `P(Not Spam|F1,F2) = 0.042 / 0.102 ≈ 0.412`
  - Since 0.588 > 0.412 → classified as **spam**.

### 1.10 The SMS Spam Dataset

- **What:** SMS Spam Collection — 5,574 SMS messages labeled `ham` (legitimate) or `spam`, assembled by Almeida, Yamakami (UNICAMP, Brazil) and Gómez Hidalgo (Optenet, Spain), presented at ACM DocEng 2011, sourced from Grumbletext, the NUS SMS Corpus, and Caroline Tag's PhD thesis.
- **Why SMS specifically:** Most existing spam-filtering resources at the time focused on email, not text messages.
- **`ham` vs `spam`:** ham = legitimate messages from known contacts/subscriptions; spam = unsolicited, low/no value, possibly risky content.

### 1.11 Text Preprocessing Pipeline (Spam)

Order matters — this is the sequence used in the module:

1. **Lowercasing:** treats "Free" and "free" as the same token.
2. **Removing punctuation/numbers (selectively):** strip most non-letter characters but **keep `$` and `!`**, since they can carry spam-relevant signal (money mentions, emphasis).
3. **Tokenization:** split text into individual word tokens (`nltk.word_tokenize`).
4. **Stop-word removal:** drop common low-信息 words (`and`, `the`, `is`, ...) that don't help distinguish classes.
5. **Stemming:** reduce words to a base/root form (e.g. "running" → "run") via `PorterStemmer`, shrinking vocabulary and consolidating variants.
6. **Rejoin tokens into a string:** most vectorizers (e.g. CountVectorizer, TF-IDF) expect raw text strings, not token lists.

### 1.12 Feature Extraction — Bag-of-Words / CountVectorizer

- **What:** Converts text into numeric vectors of term counts (bag-of-words model).
- **Unigrams vs bigrams:**
  - **Unigrams** (single words) ignore word order entirely.
  - **Bigrams** (consecutive word pairs) recapture a little local order/context (e.g. "free prize" is more specific than "free" alone).
  - CountVectorizer with `ngram_range=(1,2)` still does **not** preserve full sentence structure — only localized patterns within the chosen n-gram window.
- **Key CountVectorizer parameters:**
  - `min_df`: minimum documents a term must appear in to be kept (filters out overly rare terms).
  - `max_df`: maximum document-frequency proportion before a term is excluded as "too common" (e.g. 0.9 = drop terms in >90% of documents).
  - `ngram_range`: which n-gram sizes to extract (e.g. `(1,2)` = unigrams + bigrams).
- **Three stages (source):** Tokenization → Vocabulary building (apply `min_df`/`max_df`) → Vectorization (count term occurrences per document).

### 1.13 Training Pipeline Pattern (Spam Model)

- **Pipeline:** chains vectorization + classifier into a single object so the same transformation is consistently applied; simplifies both development and later inference.
- **Hyperparameter tuning with `GridSearchCV`:**
  - Systematically tests parameter combinations (here: `alpha`, MultinomialNB's Laplace-smoothing factor controlling how unseen words are handled).
  - Uses cross-validation (`cv=5`) and a scoring metric (`"f1"`) to pick the best configuration.
- **Why `alpha` matters:** smoothing avoids zero probabilities for unseen words during inference, balancing bias/variance.

### 1.14 Model Evaluation Workflow (Spam Model)

- **Consistency requirement:** New messages must go through the **exact same preprocessing** (lowercase → strip chars → tokenize → remove stopwords → stem → rejoin) before vectorizing/predicting, or the model receives data in an unfamiliar format.
- **Output interpretation:** For each message, the model gives a predicted label plus class probabilities (`predict_proba`), showing confidence rather than just a binary call.
- **Saving/loading with joblib:**
  - **Why:** avoids retraining from scratch every time; speeds up deployment and reduces startup latency.
  - Saves the **entire pipeline state** (vectorizer + classifier + learned parameters), so a reloaded model is immediately ready to predict.

### 1.15 Random Forests for Anomaly Detection

- **What:** Ensemble of decision trees; classification uses majority vote across trees, regression averages outputs.
- **Three construction concepts:**
  1. **Bootstrapping:** sample subsets of training data with replacement; each subset trains a separate tree.
  2. **Tree construction:** a random subset of features considered at each split, reducing correlation between trees.
  3. **Voting:** majority vote (classification) or averaging (regression) across all trees.
- **Why Random Forest for anomaly detection:** Trained on data representing **normal conditions**; new points that don't fit well, or get low-confidence predictions, are flagged as potential anomalies.

### 1.16 The NSL-KDD Dataset

- **What:** A refined version of the original KDD Cup 1999 intrusion-detection dataset, with redundant entries removed and class imbalance corrected. A common benchmark for intrusion detection research.
- **Capabilities:** Supports both **binary** (normal vs. abnormal) and **multi-class** (attack-type) classification.
- **Attack category groupings used in this module:**

| Category | Label | Example attacks |
|---|---|---|
| Normal | 0 | — |
| DoS (Denial of Service) | 1 | neptune, smurf, back, land, mailbomb, pod, etc. |
| Probe | 2 | ipsweep, mscan, nmap, portsweep, saint, satan |
| Privilege Escalation | 3 | buffer_overflow, rootkit, sqlattack, perl, ps, xterm, loadmdoule |
| Access | 4 | guess_passwd, ftp_write, phf, imap, multihop, snmpguess, etc. |

- **Why map to numeric categories:** turns raw attack-name strings into targets a classifier can learn (0–4 integer labels).

### 1.17 Preprocessing the NSL-KDD Data

- **Binary target:** `attack_flag` = 0 if `attack == 'normal'`, else 1.
- **Multi-class target:** `attack_map` via a lookup function mapping each named attack into one of the 4 categories above, defaulting to 0 (normal) for unlisted/unknown strings.
- **Encoding:** One-hot encode `protocol_type` and `service` (categorical) using `pd.get_dummies`, avoiding implied ordering.
- **Numeric features retained:** raw volume metrics (`duration`, `src_bytes`, `dst_bytes`) plus many **rate/statistical** features (`serror_rate`, `dst_host_srv_diff_host_rate`, etc.) that capture session-level behavior patterns.
- **Final feature set:** one-hot columns joined with the selected numeric columns → `train_set`.
- **Splitting:** 80/20 train/test, then a further 70/30 split of the 80% training portion into final-training/validation subsets (note: different ratio from the spam example, but same overall idea).

### 1.18 Training & Evaluating the Random Forest Model

- **Training:** `RandomForestClassifier(random_state=...)`, fit on the multi-class training subset for reproducibility.
- **Validation evaluation:** accuracy, weighted precision/recall/F1 (weighted = accounts for class imbalance across the 5 categories), confusion matrix, classification report.
- **Final test evaluation:** same metrics computed on the held-out test set, which the model has never seen in any form (not even during hyperparameter choices).
- **Why both validation and test evaluation:** validation metrics guide tuning; test metrics give the unbiased, final performance read.

### 1.19 Malware Classification via Images

- **What:** Classifying malware **families** (e.g. Emotet, WannaCry) not by direct binary analysis, but by converting the binary into a grayscale image and training a CNN on that image.
- **Why images:**
  - Avoids handling potentially dangerous malicious binaries directly — safer for a learning environment.
  - Manual classification (static + dynamic analysis, reverse engineering) is slow; ML speeds this up.
- **How the image is built:** Each byte of the PE (Windows executable) binary (value 0–255) becomes one pixel's brightness — 0 = black, 255 = white, values between = corresponding gray shade. The image **losslessly** encodes the binary (it can be reconstructed exactly from the image).
- **Why it works as a classification signal:** Different malware families show visually distinct structural/byte patterns in their images.

### 1.20 The Malimg Dataset

- **What:** 9,339 grayscale malware images across **25 malware families**, organized into one folder per family.
- **Exploring class balance:** Plotting class (family) frequencies reveals over/under-represented families — important because skewed class distribution can bias the trained model; if performance (accuracy, false positive/negative counts) isn't good enough, rebalancing the dataset may be needed before retraining.

### 1.21 Preprocessing the Malware Image Dataset

- **Splitting:** `splitfolders.ratio(...)` creates `train`/`val`/`test` folders from the per-class image folders. This module uses an 80/20 train/test split (val folder left empty / unused here).
- **Image preprocessing for a CNN:**
  - **Resize:** all images to a uniform size (module uses 75×75) — CNNs require consistent input dimensions.
  - **Normalize:** standardize pixel value distributions (ImageNet-style mean/std here), which makes training easier/more stable.
  - **Effect:** some fine detail is lost in the resized/normalized version vs. the raw image, but overall structure remains roughly visible.
- **DataLoaders:** wrap datasets for iteration during training/inference; batch size and `num_workers` (parallel loading) affect throughput.
- **Dynamic class count:** reading `n_classes` from the dataset itself (rather than hardcoding 25) lets the same code keep working if classes are added/removed later.

### 1.22 The Model — Transfer Learning with ResNet50

- **What:** ResNet50 — a 50-layer-deep CNN (~23M parameters) from the 2015 ResNet paper, strong at image classification.
- **Why start from pretrained weights:** Training from scratch would take **days to weeks**; starting from pretrained weights and fine-tuning saves enormous time.
- **Freezing layers:** All ResNet layers except the final one are **frozen** (weights don't update during training) — only the final layer's weights change.
  - **Trade-off:** faster training, but potentially lower ceiling on accuracy, a reasonable trade-off for a "simple proof-of-concept experiment" per the source.
- **Final layer customization:** Replaced to match the number of target classes, with a small added hidden layer (`HIDDEN_LAYER_SIZE = 1000`) and ReLU activation before the final class-count output layer.

### 1.23 Training & Evaluating the Malware CNN

- **Loss function:** `CrossEntropyLoss` (standard for multi-class classification).
- **Optimizer:** `Adam`.
- **Training loop per epoch:** forward pass → compute loss → backward pass (`loss.backward()`) → optimizer step → track running loss/accuracy.
- **Evaluation mode:** `model.eval()` + `torch.no_grad()` disables gradient tracking for inference, saving memory/compute and preventing accidental weight updates.
- **Result from the source's run:** Training accuracy climbed from ~57% (epoch 1) to ~96% (epoch 10), while final **test/inference accuracy was 88.54%** — illustrating that training accuracy can outpace generalization (test) accuracy, consistent with the frozen-layers / speed-over-performance trade-off made earlier.
- **Saving the model:** `torch.jit.script(model)` then `.save(path)` — different from joblib, since this is a PyTorch model, not a scikit-learn pipeline.

---

## 2. Important Terminology

| Term | Short definition |
|---|---|
| Miniconda / conda | Minimal Python+package-manager installer / the package & environment manager itself |
| Virtual environment | Isolated space for project-specific dependencies |
| JupyterLab / kernel | Web-based interactive coding environment / the process executing its code and holding state |
| Stateful notebook | Variables/imports persist across cells as long as the kernel runs |
| Bag-of-words | Representing text as counts of terms, ignoring most word order |
| Unigram / Bigram | Single word / pair of consecutive words as a feature unit |
| `min_df` / `max_df` | Minimum/maximum document frequency thresholds for keeping a term |
| Tokenization | Splitting text into word/sub-word units |
| Stop words | Common low-information words removed during text preprocessing |
| Stemming | Reducing words to a base/root form |
| Naive Bayes (Multinomial) | Bayes'-theorem classifier assuming feature independence; here, suited to word-count text features |
| `alpha` (smoothing) | NB hyperparameter preventing zero probabilities for unseen features |
| Pipeline (scikit-learn) | Chains preprocessing + model steps into one consistent object |
| `GridSearchCV` | Exhaustive search over hyperparameter combinations with cross-validation |
| joblib | Library for serializing/deserializing Python objects (e.g. trained models) to disk |
| Random Forest | Ensemble of decision trees combined via voting/averaging |
| Bootstrapping (ensemble sense) | Sampling training subsets with replacement to train separate trees |
| NSL-KDD | Refined KDD Cup 1999 dataset for intrusion detection benchmarking |
| One-hot encoding | Binary indicator columns per category, avoiding implied order |
| Log transform (`log1p`) | Compresses skewed numeric distributions, avoids log(0) |
| Imputation | Replacing missing/invalid values with estimated ones |
| KNNImputer / IterativeImputer | Advanced imputers using feature relationships |
| Weighted precision/recall/F1 | Metric averaging that accounts for class imbalance across multiple classes |
| Byteplot | A grayscale image where each pixel brightness encodes one byte of a binary |
| Malware family | A category of malware sharing characteristics (behavior, propagation, traits) |
| CNN | Convolutional Neural Network, used here for the malware image classifier |
| ResNet50 | A 50-layer pretrained CNN used as the malware classifier's backbone |
| Transfer learning | Starting from pretrained weights and fine-tuning for a new task |
| Frozen layers | Layers whose weights are not updated during training |
| `DataLoader` / `ImageFolder` | PyTorch utilities for batching data / loading images organized in per-class folders |
| Epoch | One full pass through the training dataset |
| `CrossEntropyLoss` | Standard loss function for multi-class classification |
| `model.eval()` / `torch.no_grad()` | Switches a PyTorch model to inference mode / disables gradient tracking |

---

## 3. Commands and Code Reference

### 3.1 Environment Setup

#### Install Miniconda

**Windows (via Scoop)**
```powershell
Set-ExecutionPolicy RemoteSigned -scope CurrentUser
irm get.scoop.sh | iex
scoop bucket add extras
scoop install miniconda3
conda --version
```

**macOS (via Homebrew)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install --cask miniconda
conda --version
```

**Linux**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -u
eval "$(/home/$USER/miniconda3/bin/conda shell.$(ps -p $$ -o comm=) hook)"
conda --version
```
- **Purpose:** Install conda and verify.
- **`-b -u`** (Linux installer): batch/silent mode, update if already present.
- **Expected result:** `conda --version` prints something like `conda 24.9.2`.

#### Initialize conda for your shell
```bash
conda init
conda config --add channels defaults
conda config --add channels conda-forge
conda config --add channels nvidia   # only if you have an nvidia GPU
conda config --add channels pytorch
conda config --set channel_priority strict
```
- **Purpose:** Lets your shell use `conda activate`/commands; sets package channel priorities.
- **When to use:** Once, right after installing Miniconda. Requires reopening your terminal afterward.

#### Stop auto-activating the base environment
```bash
conda config --set auto_activate_base false
```
- **Purpose:** Prevents the `(base)` prefix from appearing in every new terminal.

#### Create, activate, deactivate a virtual environment
```bash
conda create -n ai python=3.11
conda activate ai
conda deactivate
```
- **Purpose:** Isolate this module's dependencies from your system Python and other projects.
- **Expected result:** Prompt shows `(ai)` once activated; disappears after `conda deactivate`.

#### Install core packages
```bash
conda install -y numpy scipy pandas scikit-learn matplotlib seaborn transformers datasets tokenizers accelerate evaluate optimum huggingface_hub nltk category_encoders
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install requests requests_toolbelt
```
- **Purpose:** Install everything needed for this module's three models.
- **Flag:** `-y` auto-confirms the install.
- **`-c pytorch -c nvidia`:** specify the channels for GPU-enabled PyTorch packages.
- **Note:** `pytorch-cuda=12.4` assumes an NVIDIA GPU setup; **⚠ Verify** this version against your own CUDA/driver setup before running — the module doesn't explain what to do on CPU-only or non-NVIDIA systems beyond the general advice to use the Playground VM.

#### Keep packages updated
```bash
conda update --all
```
- **Purpose:** Updates conda-managed packages in the active environment only (not pip-installed ones).

#### Install and launch JupyterLab
```bash
conda install -y jupyter jupyterlab notebook ipykernel
jupyter lab
```
- **Purpose:** Install and start the notebook interface (opens a browser tab automatically).
- **When to use:** Run from inside the `ai` environment.

### 3.2 Spam Classifier Pipeline

#### Download and extract the SMS Spam Collection dataset
```python
import requests
import zipfile
import io

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
response = requests.get(url)
if response.status_code == 200:
    print("Download successful")
else:
    print("Failed to download the dataset")

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")
    print("Extraction successful")

import os
extracted_files = os.listdir("sms_spam_collection")
print("Extracted files:", extracted_files)
```
- **Expected result:** A `sms_spam_collection/` directory containing `SMSSpamCollection`.

#### Load and inspect
```python
import pandas as pd

df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)

print(df.head())
print(df.describe())
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Duplicate entries:", df.duplicated().sum())
df = df.drop_duplicates()
```
- **Purpose:** Load the tab-separated file, confirm structure, check for missing/duplicate entries, and clean duplicates.

#### Preprocess the text
```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Lowercase
df["message"] = df["message"].str.lower()

# Strip non-essential characters, keep $ and !
import re
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))

# Tokenize
from nltk.tokenize import word_tokenize
df["message"] = df["message"].apply(word_tokenize)

# Remove stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])

# Stem
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])

# Rejoin into strings
df["message"] = df["message"].apply(lambda x: " ".join(x))
```
- **How it works:** Each line transforms the `message` column in place, in the order described in §1.11.
- **When to use:** Before feature extraction, and identically again on any new message at inference time.

#### Feature extraction with CountVectorizer
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["message"])
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)
```
- **Expected result:** `X` is a sparse numeric feature matrix; `y` is a binary label vector (1=spam, 0=ham).

#### Build pipeline, tune, and train
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])

param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
grid_search.fit(df["message"], y)

best_model = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params_)
```
- **Purpose:** Combine vectorization + classification into one reusable object, then find the best `alpha` via 5-fold CV optimizing F1.
- **Expected result:** `best_model` is a fitted `Pipeline`; `grid_search.best_params_` shows the chosen `alpha`.

#### Evaluate on new messages
```python
import re
import numpy as np

def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
]

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
```
- **Purpose:** Run identical preprocessing on new, unseen messages before prediction.
- **Expected result:** Each message gets a predicted label plus spam/not-spam probabilities (per source's sample output, around 0.00–1.00 range depending on confidence).

#### Save and reload the model
```python
import joblib

model_filename = "spam_detection_model.joblib"
joblib.dump(best_model, model_filename)

loaded_model = joblib.load(model_filename)
new_data_processed = [preprocess_message(msg) for msg in new_messages]
predictions = loaded_model.predict(new_data_processed)
```
- **Purpose:** Persist the trained pipeline; reload later without retraining.
- **When to use:** Before uploading to the evaluation portal (§3.5).

### 3.3 Network Anomaly Detection Pipeline

#### Download and load the NSL-KDD dataset
```python
import requests, zipfile, io

url = "https://cdn.services-k8s.prod.aws.htb.systems/content/modules/292/KDD_dataset.zip"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.')
```
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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

#### Create binary and multi-class targets
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
    if attack in dos_attacks:
        return 1
    elif attack in probe_attacks:
        return 2
    elif attack in privilege_attacks:
        return 3
    elif attack in access_attacks:
        return 4
    else:
        return 0

df['attack_map'] = df['attack'].apply(map_attack)
```

#### Encode categorical features, select numeric features, combine
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

#### Split into train/validation/test
```python
train_X, test_X, train_y, test_y = train_test_split(train_set, multi_y, test_size=0.2, random_state=1337)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1337)
```

#### Train the Random Forest
```python
rf_model_multi = RandomForestClassifier(random_state=1337)
rf_model_multi.fit(multi_train_X, multi_train_y)
```

#### Evaluate on validation set
```python
multi_predictions = rf_model_multi.predict(multi_val_X)
accuracy = accuracy_score(multi_val_y, multi_predictions)
precision = precision_score(multi_val_y, multi_predictions, average='weighted')
recall = recall_score(multi_val_y, multi_predictions, average='weighted')
f1 = f1_score(multi_val_y, multi_predictions, average='weighted')
print(f"Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

conf_matrix = confusion_matrix(multi_val_y, multi_predictions)
class_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Network Anomaly Detection - Validation Set')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.show()

print(classification_report(multi_val_y, multi_predictions, target_names=class_labels))
```

#### Final evaluation on test set
```python
test_multi_predictions = rf_model_multi.predict(test_X)
test_accuracy = accuracy_score(test_y, test_multi_predictions)
test_precision = precision_score(test_y, test_multi_predictions, average='weighted')
test_recall = recall_score(test_y, test_multi_predictions, average='weighted')
test_f1 = f1_score(test_y, test_multi_predictions, average='weighted')
print(f"Accuracy: {test_accuracy:.4f}  Precision: {test_precision:.4f}  Recall: {test_recall:.4f}  F1: {test_f1:.4f}")

test_conf_matrix = confusion_matrix(test_y, test_multi_predictions)
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Network Anomaly Detection')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.show()

print(classification_report(test_y, test_multi_predictions, target_names=class_labels))
```

#### Save the model
```python
import joblib
model_filename = 'network_anomaly_detection_model.joblib'
joblib.dump(rf_model_multi, model_filename)
```

### 3.4 Malware Image Classifier Pipeline

#### Download and unpack the Malimg dataset
```bash
wget https://www.kaggle.com/api/v1/datasets/download/ikrambenabd/malimg-original -O malimg.zip
unzip malimg.zip
```
- **Expected result:** A `malimg_paper_dataset_imgs/` directory with 25 subfolders (one per malware family).

#### Explore class distribution
```python
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_BASE_PATH = "./malimg_paper_dataset_imgs/"

dist = {}
for mlw_class in os.listdir(DATA_BASE_PATH):
    mlw_dir = os.path.join(DATA_BASE_PATH, mlw_class)
    dist[mlw_class] = len(os.listdir(mlw_dir))

classes = list(dist.keys())
frequencies = list(dist.values())

plt.figure()
sns.barplot(y=classes, x=frequencies, orient='h')
plt.title("Malware Class Distribution")
plt.xlabel("Malware Class Frequency")
plt.ylabel("Malware Class")
plt.show()
```
- **Purpose:** Spot over/under-represented malware families before training.
- **(Extra, styling omitted)** The source's version adds HTB-themed colors; omitted here for brevity since it doesn't affect the analysis.

#### Split into train/val/test folders
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
- **Expected result:** `./newdata/train/`, `./newdata/val/` (empty here), `./newdata/test/` — roughly 7459 train / 0 val / 1880 test files in the source's run.

#### Preprocessing + DataLoaders (reusable function)
```python
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
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
- **Purpose:** Resize/normalize images, batch them, and report the live class count.

#### Model: ResNet50-based classifier
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

#### Training and evaluation functions
```python
import torch
import time

def compute_accuracy(n_correct, n_total):
    return round(100 * n_correct / n_total, 2)

def train(model, train_loader, n_epochs, verbose=False):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    training_data = {"accuracy": [], "loss": []}

    for epoch in range(n_epochs):
        running_loss = 0
        n_total = 0
        n_correct = 0
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

def save_model(model, path):
    model_scripted = torch.jit.script(model)
    model_scripted.save(path)

def predict(model, test_data):
    model.eval()
    with torch.no_grad():
        output = model(test_data)
        _, predicted = torch.max(output.data, 1)
    return predicted

def evaluate(model, test_loader):
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            predicted = predict(model, data)
            n_total += target.size(0)
            n_correct += (predicted == target).sum().item()
    return compute_accuracy(n_correct, n_total)
```

#### Plotting helpers
```python
import matplotlib.pyplot as plt

def plot(data, title, label, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data)+1), data, label=label)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_training_accuracy(training_data):
    plot(training_data['accuracy'], "Training Accuracy", "Accuracy", "Epoch", "Accuracy (%)")

def plot_training_loss(training_data):
    plot(training_data['loss'], "Training Loss", "Loss", "Epoch", "Loss")
```

#### Running the full pipeline
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
- **Expected result (per source's run):** Training accuracy rising from ~57% to ~96% over 10 epochs; final test/inference accuracy **88.54%**. Actual numbers vary with the random train/test split and any parameter changes.
- **When to use:** After confirming `load_datasets`, `MalwareClassifier`, `train`, `evaluate`, and `save_model` are all defined.

### 3.5 Model Evaluation Portal (Pattern Used for All Three Models)

Each model is uploaded to a local evaluation API on the Playground VM, which returns a flag on success. The pattern is identical across models — only the port and filename change:

| Model | Port | File uploaded |
|---|---|---|
| Spam Classifier | 8000 | `spam_detection_model.joblib` |
| Network Anomaly Detector | 8001 | `network_anomaly_detection_model.joblib` |
| Malware Classifier | 8002 | `malware_classifier.pth` |
| Skills Assessment model | 5000 | `skills_assessment.joblib` |

**Generic upload script**
```python
import requests
import json

url = "http://localhost:PORT/api/upload"
model_file_path = "MODEL_FILENAME"

with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

print(json.dumps(response.json(), indent=4))
```
- **Purpose:** POST the trained model file to the local evaluation server; prints the JSON response (which includes a flag on success).
- **When run from your own machine instead of the Playground VM's Jupyter:** connect the HTB VPN, spawn the VM, then browse to `http://<VM-IP>:PORT/` to use the web upload portal instead of the script.
- **Note from source (malware model specifically):** a single training epoch on the Playground VM may take up to ~10 minutes; **3 epochs** were stated as sufficient to reach the required accuracy there; evaluating an uploaded model can take up to 2 minutes.

---

## 4. Practical Workflow (General Pattern Across All Three Models)

1. **Set up the environment:** Miniconda → create/activate an environment → install packages → launch JupyterLab (or use the Playground VM).
2. **Acquire the dataset:** download (via `requests`/`wget`), extract (`zipfile`/`unzip`), confirm file listing.
3. **Load into a working structure:** `pd.read_csv` for tabular/text data, `ImageFolder` for image data.
4. **Explore:** `head()`, `info()`, `describe()`, `isnull().sum()`, duplicate checks, class-distribution plots.
5. **Clean/preprocess:**
   - Tabular: validate columns, standardize invalid markers to `NaN`, impute or drop, encode categoricals, transform skewed numerics.
   - Text: lowercase → strip chars → tokenize → remove stopwords → stem → rejoin.
   - Images: resize → tensor conversion → normalize.
6. **Extract features:** vectorize text (CountVectorizer) or combine encoded + numeric columns (tabular) or rely on the CNN to learn features directly (images).
7. **Split data:** train / validation / test, with a fixed `random_state` for reproducibility.
8. **Build and train the model:** Naive Bayes (+ `Pipeline` + `GridSearchCV`), Random Forest, or a transfer-learning CNN (ResNet50 with frozen layers).
9. **Evaluate:** accuracy, precision, recall, F1 (weighted for multi-class), confusion matrix, classification report — on validation first, then test.
10. **Save the model:** `joblib.dump` (scikit-learn) or `torch.jit.script(...).save(...)` (PyTorch).
11. **Deploy/verify:** upload to the evaluation portal; confirm it meets the required performance criteria.

### Common Mistakes
- Running notebook cells out of order and trusting stale variable state.
- Forgetting to apply the **exact same preprocessing** to new/inference data as was used in training.
- Mixing `conda` and `pip` installs carelessly, risking dependency conflicts.
- Judging a model only on training accuracy, ignoring the gap to test/inference accuracy (seen directly in the malware CNN example: ~96% train vs. 88.54% test).
- Dropping too much data instead of imputing when the dataset is already small (demo dataset dropped to only 77 rows).
- Using **plain accuracy** as the sole metric on an imbalanced dataset (classic spam-filter pitfall from §1.8).
- Forgetting `model.eval()` / `torch.no_grad()` during inference, wasting compute and risking inadvertent training-mode behavior (e.g. dropout/batchnorm differences).
- Not checking class balance in a multi-class image dataset before training.

### Troubleshooting
- **`conda` not found after install:** reopen your terminal; confirm `conda init` ran.
- **Package conflicts:** prefer conda-first installs; use pip only for packages conda lacks, and expect to manage those separately.
- **Notebook behaving unexpectedly:** restart the kernel, then re-run cells top to bottom.
- **Model trains well but evaluates poorly:** check that validation/test data went through identical preprocessing; check for data leakage between splits; check class balance.
- **Playground VM training is slow:** expected — the source notes up to ~10 minutes per epoch there; your own hardware (per §1.1's RAM/CPU guidance) will likely be faster.

---

## 5. Quick Revision Sheet

### 5.1 Top 10 Things to Remember
1. Miniconda + conda environments isolate and simplify ML dependency management; `conda init` wires up your shell, `conda create -n NAME python=X.Y` + `conda activate` makes a project environment.
2. Jupyter notebooks are **stateful** — cell execution order, not just code order, determines what variables hold.
3. A "good" dataset is relevant, complete, consistent, high-quality, representative, balanced, and appropriately sized.
4. Handling bad data: **drop** (simple, may lose a lot) vs **impute** (keeps more data; simple strategies for quick fixes, KNN/Iterative for relationship-aware fixes).
5. Text pipeline order: lowercase → strip punctuation (keep `$`/`!`) → tokenize → remove stopwords → stem → rejoin, then vectorize (CountVectorizer, unigrams+bigrams).
6. Accuracy can be misleading under class imbalance; precision/recall/F1 (and weighted variants for multi-class) give a fuller picture; trade-offs depend on context (threat detection favors recall; limited-resource settings favor precision).
7. Random Forests = many decision trees (bootstrapped samples, random feature subsets per split) combined by majority vote/averaging; good for anomaly detection by learning "normal" and flagging poor fits.
8. NSL-KDD supports both binary (normal/attack) and multi-class (DoS/Probe/Privilege/Access) intrusion detection targets.
9. Malware images are lossless byteplots (1 pixel = 1 byte, grayscale); CNNs (here, a frozen-backbone ResNet50 with a custom final layer) classify malware families from these images instead of raw binaries.
10. Save scikit-learn models with `joblib.dump`/`joblib.load`; save PyTorch models with `torch.jit.script(...).save(...)`.

### 5.2 Compact Command/Code Reference

| Task | Command / Code |
|---|---|
| Create+activate env | `conda create -n ai python=3.11` / `conda activate ai` |
| Install core packages | `conda install -y numpy scipy pandas scikit-learn matplotlib seaborn ...` |
| Launch Jupyter | `jupyter lab` |
| Load CSV | `pd.read_csv("file.csv")` |
| Check missing values | `data.isnull().sum()` |
| Impute (simple) | `SimpleImputer(strategy='median'/'most_frequent')` |
| Impute (advanced) | `KNNImputer(n_neighbors=5)` |
| One-hot encode | `OneHotEncoder(...)` or `pd.get_dummies(df[cols])` |
| Log-transform skew | `np.log1p(df["col"])` |
| Train/val/test split | `train_test_split(X, y, test_size=..., random_state=...)` |
| Vectorize text | `CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))` |
| Build pipeline | `Pipeline([("vectorizer", v), ("classifier", c)])` |
| Hyperparameter search | `GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")` |
| Train Random Forest | `RandomForestClassifier(random_state=...).fit(X, y)` |
| Evaluate (weighted) | `precision_score(y, pred, average='weighted')` (same pattern for recall/F1) |
| Confusion matrix plot | `sns.heatmap(confusion_matrix(y, pred), annot=True, fmt='d')` |
| Save sklearn model | `joblib.dump(model, "model.joblib")` |
| Load sklearn model | `joblib.load("model.joblib")` |
| Resize+normalize images | `transforms.Compose([Resize, ToTensor, Normalize])` |
| Load image folder dataset | `ImageFolder(root=..., transform=...)` |
| Batch data | `DataLoader(dataset, batch_size=..., shuffle=...)` |
| Freeze pretrained layers | `for p in model.parameters(): p.requires_grad = False` |
| Train step (PyTorch) | `loss.backward(); optimizer.step()` after `optimizer.zero_grad()` |
| Inference mode | `model.eval()` + `with torch.no_grad():` |
| Save PyTorch model | `torch.jit.script(model).save("model.pth")` |
| Upload model for evaluation | `requests.post(url, files={"model": open(path,"rb")})` |

### 5.3 Common Mistakes to Avoid
- Treating Jupyter as stateless — re-run cells in order after any edit.
- Skipping identical preprocessing between training and inference.
- Relying on plain accuracy for imbalanced classification tasks.
- Dropping (instead of imputing) when data is scarce.
- Ignoring class imbalance in a multi-class image/tabular dataset before training.
- Forgetting `model.eval()`/`no_grad()` at inference time in PyTorch.
- Confusing `joblib` (scikit-learn) save/load with PyTorch's `torch.jit.script(...).save(...)`.

### 5.4 Revision Questions

1. **Why use Miniconda instead of a plain Python install for this module?**
   Better performance for data science/ML tasks, simplified dependency management via conda, and environment isolation between projects.
2. **What does it mean that a Jupyter notebook is "stateful," and why does it matter?**
   Variables/imports persist across cells as long as the kernel runs; running cells out of order can produce results that don't match the visible code order.
3. **Name two strategies for handling invalid/missing dataset entries, and a trade-off for each.**
   Dropping (simple, clean, but can lose a lot of data) vs imputing (preserves more data, but estimates may not reflect reality; advanced imputers like KNN account for feature relationships).
4. **In the spam pipeline, why keep `$` and `!` when stripping other punctuation?**
   They can carry spam-relevant signal (money amounts, urgency/emphasis).
5. **What do `min_df` and `max_df` do in CountVectorizer?**
   `min_df` filters out terms that are too rare (appear in too few documents); `max_df` filters out terms that are too common (appear in too high a proportion of documents).
6. **Why might accuracy alone mislead you about a spam classifier's quality?**
   With a heavily imbalanced dataset (e.g. 99% legitimate), a model that always predicts "legitimate" gets 99% accuracy while catching zero spam.
7. **What are the three key concepts behind building a Random Forest?**
   Bootstrapping (sampling with replacement), random feature subsets per split, and majority-vote/averaging across trees.
8. **How does a Random Forest get used for anomaly detection specifically?**
   It's trained on data representing normal conditions; new points that fit poorly or get low-confidence predictions are flagged as potential anomalies.
9. **How is a malware image constructed from a binary, and why does it work as a classification signal?**
   Each byte (0–255) becomes one pixel's grayscale brightness; the image losslessly encodes the binary, and different malware families show visually distinct structural patterns.
10. **Why freeze most of ResNet50's layers when fine-tuning it for malware classification?**
    It massively speeds up training (avoiding the days/weeks needed from scratch) at some cost to potential peak accuracy — an accepted trade-off for a proof-of-concept.

### 5.5 Two-Minute Final Summary

This module walks the full ML lifecycle three times. Setup: Miniconda + conda environments isolate and manage dependencies; JupyterLab provides a stateful, cell-by-cell coding environment — remember that execution order, not code order, determines what state you see.

**Spam classifier:** SMS Spam Collection dataset → text preprocessing (lowercase, strip punctuation but keep `$`/`!`, tokenize, remove stopwords, stem, rejoin) → bag-of-words features via CountVectorizer (unigrams+bigrams, `min_df`/`max_df` filtering) → MultinomialNB inside a scikit-learn Pipeline, tuned with GridSearchCV on `alpha` using F1 scoring → evaluate with identical preprocessing on new messages → save with joblib.

**Network anomaly detector:** NSL-KDD dataset → binary (`attack_flag`) and multi-class (`attack_map`: DoS/Probe/Privilege/Access) targets → one-hot encode categorical fields, keep numeric/statistical features → RandomForestClassifier, evaluated with weighted precision/recall/F1 and confusion matrices on both validation and test sets → save with joblib.

**Malware classifier:** Malimg dataset of 25 malware families represented as lossless grayscale byteplots (1 pixel = 1 byte) → check class balance → split into train/test → resize/normalize images for a CNN → transfer learning with a frozen ResNet50 backbone plus a custom final layer → train with CrossEntropyLoss + Adam over several epochs, tracking accuracy/loss → evaluate on held-out test data (watch for a training-vs-test accuracy gap) → save with `torch.jit.script(...).save(...)`.

Across all three: always split data for unbiased evaluation, never trust accuracy alone (especially under class imbalance), apply identical preprocessing between training and inference, and remember accuracy/precision/recall trade-offs depend on the real-world cost of false positives vs. false negatives.

---

## 6. Model Evaluation & Skills Assessment: Approach Only

The module has four "submit your model" checkpoints (spam, network anomaly, malware image, and a skills-assessment IMDB sentiment model), each requiring a trained model uploaded to a local evaluation API that returns a flag on success. These notes contain no flags or specific accuracy thresholds. Use the general workflow in §4 and the upload pattern in §3.5 — build, evaluate until the metrics look solid, save the model in the correct format for that step, and upload it to the matching port.
