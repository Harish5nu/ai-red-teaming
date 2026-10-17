---
tags: [htb-academy, ai, machine-learning, revision]
module: Fundamentals of AI
---

# Fundamentals of AI / Machine Learning: Revision Notes

> **Legend:** Unmarked = from your module text. **(Extra)** = my own added explanation or example. **⚠ Verify** = unclear or inconsistent in the source.
> **Note on format:** This module is theory-only. It has no CLI tools, so section 3 covers **formulas and the one code snippet** instead of commands.
> **Why it matters for web pentesting (Extra):** The module itself lists cybersecurity uses: threat detection, intrusion prevention, malware analysis, and network-security anomaly detection. Knowing how these models learn also helps you reason about how they can fail or be abused, but that application is outside this module's text.

---

## 1. Main Concepts: What, Why, How

### 1.1 AI vs ML vs DL

- **What:**
  - **AI:** Broad field of building systems that do tasks needing human intelligence (language, perception, decisions, problem solving). Key areas: NLP, Computer Vision, Robotics, Expert Systems.
  - **ML:** Subfield of AI where systems **learn from data** without explicit programming, using statistical techniques to find patterns.
  - **DL:** Subfield of ML using **neural networks with many layers** to learn features from complex data (images, audio, text).
- **Why:** ML and DL provide the learning and adaptation that underpins many AI systems.
- **How (nesting):** AI ⊃ ML ⊃ Neural Nets ⊃ DL (concentric circles in the module).
- **Three ML types:**

| Type | Data | Examples |
|---|---|---|
| **Supervised** | Labeled | Image classification, spam detection, fraud prevention |
| **Unsupervised** | Unlabeled | Customer segmentation, anomaly detection, dimensionality reduction |
| **Reinforcement** | Environment feedback (rewards/penalties) | Game playing, robotics, autonomous driving |

- **DL characteristics:** Hierarchical feature learning, end-to-end learning (raw input → output), scalability.
- **Common DL networks:** CNNs (images/video), RNNs (sequences), Transformers (NLP, self-attention).
- **Example from source:** Train on images labeled "cat" or "dog"; the model predicts the label for a new image.

---

### 1.2 Supervised Learning (core ideas)

- **What:** Learn from labeled examples to predict labels for new, unseen data.
- **Why:** Foundation for many practical ML applications.
- **How:**
  1. Collect **training data** (features + labels).
  2. Train a **model** by adjusting parameters to minimize prediction error.
  3. **Predict** on new data.
  4. **Evaluate** performance and check generalization.
- **Two problem types:**
  - **Classification:** predict a category (spam / not spam).
  - **Regression:** predict a continuous value (house price).
- **Key concepts:**

| Concept | Meaning |
|---|---|
| Features | Input variables (size, bedrooms, location) |
| Labels | Known outcomes ("correct answers") |
| Model | Mathematical mapping from features to labels |
| Prediction vs inference | Prediction = actionable output; inference = broader, understanding structure/feature importance/coefficients |
| Generalization | Doing well on unseen data |
| Overfitting | Memorizes training data including noise, so it generalizes poorly |
| Underfitting | Too simple; poor on training and new data |
| Cross-validation | Split into folds, train on some and validate on the rest, to estimate generalization |
| Regularization | Penalty added to the loss to discourage over-complex models: **L1** (absolute value of coefficients), **L2** (square of coefficients) |

- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score (harmonic mean of precision and recall).

---

### 1.3 Linear Regression

- **What:** Predicts a **continuous** target by fitting a linear relationship to one or more predictors.
- **Why:** Simple, interpretable baseline for numeric prediction.
- **How:**
  1. Model the relationship: simple `y = mx + c`; multiple `y = b0 + b1x1 + ... + bnxn`.
  2. Use **Ordinary Least Squares (OLS)** to find coefficients:
     1. Compute residuals (actual − predicted).
     2. Square them.
     3. Sum them (RSS).
     4. Adjust coefficients to minimize RSS.
- **Assumptions:** Linearity, Independence, Homoscedasticity (constant error variance), Normality of errors.
- **Example:** Predict house price from size.

---

### 1.4 Logistic Regression

- **What:** Despite the name, a **classification** algorithm (binary outcomes) that outputs a probability from 0 to 1.
- **Why:** Simple binary classification (spam / not spam, will click an ad / won't).
- **How:**
  1. Compute a linear combination `z = m1x1 + ... + mnxn + c`.
  2. Pass it through the **sigmoid**: `P(x) = 1 / (1 + e^-z)` (S-shaped, maps any value to 0–1).
  3. Compare to a **threshold** (often 0.5, adjustable): above = positive class, below = negative.
- **Decision boundary:** The separator between classes; a line in 2D, a **hyperplane** in higher dimensions.
- **Assumptions:** Binary outcome, linearity of log-odds, little multicollinearity, large sample size.
- **Example from source:** Spam if the score exceeds the threshold. A 0.8 spam probability with a 0.5 threshold is classified as spam; at a 0.6 threshold it is still spam, but borderline emails shift.

---

### 1.5 Decision Trees

- **What:** Tree of simple decision rules for classification or regression.
- **Why:** Intuitive and interpretable; minimal data assumptions (no linearity/normality assumption, fairly robust to outliers).
- **Components:** Root node (whole dataset), internal nodes (feature tests), leaf nodes (final prediction).
- **How:**
  1. Pick the best feature to split on using **Gini impurity**, **entropy** or **information gain**.
  2. Split the data into subsets by feature value.
  3. Recurse on each subset.
  4. Stop when: max depth reached, minimum data points in a node, or the node is pure.
- **Example from source:** "Play tennis?" using Outlook, Temperature, Humidity, Wind.
- **Worked numbers (from source):** A set with 30 class A and 20 class B has Gini = 0.48 and entropy ≈ 0.971. The example split on feature F gives information gain ≈ 0.020.

---

### 1.5b Naive Bayes

- **What:** Probabilistic classifier based on **Bayes' theorem** with the "naive" assumption that features are **conditionally independent given the class**.
- **Why:** Simple, efficient, performs surprisingly well in practice (spam filtering, sentiment analysis).
- **How:**
  1. Compute **prior** probability of each class.
  2. Compute **likelihood** of each feature given each class.
  3. Apply Bayes' theorem to get the **posterior** for each class.
  4. Predict the class with the highest posterior.
- **Types:** **Gaussian** (continuous, bell-curve), **Multinomial** (discrete counts, e.g. word frequencies), **Bernoulli** (binary presence/absence).
- **Example from source (Bayes):** Disease prevalence 1%, test 95% sensitive, 5% false positive rate → P(disease | positive) ≈ **16.1%**. Lesson: low base rates make even accurate tests give many false positives.

---

### 1.6 Support Vector Machines (SVM)

- **What:** Supervised algorithm for classification/regression that finds the **hyperplane maximizing the margin** between classes.
- **Why:** Handles high-dimensional data and non-linear relationships; few distributional assumptions; relatively robust to outliers.
- **How:**
  1. Find hyperplane `w · x + b = 0` that maximizes the margin.
  2. **Support vectors** = nearest points to the hyperplane; they define it.
  3. For non-linear data, use the **kernel trick**: map data into a higher-dimensional space where it is linearly separable.
- **Kernels:** Polynomial, RBF (Gaussian; popular and versatile), Sigmoid.
- **Optimization form:** Minimize ½‖w‖² subject to yᵢ(w·xᵢ + b) ≥ 1.
- **⚠ Verify:** The source calls the margin "the separating hyperplane", which is loose wording. The hyperplane is the boundary; the margin is the distance to the nearest points.
- **Example:** Spam vs not-spam using frequency of "free" and "money"; image classification with an RBF kernel.

---

### 1.7 Unsupervised Learning (overview)

- **What:** Finds structure in **unlabeled** data.
- **Why:** Useful when labels are scarce, expensive or unavailable.
- **Three categories:** Clustering, Dimensionality reduction, Anomaly detection.
- **Core ideas:**

| Concept | Meaning |
|---|---|
| Similarity measures | Euclidean (straight-line), Cosine (angle between vectors), Manhattan (sum of absolute differences) |
| Clustering tendency | Whether data naturally forms clusters |
| Cluster validity | Cohesion (tightness inside a cluster) and separation (distance between clusters); silhouette score and Davies-Bouldin index |
| Dimensionality | Number of features; high dimensionality → "curse of dimensionality" (sparse data, less meaningful distances) |
| Intrinsic dimensionality | The underlying dimensionality, possibly lower than the feature count |
| Anomaly / outlier | A point deviating from the norm / a point far from the majority |
| Feature scaling | Min-Max scaling (fixed range), Standardization/Z-score (zero mean, unit variance) |

---

### 1.8 K-Means Clustering

- **What:** Partitions data into **K** non-overlapping clusters.
- **Why:** E.g. customer segmentation for targeted marketing.
- **How:**
  1. **Initialize:** pick K random points as centroids.
  2. **Assign:** each point to the nearest centroid (Euclidean distance).
  3. **Update:** recompute each centroid as the mean of its points.
  4. **Repeat** until centroids stop changing (or max iterations).
- **Choosing K:**
  - **Elbow Method:** run K-means for a range of K, plot WCSS (within-cluster sum of squares) vs K, pick the point where the decrease slows.
  - **Silhouette analysis:** score from −1 to 1 per point; choose the K with the highest average. Near 1 = well matched, near 0 = on a boundary, near −1 = probably wrong cluster.
  - Also use domain expertise, computational cost, interpretability.
- **Assumptions:** Roughly spherical, similar-sized clusters; **sensitive to feature scale** (standardize first); sensitive to outliers.

---

### 1.9 Principal Component Analysis (PCA)

- **What:** Dimensionality reduction that projects data onto new axes (**principal components**) capturing maximum variance.
- **Why:** Feature extraction, visualization, noise reduction.
- **How:**
  1. Standardize the data.
  2. Compute the covariance matrix.
  3. Compute eigenvectors and eigenvalues.
  4. Sort eigenvectors by eigenvalue (descending).
  5. Select the top **k**.
  6. Transform: `Y = X * V`.
- **Eigenvector/eigenvalue:** `A * v = λ * v` (v keeps its direction; λ is the scale factor). In PCA: `C * v = λ * v`, where eigenvectors give directions and eigenvalues give variance explained.
- **Solving methods:** Eigenvalue decomposition or SVD (more numerically stable).
- **Choosing k:** Plot explained variance ratio; keep enough components for e.g. 95% of variance.
- **Assumptions:** Linear relationships, correlated features, scale-sensitive (standardize).
- **Example from source:** Faces reduced to principal components capturing major variations (eigenfaces).

---

### 1.10 Anomaly Detection

- **What:** Finding data points that deviate significantly from normal behaviour.
- **Why:** Fraud, system failures, medical emergencies, network security.
- **Types of anomalies:** **Point** (single odd value), **Contextual** (odd only in context, e.g. 30°C in winter), **Collective** (a group that is odd together, e.g. many login attempts from unknown IPs).
- **Technique families:** Statistical (z-score, modified z-score, boxplots), clustering-based, ML-based.
- **Algorithms:**
  - **One-Class SVM:** learns a boundary around normal data; points outside are outliers (kernels handle non-linearity).
  - **Isolation Forest:** randomly partitions data; anomalies are "few and different", so they isolate in **shorter paths**. Score closer to 1 = more anomalous, around 0.5 = normal.
  - **Local Outlier Factor (LOF):** compares a point's local density to its k neighbours; higher LOF = more outlier-like. Good when density varies.

---

### 1.11 Reinforcement Learning (RL)

- **What:** An **agent** learns by interacting with an **environment**, receiving **rewards/penalties**, to learn a **policy** that maximizes cumulative reward.
- **Why:** Sequential decision-making in dynamic environments (games, robotics, driving).
- **How (loop):** Observe state → choose action → environment returns new state + reward → update the policy.
- **Model-based vs model-free:** Model-based learns a model of the environment and plans with it (like having a map). Model-free learns directly from experience (no map).
- **Concepts:**

| Term | Meaning |
|---|---|
| Agent | Learner/decision-maker |
| Environment | Everything the agent interacts with |
| State | Current situation snapshot |
| Action | Agent's move |
| Reward | Scalar feedback (positive, negative or zero) |
| Policy | Mapping from states to actions (deterministic or stochastic) |
| Value function | Expected long-term reward (state-value or action-value) |
| Discount factor γ | 0–1; near 0 = short-term focus, near 1 = long-term focus |
| Episodic vs continuous | Ends at a terminal state vs runs indefinitely |

---

### 1.12 Q-Learning

- **What:** Model-free, **off-policy** RL algorithm that learns Q-values (expected cumulative reward for an action in a state, then following the optimal policy).
- **Why:** Learns good policies without a model of the environment.
- **How:**
  1. Initialize the **Q-table** (rows = states, columns = actions), usually zeros.
  2. Choose an action (explore vs exploit).
  3. Take it; observe new state and reward.
  4. Update Q with the rule below.
  5. Move to the new state.
  6. Repeat until convergence or a stop condition.
- **Update rule:** `Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]`
- **Exploration vs exploitation:** **Epsilon-greedy** (random action with probability ε, otherwise the best known; high ε early, lower later) or **Softmax** (probabilities weighted by Q-values).
- **Assumptions:** Markov property; stationary environment.
- **Worked example (source):** α=0.1, γ=0.9, r=0.5, old Q=0.2, max next Q=1.0 → new Q = **0.32**.

---

### 1.13 SARSA

- **What:** Model-free, **on-policy** RL algorithm. Updates using the Q-value of the **action actually taken** next.
- **Why:** More conservative/safer behaviour; learns the value of the policy it follows, including exploration.
- **How:** Same loop as Q-learning, except you choose the next action `a'` using the current policy *before* the update, then set `s = s'`, `a = a'`.
- **Update rule:** `Q(s,a) <- Q(s,a) + α * (r + γ * Q(s',a') - Q(s,a))`
- **Q-learning vs SARSA:**

| | Q-learning | SARSA |
|---|---|---|
| Policy type | Off-policy | On-policy |
| Update uses | **max** Q of next state | Q of the **actual next action** |
| Character | More exploratory, finds optimal policy | More cautious, stable |

- **Tuning:** α (learning rate) high = fast but unstable; low = stable but slow. γ high = long-term focus. Techniques like grid search can tune them.

---

### 1.14 Deep Learning Building Blocks

- **What:** Multi-layer artificial neural networks (ANNs) that learn features automatically from raw data.
- **Why:** Solve complex problems (vision, speech, language) and reduce manual feature engineering.
- **Parts:**

| Part | Role |
|---|---|
| Layers | Input (receives data), Hidden (extract features), Output (result) |
| Weights | Connection strengths, adjusted during learning |
| Activation function | Adds non-linearity: Sigmoid (0–1), ReLU (0 for negatives, identity for positives), Tanh (−1 to 1), Softmax (probability distribution for multi-class output) |
| Loss function | Measures prediction error (MSE for regression, cross-entropy for classification) |
| Optimizer | Updates weights (SGD, Adam, RMSprop) |
| Hyperparameters | Set before training (learning rate, number of layers, neurons per layer) |
| Backpropagation | Computes gradients of the loss w.r.t. weights |

---

### 1.15 Perceptron

- **What:** Simplest neuron model: inputs × weights, summed, plus bias, through an activation function (step function) → binary output.
- **How:**
  1. Multiply each input by its weight.
  2. Sum them.
  3. Add the bias.
  4. Apply `f(x) = 1 if x > 0 else 0`.
- **Limitation:** Single-layer perceptrons only learn **linearly separable** problems; cannot solve **XOR**.
- **Worked example (source):** Weights 0.3, 0.2, −0.4, −0.2, bias 0.1; inputs (Sunny=0, Mild=1, High humidity=0, Weak wind=0) → sum 0.2 + 0.1 = 0.3 → output 1 (play).

---

### 1.16 Neural Networks (MLPs)

- **What:** Multi-layer perceptrons: input layer + one or more hidden layers + output layer.
- **Why:** Hidden layers with non-linear activations learn non-linear boundaries (so XOR becomes solvable) and hierarchical features.
- **Neuron:** Weighted sum + bias → activation function (not limited to a step function).
- **Output layer:** 1 neuron for binary classification; 1 neuron per class for multi-class.
- **Training:**
  1. **Forward pass** → prediction.
  2. **Calculate error** with the loss function.
  3. **Backward pass** → gradients through the chain rule.
  4. **Update** weights/biases using **gradient descent** (step against the gradient; size set by the learning rate).
  5. Repeat until the loss converges.

---

### 1.17 Convolutional Neural Networks (CNNs)

- **What:** Networks for grid-like data (images, video).
- **Why:** Capture spatial hierarchies of features; strong for image recognition, object detection, segmentation.
- **Layers:**
  - **Convolutional:** learnable filters slide over the input producing **feature maps** (edges, corners, textures).
  - **Pooling:** downsample feature maps (max/average) to cut computation and overfitting.
  - **Fully connected:** high-level reasoning and final prediction.
- **How:** Stack conv + pool layers, flatten, then fully connected layers.
- **Hierarchy:** Early layers = edges; middle = corners/patterns; deep = shapes/object parts.
- **Assumptions:** Grid structure, spatial feature hierarchy, feature **locality**, feature **stationarity** (weight sharing), enough data, normalized input.

---

### 1.18 Recurrent Neural Networks (RNNs)

- **What:** Networks for **sequential** data; they keep a hidden state ("memory") across steps.
- **How:** At each step, take the current input + previous hidden state → produce an output + updated hidden state.
- **Problem:** **Vanishing gradient**: during backpropagation through time (BPTT), gradients shrink as they pass back, so long-term dependencies are hard to learn.
- **Fixes:**
  - **LSTM:** memory cell with **input, forget, output** gates.
  - **GRU:** simpler, **update and reset** gates, often comparable performance and more efficient.
- **Bidirectional RNNs:** One RNN reads forward and another backward; combined for full context when the whole sequence is available.

---

### 1.19 Generative AI

- **What:** Models that create new content (text, images, music, code) resembling human-made output.
- **How:** Train on a large dataset → generate by sampling from the learned distribution → evaluate.
- **Model types:**

| Model | Idea |
|---|---|
| GAN | Generator vs discriminator compete |
| VAE | Learns a compressed latent representation to sample from |
| Autoregressive | Generates one element at a time from previous ones |
| Diffusion | Adds noise, then learns to reverse it |

- **Concepts:** Latent space (compressed representation), sampling, **mode collapse** (generator produces limited variety), overfitting.
- **Metrics:** Inception Score (IS), Fréchet Inception Distance (FID; lower is better), BLEU (text similarity to a reference).

---

### 1.20 Large Language Models (LLMs)

- **What:** Very large models trained on massive text to understand and generate language, based on **transformers**.
- **Characteristics:** Massive scale (billions to trillions of parameters), few-shot learning, contextual understanding.
- **How it works:**
  1. **Tokenization:** text → tokens (words, subwords or characters).
  2. **Embeddings:** tokens → vectors; similar meaning = closer in space.
  3. **Transformer** processes tokens **in parallel** (faster than RNNs).
  4. **Self-attention** scores how much each word attends to the others, capturing long-range dependencies.
  5. **Encoders** capture meaning; **decoders** generate output.
  6. **Training:** unsupervised on massive text, minimizing prediction error via gradient descent variants (needs GPUs/TPUs).
- **Example from source:** Prompt "Once upon a time, there was a cat named Whiskers." → the model continues the story word by word.

---

### 1.21 Diffusion Models

- **What:** Generative models that learn to **remove noise** to create data (especially images).
- **How:**
  1. **Forward process:** gradually add noise until the data becomes pure noise.
  2. **Reverse process:** a denoising network learns to predict and remove the noise step by step.
  3. **Training:** minimize `L = E[||ε - ε_pred||²]` (MSE between actual and predicted noise).
  4. **Sampling:** start from pure noise, denoise iteratively to get a sample.
  5. **Text-to-image:** a text encoder (e.g. Transformer or CLIP) converts the prompt to an embedding that conditions the denoising at every step.
- **Noise schedule:** Controls how much noise is added per step (e.g. linear: `β_t = β_min + (t/T)(β_max − β_min)`).
- **Denoising network:** Typically a CNN or transformer.
- **Assumptions:** Markov property, static data distribution, smoothness (not strict).

---

## 2. Important Terminology

| Term | Short definition |
|---|---|
| AI | Systems doing tasks needing human-like intelligence |
| ML | Learning from data without explicit programming |
| DL | ML with multi-layer neural networks |
| Supervised / Unsupervised / Reinforcement | Labeled / unlabeled / reward-based learning |
| Classification / Regression | Predict category / predict continuous value |
| Feature / Label | Input variable / known output |
| Model | Learned mapping from features to prediction |
| Training / Inference | Fitting the model / using it (and interpreting it) |
| Overfitting / Underfitting | Too tailored to training data / too simple |
| Generalization | Performing well on unseen data |
| Cross-validation | Fold-based estimate of generalization |
| Regularization (L1/L2) | Penalty on coefficient size to curb overfitting |
| Precision / Recall / F1 | Correct positives among predicted / among actual / balance of both |
| OLS / RSS | Least squares method / residual sum of squares |
| Residual | Actual − predicted |
| Homoscedasticity | Constant error variance |
| Multicollinearity | Highly correlated predictors |
| Sigmoid | S-shaped function mapping values into 0–1 |
| Decision boundary / Hyperplane | Separator between classes / its higher-dimensional form |
| Threshold probability | Cutoff for assigning the positive class |
| Gini / Entropy / Information gain | Impurity / disorder / reduction in entropy from a split |
| Prior / Likelihood / Posterior | Belief before / probability of evidence given class / updated belief |
| Support vector / Margin | Closest points to the hyperplane / distance to them |
| Kernel trick | Implicit mapping to higher dimensions to separate non-linear data |
| Cohesion / Separation | Tightness within clusters / distance between clusters |
| Centroid | Cluster center (mean) |
| WCSS | Within-cluster sum of squares |
| Silhouette score | Cluster fit measure, −1 to 1 |
| Curse of dimensionality | Sparse data and weak distances in high dimensions |
| Principal component | Direction of maximum variance |
| Eigenvector / Eigenvalue | Direction preserved by a transform / its scale factor |
| Anomaly / Outlier | Deviation from normal / far from the majority |
| Agent / Environment / State / Action / Reward | RL basics (see §1.11) |
| Policy | Strategy mapping states to actions |
| Q-value / Q-table | Expected return of a state-action pair / table storing them |
| On-policy / Off-policy | Learns the followed policy's value / learns the optimal policy's value |
| Epsilon-greedy | Random action with probability ε, else best-known |
| Learning rate α / Discount γ | Update step weight / future reward weight |
| Markov property | Next state depends only on current state and action |
| Neuron / Weight / Bias | Computation unit / connection strength / shifting term |
| Activation function | Adds non-linearity |
| Backpropagation | Computes gradients backward through the network |
| Gradient descent | Iteratively steps against the gradient to minimize loss |
| Epoch (Extra) | One pass through the training data |
| MLP | Multi-layer perceptron |
| XOR problem | Classic non-linearly-separable problem a single perceptron can't solve |
| Feature map / Pooling | Output of a filter / downsampling |
| Hidden state | RNN memory carried between steps |
| Vanishing gradient | Gradients shrink to near zero during training |
| LSTM / GRU | Gated RNNs for long-term dependencies |
| GAN / VAE / Diffusion | Generative model families |
| Latent space | Compressed hidden representation |
| Mode collapse | GAN generator outputs limited variety |
| Token / Embedding | Text unit / numeric vector for a token |
| Self-attention | Weighs relevance between words |
| Transformer | Parallel, attention-based architecture |
| Few-shot learning | Doing a new task from a few examples |
| Noise schedule | How much noise is added at each step |

---

## 3. Formulas and Code (no CLI commands in this module)

### 3.1 Math Notation Quick Reference (Section 2 of the module)

| Notation | Meaning |
|---|---|
| `x_t` | Variable at time step t |
| `x^n` | x to the power n |
| `‖v‖` | Norm (Euclidean: √(v₁²+...+vₙ²); L1: sum of absolute values; L∞: maximum absolute value) |
| `Σ` | Summation |
| `log2(x)`, `ln(x)`, `e^x`, `2^x` | Logarithms and exponentials (e.g. log2(8)=3, 2³=8) |
| `A*v`, `A*B` | Matrix-vector / matrix-matrix multiplication |
| `Aᵀ`, `A⁻¹`, `det(A)`, `tr(A)` | Transpose, inverse, determinant (non-zero → invertible), trace (sum of the diagonal) |
| `|S|`, `∪`, `∩`, `Aᶜ` | Cardinality, union, intersection, complement |
| `λ` | Eigenvalue |
| `P(x|y)` | Probability of x given y |
| `E[X]`, `Var(X)`, `σ(X)` | Expectation, variance, standard deviation |
| `Cov(X,Y)`, `ρ(X,Y)` | Covariance, correlation (−1 to 1) |

- **⚠ Verify:** The module's example `x_t = q(x_t | x_{t-2})` looks like a typo. The diffusion section uses `q(x_t | x_{t-1})`.

### 3.2 Key Formulas

| Topic | Formula | Purpose |
|---|---|---|
| Simple linear regression | `y = mx + c` | Predict a continuous value from one predictor |
| Multiple linear regression | `y = b0 + b1x1 + ... + bnxn` | Same, with several predictors |
| Sigmoid | `P(x) = 1 / (1 + e^-z)` | Convert a linear score to a probability |
| Gini | `1 − Σ pᵢ²` | Impurity of a set |
| Entropy | `−Σ pᵢ log2(pᵢ)` | Disorder of a set |
| Information gain | `Entropy(S) − Σ (|Sv|/|S|) · Entropy(Sv)` | Choose the best split |
| Bayes' theorem | `P(A|B) = P(B|A)·P(A) / P(B)` | Update belief with evidence |
| SVM hyperplane | `w·x + b = 0` | Decision boundary |
| SVM optimization | Minimize ½‖w‖² s.t. yᵢ(w·xᵢ+b) ≥ 1 | Maximize the margin |
| Euclidean distance | `d(x,y) = √Σ(xᵢ−yᵢ)²` | Distance for K-means |
| PCA eigen equation | `C·v = λ·v` | Find principal components |
| PCA transform | `Y = X·V` | Project the data |
| Isolation Forest score | `2^(−E(h(x))/c(n))` | Anomaly score |
| LOF | `(Σ lrd(o)/k) / lrd(p)` | Local outlier factor |
| lrd | `1 / (Σ reach_dist(p,o)/k)` | Local reachability density |
| Q-learning update | `Q(s,a) += α[r + γ·max Q(s',a') − Q(s,a)]` | Off-policy value update |
| SARSA update | `Q(s,a) += α[r + γ·Q(s',a') − Q(s,a)]` | On-policy value update |
| Diffusion loss | `L = E[‖ε − ε_pred‖²]` | Train the denoiser |
| Noise schedule (linear) | `β_t = β_min + (t/T)(β_max−β_min)` | Noise variance per step |

### 3.3 The One Code Snippet in the Module: Perceptron Decision

```python
def step_activation(x):
    """Step activation function."""
    return 1 if x > 0 else 0

# Input features
outlook = 0
temperature = 1
humidity = 0
wind = 0

# Weights and bias
w1 = 0.3
w2 = 0.2
w3 = -0.4
w4 = -0.2
b = 0.1

weighted_sum = (w1 * outlook) + (w2 * temperature) + (w3 * humidity) + (w4 * wind)
total_input = weighted_sum + b
output = step_activation(total_input)
print(f"Output: {output}")  # Output: 1 (Play Tennis)
```

- **Purpose:** Shows a perceptron making a binary decision.
- **How it works:**
  1. Multiply each input by its weight and sum.
  2. Add the bias.
  3. Apply the step function.
- **Expected result:** `Output: 1` for these inputs (the module's own comment says the same).
- **When to use it:** To understand how a single neuron computes its output. No libraries needed.

---

## 4. Practical Workflow (Choosing and Applying an ML Approach)

> **(Extra)** This workflow is my synthesis of the module's concepts, not a step list from the course.

1. **Define the problem.** Is there a label to predict?
   - Labeled + category → **classification** (Logistic Regression, Decision Tree, Naive Bayes, SVM).
   - Labeled + number → **regression** (Linear Regression, SVM regression).
   - No labels → **unsupervised** (K-Means, PCA, anomaly detection).
   - Sequential decisions with feedback → **reinforcement learning** (Q-learning, SARSA).
   - Images / sequences / text / generation → **deep learning** (CNN / RNN / Transformer / diffusion).
2. **Prepare the data.** Choose relevant features, **scale/standardize** (essential for K-means and PCA), normalize inputs for CNNs.
3. **Check assumptions.** E.g. linearity and homoscedasticity for linear regression; feature independence for Naive Bayes.
4. **Train** the model (adjust parameters to minimize loss or error).
5. **Evaluate.** Use accuracy, precision, recall, F1 (supervised); silhouette/elbow (clustering); FID/IS/BLEU (generative).
6. **Guard against overfitting.** Cross-validation, regularization, limits on tree depth, enough data.
7. **Tune hyperparameters.** Learning rate, K, depth, γ, ε, layers.
8. **Interpret results.** Use inference to understand feature importance or coefficients, not just predictions.

### Common Mistakes
- Using accuracy alone when classes are imbalanced (see the Bayes example: low prevalence → many false positives).
- Forgetting to standardize before K-means or PCA.
- Treating logistic regression as regression because of its name.
- Confusing prediction with inference.
- Picking K by gut feeling instead of elbow/silhouette plus domain knowledge.
- Confusing on-policy (SARSA) with off-policy (Q-learning).
- Expecting a single-layer perceptron to solve XOR.
- Ignoring the vanishing gradient problem in plain RNNs (use LSTM/GRU).
- Training and testing on the same data (overfitting goes unnoticed).

### Troubleshooting Tips
- **Great training score, poor new-data score:** overfitting → regularize, simplify, add data, cross-validate.
- **Poor on both:** underfitting → richer model or better features.
- **K-means results look odd:** check scaling, outliers and cluster shape (assumes spherical clusters).
- **Q-learning/SARSA unstable:** lower α, tune γ and ε.
- **GAN outputs lack variety:** possible mode collapse.
- **RNN can't learn long-range context:** switch to LSTM, GRU or a transformer.

---

## 5. Quick Revision Sheet

### 5.1 Top 10 Things to Remember
1. AI ⊃ ML ⊃ DL; ML has three types: supervised, unsupervised, reinforcement.
2. Supervised = labeled data; classification predicts categories, regression predicts numbers.
3. Overfitting = memorizing noise; fight it with regularization, cross-validation and more data.
4. Logistic regression is **classification**, using the sigmoid and a threshold.
5. Decision trees split using Gini, entropy or information gain; Naive Bayes assumes feature independence given the class.
6. SVMs maximize the margin; the kernel trick handles non-linear data.
7. K-means: initialize → assign → update → repeat; pick K with elbow/silhouette; standardize first.
8. PCA: standardize → covariance → eigen-decomposition → keep top-k components.
9. Q-learning is off-policy (max); SARSA is on-policy (actual next action); both balance exploration and exploitation.
10. Neural networks learn via forward pass → loss → backpropagation → gradient descent; CNNs for images, RNN/LSTM/GRU for sequences, transformers (self-attention) power LLMs, and diffusion models denoise noise into samples.

### 5.2 Compact Formula/Concept Table

| Algorithm | Type | Key idea | Key formula/term |
|---|---|---|---|
| Linear Regression | Supervised, regression | Best-fit line via OLS | `y = b0 + b1x1 + ...` |
| Logistic Regression | Supervised, classification | Sigmoid probability + threshold | `1/(1+e^-z)` |
| Decision Tree | Supervised, both | Recursive best splits | Gini / entropy / information gain |
| Naive Bayes | Supervised, classification | Bayes + independence | `P(A|B) = P(B|A)P(A)/P(B)` |
| SVM | Supervised, both | Maximize margin, kernels | `w·x + b = 0` |
| K-Means | Unsupervised, clustering | Centroids, assign/update | Euclidean distance, WCSS |
| PCA | Unsupervised, dim. reduction | Max-variance directions | `C·v = λ·v` |
| One-Class SVM / Isolation Forest / LOF | Anomaly detection | Boundary / short paths / low density | Isolation score, LOF |
| Q-Learning | RL, off-policy | Learn Q-values with max | `max Q(s',a')` |
| SARSA | RL, on-policy | Learn with the actual next action | `Q(s',a')` |
| MLP | DL | Layers + backprop + gradient descent | Activation, loss |
| CNN | DL | Convolution + pooling | Feature maps |
| RNN/LSTM/GRU | DL | Hidden state, gates | Vanishing gradient |
| Transformer/LLM | DL | Self-attention, tokens, embeddings | Parallel processing |
| Diffusion | Generative | Add noise, learn to reverse | `‖ε − ε_pred‖²` |

### 5.3 Common Mistakes to Avoid
- Mixing up regression vs classification (especially logistic regression).
- Skipping feature scaling for distance-based or variance-based methods.
- Using on-policy and off-policy interchangeably.
- Judging models only on training performance.
- Forgetting that perceptrons can't handle XOR without hidden layers.

### 5.4 Revision Questions

1. **What is the difference between supervised and unsupervised learning?**
   Supervised uses labeled data to predict outcomes; unsupervised finds structure in unlabeled data.
2. **What is overfitting and how can you reduce it?**
   The model learns noise in the training data and generalizes poorly. Reduce it with regularization, cross-validation, more data and simpler models.
3. **Why is logistic regression a classification algorithm?**
   It outputs a probability (via the sigmoid) that's compared with a threshold to choose a class.
4. **What does the "naive" assumption in Naive Bayes mean?**
   Features are conditionally independent given the class.
5. **What are support vectors?**
   The data points nearest the hyperplane that define the margin.
6. **How do you choose K in K-means?**
   Elbow method (WCSS), silhouette analysis, plus domain knowledge and cost/interpretability.
7. **What do eigenvalues represent in PCA?**
   The amount of variance explained by each principal component (eigenvector).
8. **How do Q-learning and SARSA differ?**
   Q-learning uses the max Q of the next state (off-policy); SARSA uses the Q of the action actually taken (on-policy).
9. **Why can't a single perceptron solve XOR, and what fixes that?**
   XOR isn't linearly separable; multi-layer networks with non-linear activations learn non-linear boundaries.
10. **What problem do LSTMs and GRUs address?**
    The vanishing gradient problem, which makes long-term dependencies hard for plain RNNs; gates control what's remembered or forgotten.

### 5.5 Two-Minute Final Summary

AI is the broad goal; ML is learning from data; DL is ML with deep neural networks. Supervised learning uses labeled data for classification (categories) or regression (numbers). Linear regression fits a line by minimizing squared errors. Logistic regression pushes a linear score through a sigmoid and a threshold to classify. Decision trees split data using Gini, entropy or information gain. Naive Bayes applies Bayes' theorem assuming independent features. SVMs maximize the margin and use kernels for non-linear data. Watch for overfitting and use cross-validation and regularization.

Unsupervised learning works without labels. K-means clusters by assigning points to the nearest centroid and updating the centroids, choosing K with the elbow method or silhouette scores. PCA reduces dimensions by keeping the highest-variance eigenvector directions. Anomaly detection flags deviations (point, contextual or collective) using One-Class SVM, Isolation Forest or LOF.

Reinforcement learning teaches an agent through rewards. Q-learning (off-policy, uses max) and SARSA (on-policy, uses the actual next action) both update Q-values and balance exploration against exploitation.

Deep learning stacks neurons into layers trained by backpropagation and gradient descent. Perceptrons are linear-only; MLPs add hidden layers and non-linearity. CNNs use convolution and pooling for images; RNNs, LSTMs and GRUs handle sequences; transformers use self-attention and power LLMs; and diffusion models generate data by learning to reverse noise addition.

---

## 6. Skills Assessment: Approach Only

The module ends with short theory questions. These notes contain no assessment answers. Revise using §1 and §5.4, and for each question, name the concept from its description before answering.
