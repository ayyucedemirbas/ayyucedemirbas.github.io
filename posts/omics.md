# 🧬 Building Patient Graphs for Multi-Omics Cancer Survival Classification Using Graph Neural Networks (GNNs)

### 1. Introduction

This project implements a **Graph Neural Network (GNN)** to predict patient survival status ("Alive" or "Deceased") in the **TCGA-BRCA** (Breast Invasive Carcinoma) dataset. The code integrates **multi-omics features**—RNA expression (`rs_`), copy number variation (`cn_`), mutation (`mu_`), and proteomics (`pp_`)—into a unified framework where **each patient is represented as a node** in a graph. Edges between patients are established based on feature similarity, allowing the model to capture relationships among biologically similar individuals.

---

### 2. Data Representation

Each column in the dataset corresponds to a biological feature:

* **`rs_*`** → RNA-seq expression levels (e.g., `rs_CLEC3A`, `rs_CPB1`)
* **`cn_*`** → Copy number variation features
* **`mu_*`** → Mutation information
* **`pp_*`** → Protein-level data

The target label, `vital.status`, indicates the **survival outcome**:

* `0` → Alive
* `1` → Deceased

Before modeling, the dataset is **balanced** via resampling to address class imbalance, ensuring fair learning between survival outcomes.

---

### 3. Data Preprocessing

The `load_and_preprocess_data()` function performs:

1. **Missing value handling:** replaces missing numeric entries with column medians.
2. **Label normalization:** converts various label forms (`Alive/Dead`, `Living/Deceased`) into binary numeric format.
3. **Class balancing:** uses bootstrapped upsampling of the minority class.
4. **Feature categorization:** automatically detects feature types by prefix.

This results in a clean, balanced dataset ready for graph construction.


```python
def load_and_preprocess_data():
    try:
        df = pd.read_csv('data.csv')
        print(f"Shape: {df.shape}")

        print(f"Missing values per column: {df.isnull().sum().sum()}")
        if df.isnull().sum().sum() > 0:
            # Fill missing values with median for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            print("Missing values filled with median")

        unique_values = df['vital.status'].unique()
        print(f"Unique vital.status values: {unique_values}")

        if set(unique_values).issubset({0, 1}):
            pass
        elif set(unique_values).issubset({'Alive', 'Dead', 'alive', 'dead'}):
            df['vital.status'] = df['vital.status'].str.lower().map({'alive': 0, 'dead': 1})
        elif set(unique_values).issubset({'Living', 'Deceased', 'living', 'deceased'}):
            df['vital.status'] = df['vital.status'].str.lower().map({'living': 0, 'deceased': 1})
        else:
            le = LabelEncoder()
            df['vital.status'] = le.fit_transform(df['vital_status'])
            print(f"Label encoding applied: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Check initial label distribution
        print("Initial label distribution:")
        label_counts = df["vital.status"].value_counts()
        print(label_counts)

        df_majority = df[df["vital.status"] == 0]
        df_minority = df[df["vital.status"] == 1]

        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=len(df_majority),
                                         random_state=42)

        # Combine majority class with upsampled minority class
        df_balanced = pd.concat([df_majority, df_minority_upsampled])

        # Verify the new distribution
        print("After upsampling:")
        label_counts_upsampled = df_balanced["vital.status"].value_counts()
        print(label_counts_upsampled)

        # Separate feature types
        rna_cols = [col for col in df_balanced.columns if col.startswith('rs_')]
        cn_cols = [col for col in df_balanced.columns if col.startswith('cn_')]
        mu_cols = [col for col in df_balanced.columns if col.startswith('mu_')]
        pp_cols = [col for col in df_balanced.columns if col.startswith('pp_')]

        print(f"Feature distribution:")
        print(f"  RNA-seq (rs_): {len(rna_cols)}")
        print(f"  Copy Number (cn_): {len(cn_cols)}")
        print(f"  Mutation (mu_): {len(mu_cols)}")
        print(f"  Protein (pp_): {len(pp_cols)}")

        return df_balanced

    except FileNotFoundError:
        print("Error: data.csv file not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

df = load_and_preprocess_data()

```
```python

def prepare_patient_graph_data(X_train, X_test, adj_matrix_full, train_indices, test_indices):
    n_patients_total = X_train.shape[0] + X_test.shape[0]
    n_features = X_train.shape[1]

    # Combine all patient features for the full graph
    X_all = np.vstack([X_train, X_test])

    # Reshape for graph input: [1, n_patients, n_features]
    X_graph = X_all.reshape(1, n_patients_total, n_features)

    # Adjacency matrix: [1, n_patients, n_patients]
    adj_batch = adj_matrix_full.reshape(1, n_patients_total, n_patients_total)

    return X_graph, adj_batch, train_indices, test_indices

```


---

### 4. Constructing the Patient Graph

The **patient adjacency matrix** defines how patients are connected:

```python
adj_matrix = (cosine_similarity(X) > threshold).astype(np.float32)
```

* **Nodes** = Patients
* **Edges** = Cosine similarity > threshold (e.g., 0.5)

This graph captures patient-to-patient relationships based on molecular profile similarity. The adjacency matrix is normalized using the **symmetric degree normalization** technique common in GCNs:
[
\hat{A} = D^{-1/2}(A + I)D^{-1/2}
]
This normalization stabilizes message passing during training.

```python
def create_patient_adjacency_matrix(X, method='cosine', threshold=0.7, k_neighbors=10):
    #Create adjacency matrix where nodes are patients (samples)
    n_patients = X.shape[0]
    print(f"Creating patient adjacency matrix for {n_patients} patients")

    if method == 'cosine':
        # Compute cosine similarity between patients
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(X)
        # Convert to adjacency matrix
        adj_matrix = (similarity_matrix > threshold).astype(np.float32)

    elif method == 'correlation':
        # Compute correlation between patient feature vectors
        corr_matrix = np.corrcoef(X)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        adj_matrix = (np.abs(corr_matrix) > threshold).astype(np.float32)

    else:
        from sklearn.neighbors import kneighbors_graph
        adj_matrix = kneighbors_graph(X, n_neighbors=min(k_neighbors, n_patients//4),
                                    mode='connectivity', include_self=False)
        adj_matrix = adj_matrix.toarray().astype(np.float32)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric

    # Remove self-loops initially, then add them back
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix += np.eye(n_patients, dtype=np.float32)

    # Normalize adjacency matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix) + 1e-6))
    adj_normalized = degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt

    print(f"Patient adjacency matrix shape: {adj_normalized.shape}")
    print(f"Number of patient connections: {np.sum(adj_matrix > 0) // 2}")

    return adj_normalized.astype(np.float32)
```

---

### 5. Graph Neural Layer

The custom `GraphNeuralLayer` class defines how each patient (node) updates its representation:
[
h_i^{(l+1)} = \sigma(W_{\text{self}}h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} A_{ij}W_{\text{msg}}h_j^{(l)})
]

* **`W_self`** → weights for self-feature updates
* **`W_msg`** → weights for neighbor messages
* **`A`** → adjacency matrix
* **`σ`** → activation (ReLU)

This layer captures both **individual patient characteristics** and **peer context** within the graph.
```python

class GraphNeuralLayer(layers.Layer):
    def __init__(self, units, activation='relu', use_bias=True, **kwargs):
        super(GraphNeuralLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        # input_shape: [(batch, nodes, features), (batch, nodes, nodes)]
        feature_dim = input_shape[0][-1]

        # Message passing weights
        self.W_msg = self.add_weight(
            name='W_msg',
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )

        # Self-connection weights
        self.W_self = self.add_weight(
            name='W_self',
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )

        super(GraphNeuralLayer, self).build(input_shape)

    def call(self, inputs):
        features, adjacency = inputs

        # Self-connection: process own features
        self_output = tf.matmul(features, self.W_self)

        # Message passing: aggregate neighbor information
        neighbor_messages = tf.matmul(features, self.W_msg)
        aggregated = tf.matmul(adjacency, neighbor_messages)

        # Combine self and neighbor information
        output = self_output + aggregated

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return self.activation(output)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        }
        base_config = super(GraphNeuralLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```
---

### 6. GNN Architecture

The model (`build_patient_gnn_model`) stacks three graph convolution layers:

| Layer       | Units | Activation | Dropout |
| :---------- | :---- | :--------- | :------ |
| GNN Layer 1 | 128   | ReLU       | 0.3     |
| GNN Layer 2 | 64    | ReLU       | 0.3     |
| GNN Layer 3 | 32    | ReLU       | 0.2     |

The network ends with a **Dense(sigmoid)** layer outputting one probability per patient (death likelihood).

```python
def build_patient_gnn_model(n_patients, n_features, num_classes=2):
    # Input: [batch_size=1, n_patients, n_features]
    feature_input = keras.Input(shape=(n_patients, n_features), name='patient_features')
    adjacency_input = keras.Input(shape=(n_patients, n_patients), name='patient_adjacency')

    # GNN layers for patient nodes
    x = GraphNeuralLayer(128, activation='relu')([feature_input, adjacency_input])
    x = layers.Dropout(0.3)(x)

    x = GraphNeuralLayer(64, activation='relu')([x, adjacency_input])
    x = layers.Dropout(0.3)(x)

    x = GraphNeuralLayer(32, activation='relu')([x, adjacency_input])
    x = layers.Dropout(0.2)(x)

    # Global pooling to get graph-level representation
    # Since we're predicting for individual patients, we'll extract node embeddings
    # For training, we'll use all patient nodes and their labels

    # Output layer: one prediction per patient node
    outputs = layers.Dense(1, activation='sigmoid', name='patient_predictions')(x)

    model = keras.Model(inputs=[feature_input, adjacency_input], outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

```

---

### 7. Training Procedure

A **single-graph training loop** is used, treating all patients (train + test) as nodes in one graph:

* **Training mask:** only allows gradient updates on training nodes.
* **Validation mask:** used to monitor generalization.

Custom early stopping (patience = 15) halts training if no improvement in validation accuracy occurs.

Metrics used:

* **Binary cross-entropy loss**
* **Accuracy**
* **AUC (ROC area)**

```python
def train_and_evaluate_patient_model(model, X_train, X_test, y_train, y_test, adj_matrix):
    # We need to work with the full graph structure
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, n_train + n_test)

    X_graph, adj_batch, train_idx, test_idx = prepare_patient_graph_data(
        X_train, X_test, adj_matrix, train_indices, test_indices
    )

    print(f"Graph input shape: {X_graph.shape}")
    print(f"Adjacency batch shape: {adj_batch.shape}")
    print(f"Train indices: {len(train_idx)}, Test indices: {len(test_idx)}")

    # For training, we need to create a custom training loop since we're working with a single graph containing all patients

    # Combine labels
    y_all = np.concatenate([y_train, y_test])

    # Create masks for training and testing
    train_mask = np.zeros(len(y_all), dtype=bool)
    test_mask = np.zeros(len(y_all), dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    # Custom training function for patient GNN
    def train_step(model, X_graph, adj_batch, y_all, train_mask):
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model([X_graph, adj_batch])  # Shape: [1, n_patients, 1]
            predictions = tf.squeeze(predictions, axis=[0, 2])  # Shape: [n_patients]

            # Only compute loss on training patients
            train_predictions = tf.boolean_mask(predictions, train_mask)
            train_labels = tf.boolean_mask(y_all, train_mask)

            loss = keras.losses.binary_crossentropy(train_labels, train_predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss, train_predictions, train_labels

    epochs = 100
    best_val_acc = 0
    patience = 15
    patience_counter = 0

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):
        loss, train_preds, train_labels = train_step(
            model, X_graph, adj_batch, y_all, train_mask
        )

        val_predictions = model([X_graph, adj_batch])
        val_predictions = tf.squeeze(val_predictions, axis=[0, 2])
        val_preds = tf.boolean_mask(val_predictions, test_mask)
        val_labels = tf.boolean_mask(y_all, test_mask)

        val_loss = tf.reduce_mean(keras.losses.binary_crossentropy(val_labels, val_preds))

        train_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.round(train_preds), train_labels), tf.float32
        ))
        val_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.round(val_preds), val_labels), tf.float32
        ))

        history['loss'].append(float(loss))
        history['val_loss'].append(float(val_loss))
        history['accuracy'].append(float(train_acc))
        history['val_accuracy'].append(float(val_acc))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Val_Loss={val_loss:.4f}, "
                  f"Acc={train_acc:.4f}, Val_Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save_weights('best_patient_model.weights.h5')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_weights('best_patient_model.weights.h5')
    final_predictions = model([X_graph, adj_batch])
    final_predictions = tf.squeeze(final_predictions, axis=[0, 2])

    test_pred_proba = tf.boolean_mask(final_predictions, test_mask).numpy()
    test_pred = (test_pred_proba > 0.5).astype(int)

    print(f"\nFinal Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    auc_score = roc_auc_score(y_test, test_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")

    return model, history, test_pred_proba, test_pred

```

---

### 8. Visualization and Evaluation

The `visualize_data()` and `plot_results()` functions generate:

* Class and feature-type distributions
* Patient adjacency heatmaps
* PCA embeddings (colored by survival status)
* Learning curves, confusion matrix, and ROC curves

These visualizations help interpret how omic patterns and graph structure relate to clinical outcomes.

```python
def visualize_data(df, adj_matrix):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].pie(df['vital.status'].value_counts(),
                   labels=['Alive', 'Deceased'], autopct='%1.1f%%')
    axes[0, 0].set_title('Class Distribution')

    # Feature type distributions
    rna_cols = [col for col in df.columns if col.startswith('rs_')]
    cn_cols = [col for col in df.columns if col.startswith('cn_')]
    mu_cols = [col for col in df.columns if col.startswith('mu_')]
    pp_cols = [col for col in df.columns if col.startswith('pp_')]

    feature_counts = [len(rna_cols), len(cn_cols), len(mu_cols), len(pp_cols)]
    axes[0, 1].bar(['RNA-seq', 'Copy Number', 'Mutation', 'Protein'], feature_counts)
    axes[0, 1].set_title('Feature Types Distribution')
    axes[0, 1].set_ylabel('Number of Features')

    # Sample feature distributions by class
    if len(rna_cols) > 0:
        sample_features = df[rna_cols[:10]].values
        axes[0, 2].boxplot([sample_features[df['vital.status'] == 0].flatten(),
                            sample_features[df['vital.status'] == 1].flatten()],
                           labels=['Alive', 'Deceased'])
        axes[0, 2].set_title('Sample RNA-seq Features by Class')
        axes[0, 2].set_ylabel('Expression Level')
    else:
        axes[0, 2].text(0.5, 0.5, 'No RNA-seq features found', ha='center', va='center')

    # Patient adjacency matrix heatmap
    sample_adj = adj_matrix[:50, :50]  # Show subset for visibility
    sns.heatmap(sample_adj, cmap='viridis', ax=axes[1, 0])
    axes[1, 0].set_title('Patient Adjacency Matrix (50x50 subset)')

    # Patient connectivity degree distribution
    degrees = np.sum(adj_matrix > 0, axis=1)
    axes[1, 1].hist(degrees, bins=30, alpha=0.7)
    axes[1, 1].set_title('Patient Connectivity Distribution')
    axes[1, 1].set_xlabel('Number of Connections')
    axes[1, 1].set_ylabel('Frequency')

    # PCA visualization
    features = df.drop('vital.status', axis=1).values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    colors = ['blue' if x == 0 else 'red' for x in df['vital.status']]
    axes[1, 2].scatter(features_pca[:, 0], features_pca[:, 1], c=colors, alpha=0.6)
    axes[1, 2].set_title(f'PCA Visualization\n(Explained variance: {pca.explained_variance_ratio_.sum():.3f})')
    axes[1, 2].set_xlabel('First Principal Component')
    axes[1, 2].set_ylabel('Second Principal Component')

    plt.tight_layout()
    plt.show()

```

---

### 9. Results Interpretation

Final statistics printed include:

* **Test Accuracy and AUC**
* **Patient connectivity statistics** (average, min, max degree)
* **Graph size and density**

These offer insight into whether survival outcomes correlate with structural graph properties (e.g., patients in similar molecular clusters showing similar prognoses).

---

### 10. Significance

By reframing survival prediction as a **node classification task**, this approach:

* Integrates multiple omics layers seamlessly.
* Exploits patient similarity patterns often ignored in linear models.
* Provides an interpretable graph structure that may reveal **subgroups of molecularly similar patients**.

---

### 11. Extensions

Possible next steps include:

* Adding **edge weighting** via Pearson correlation strength.
* Employing **Graph Attention Networks (GAT)** for interpretable message passing.
* Integrating **clinical and histopathological** data modalities.
* Visualizing patient embeddings via **t-SNE or UMAP** for subtyping insights.
