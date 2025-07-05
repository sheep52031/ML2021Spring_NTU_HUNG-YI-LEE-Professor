# Machine Learning Regression Learning Notes

## Overview
This notebook contains my learning journey through the COVID-19 regression task, where I built a deep neural network to predict the final number of confirmed cases based on various features.

## Key Concepts Learned

### 1. Data Analysis and Understanding
- **Target Variable**: `tested_positive.2` - the final number of confirmed cases
- **Features**: Various behavioral and demographic features from different states
- **Data Distribution**: Target values range from 2.34 to 40.96, with mean 16.44 and standard deviation 7.61

### 2. Neural Network Architecture
```python
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
```

**Components Explained**:
- **Linear Layers**: Transform input features through weighted connections
- **BatchNorm1d**: Normalizes activations, making training more stable
- **Dropout**: Randomly drops 20% of neurons during training to prevent overfitting
- **LeakyReLU**: Activation function that allows negative values to pass through slightly

### 3. Loss Function and Regularization
- **MSE Loss**: Mean Squared Error for regression tasks
- **L2 Regularization**: Penalizes large weights to prevent overfitting
- **Regularization Strength**: 0.00075 (controls how much to penalize large weights)

### 4. Training Process Understanding

#### Why Validate After Each Epoch?
- **Monitor Learning Progress**: Track if the model is improving on unseen data
- **Early Stopping**: Stop training when validation loss stops improving
- **Model Selection**: Save the best model based on validation performance
- **Prevent Overfitting**: Detect when model starts memorizing training data

#### Training vs Validation vs Test Sets
- **Training Set**: Used to train the model (model sees both features and targets)
- **Validation Set**: Used during training to monitor performance and select best model
- **Test Set**: Used only for final evaluation (no target values provided)

### 5. Optimizer Choice: Adam
- **Advantages**: Adaptive learning rate, good for most problems
- **Parameters**: Learning rate = 0.001, weight decay = 1e-5
- **Why Adam over SGD**: Automatically adjusts learning rate for each parameter

### 6. Reproducibility
```python
myseed = 2020815
torch.manual_seed(myseed)
np.random.seed(myseed)
```
- **Purpose**: Ensure same results every time you run the code
- **Why Important**: For fair comparison between different experiments

## Training Results Analysis

### Model Performance
- **Final MSE**: 0.8423 (very low compared to target standard deviation of 7.61)
- **Training Epochs**: 1464 (with early stopping)
- **Interpretation**: MSE << Standard deviation indicates excellent prediction accuracy

### Learning Curve Observations

![Learning Curve](learning_curve.png)

**Key Observations from the Learning Curve**:
- **Smooth Learning**: Both training and validation loss decrease steadily
- **No Overfitting**: Validation loss closely follows training loss
- **Good Convergence**: Model learns effectively without memorizing
- **Stable Training**: No dramatic fluctuations in loss values
- **Early Stopping**: Training stopped at epoch 1464 when no improvement was detected

**What This Tells Us**:
- The model architecture is appropriate for this task
- Regularization (Dropout + L2) is working effectively
- The learning rate and batch size are well-tuned
- The model generalizes well to unseen data

### Prediction Accuracy

![Prediction Scatter](prediction_scatter.png)

**Key Observations from the Scatter Plot**:
- **High Accuracy**: Points closely align with the perfect prediction line (y=x)
- **Low Bias**: No systematic over/under-prediction
- **Low Variance**: Predictions are consistent and stable
- **Good Coverage**: Predictions span the full range of target values
- **Minimal Outliers**: Very few points deviate significantly from the perfect line

**What This Tells Us**:
- The model has learned the underlying patterns in the data
- Predictions are reliable across different target value ranges
- The model doesn't have systematic prediction errors
- The feature selection and preprocessing were effective

## Key Insights

### 1. Data Preprocessing Importance
- **Feature Selection**: Using only 14 selected features instead of all 93
- **Normalization**: Standardizing features improves training stability
- **Train/Dev Split**: 90/10 split for validation

### 2. Model Design Decisions
- **Simple Architecture**: Single hidden layer with 32 neurons
- **Regularization**: Dropout + L2 regularization prevent overfitting
- **Activation Function**: LeakyReLU provides non-linearity

### 3. Training Strategy
- **Batch Size**: 200 (good balance between memory and convergence)
- **Early Stopping**: Prevents overfitting and saves time
- **Model Checkpointing**: Saves best model based on validation performance

## Lessons Learned

1. **Start Simple**: A simple model can achieve excellent results with proper training
2. **Monitor Validation**: Always use validation set to track real performance
3. **Regularization Works**: Dropout and L2 regularization are essential for generalization
4. **Data Quality Matters**: Good feature selection and preprocessing are crucial
5. **Reproducibility**: Set random seeds for consistent results

## Code Structure Understanding

### Data Loading Pipeline
```python
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'))
    return dataloader
```

### Training Loop
```python
def train(tr_set, dv_set, model, config, device):
    # Training loop with validation after each epoch
    # Early stopping and model checkpointing
```

### Evaluation Functions
```python
def dev(dv_set, model, device):
    # Calculate validation loss
    
def test(tt_set, model, device):
    # Generate predictions for test set
```

## Future Improvements

1. **Feature Engineering**: Try different feature combinations
2. **Architecture**: Experiment with deeper networks
3. **Hyperparameter Tuning**: Grid search for optimal parameters
4. **Ensemble Methods**: Combine multiple models
5. **Cross-Validation**: More robust validation strategy

## Visual Analysis Summary

The combination of the learning curve and prediction scatter plot provides strong evidence that:

1. **The model is well-trained**: Smooth convergence without overfitting
2. **The predictions are accurate**: Points closely follow the perfect prediction line
3. **The model generalizes well**: Good performance on validation data
4. **The approach is sound**: Simple architecture with proper regularization works effectively

These visualizations are crucial for understanding model behavior and building confidence in the results.

---

*This learning journey demonstrates the importance of understanding both the theoretical concepts and practical implementation details in machine learning, with visual evidence supporting the effectiveness of the approach.* 