# Data Preprocessing Pipeline for Financial Time Series

This document provides a comprehensive explanation of the data preprocessing pipeline implemented in `Julien_Code/Helper/preprocessing.py`. The pipeline is designed to transform raw, high-frequency financial data into a clean, robust, and structured format suitable for training sophisticated machine learning models, such as the Moirai forecasting model.

## General Philosophy and Approach

The core philosophy behind this pipeline aligns with the principles advocated in modern Financial Machine Learning (FinML) research, notably by experts like Marcos Lopez de Prado in his seminal book "Advances in Financial Machine Learning". The key objectives are:

1.  **Noise Reduction**: Financial data is inherently noisy. The pipeline employs several techniques to filter out uninformative features and reduce the impact of random fluctuations.
2.  **Robustness**: The preprocessing steps aim to make the subsequent model robust to outliers and non-stationary effects common in financial markets.
3.  **Information Preservation**: While reducing noise, the pipeline carefully preserves and even enhances predictive signals, for example, by creating event-based features.
4.  **Structural Integrity**: The data is meticulously structured into a regular time series format, which is a prerequisite for many advanced forecasting models.

## Step-by-Step Breakdown

The entire preprocessing orchestrator is the `Data` class, which executes the following steps in sequence.

### 1. Efficient Data Loading from Parquet

**What it does:** The pipeline begins by loading data from a Parquet file using the `polars` library. It performs initial type casting to optimize memory usage (e.g., `Float32`, `Int32`) and separates features (`X`), targets (`y`), and metadata.

```python
# From _load_data_from_parquet method
with timer("Polars Schema Scan"):
    lf = pl.scan_parquet(str(path))
# ...
with timer("Polars DataFrame Collection"):
    df: pl.DataFrame = lf.collect()
# ...
with timer("Polars to Pandas Conversion"):
    # ...
```

**Why it's important:**
*   **Performance**: For the large datasets typical in finance, `polars` offers significant speed and memory efficiency advantages over `pandas` for initial loading and manipulation.
*   **Format**: Parquet is a columnar storage format, which is highly efficient for the kind of tabular data used here, allowing for faster reads of specific columns.

### 2. Feature Selection: Combating the Curse of Dimensionality

**What it does:** This step prunes the feature set to remove irrelevant and redundant information. It involves three sub-steps:
1.  **Constant Feature Removal**: Drops features that have only one unique value.
2.  **Quasi-Constant Feature Removal**: Drops features where a single value dominates (e.g., >99% of the time).
3.  **High-Correlation Feature Removal**: Calculates the feature correlation matrix and drops one feature from any pair with a correlation greater than 0.95.

```python
# From _select_features method
# ... constant and quasi-constant removal ...
with timer("Feature Correlation Calculation"):
    corr_matrix = self.df[feature_cols].corr().abs() # type: ignore
# ...
high_corr_cols = corr_matrix.columns[high_corr_mask.any(axis=0)].tolist()
self.df.drop(columns=high_corr_cols, inplace=True)
```

**Why it's important (De Prado's Insights):**
*   De Prado argues that financial datasets often contain many redundant or uninformative features. Including them increases model complexity (curse of dimensionality), raises the risk of overfitting, and can lead to unstable models due to multicollinearity.
*   Removing highly correlated features ensures that the remaining features provide more unique information, leading to a more parsimonious and robust model.

### 3. Outlier Management: Winsorization

**What it does:** This step caps extreme outliers. Any value in the feature and target columns that is below the 1st percentile or above the 99th percentile is replaced with the value at that percentile.

```python
# From _winsorize_outliers method
quantiles = self.df[cols_to_winsorize].quantile([0.01, 0.99])
self.df[cols_to_winsorize] = self.df[cols_to_winsorize].clip(
    lower=quantiles.loc[0.01],
    upper=quantiles.loc[0.99],
    axis=1
)
```

**Why it's important:**
*   Financial returns are known to have "fat tails," meaning extreme events occur more often than in a normal distribution. These outliers can disproportionately influence the training process, especially for models that use squared error loss functions.
*   Winsorization is a standard technique to mitigate the impact of these outliers without completely removing the data points, which might still contain valuable information.

### 4. Low-Variance Feature Removal

**What it does:** After handling outliers but *before* standardization, the pipeline removes features with near-zero variance using `sklearn.feature_selection.VarianceThreshold`.

```python
# From _remove_low_variance_features method
selector = VarianceThreshold(threshold=0.01)
# ...
self.df.drop(columns=low_variance_cols, inplace=True)
```
**Why it's important:**
*   Features that have very low variance are almost constant and thus provide little to no predictive information.
*   This step must be performed before standardization, as standardization would scale all features to have a unit variance, obscuring the fact that some had low informational content to begin with.

### 5. Feature Standardization: Z-Score Normalization

**What it does:** Each feature column is rescaled to have a mean of 0 and a standard deviation of 1.

```python
# From _standardize_features method
means = self.df[self.feature_cols].mean()
stds: pd.Series = cast(pd.Series, self.df[self.feature_cols].std())
# ...
self.df[self.feature_cols] = (self.df[self.feature_cols] - means) / stds
```

**Why it's important:**
*   Most machine learning models, especially neural networks trained with gradient descent, converge faster and perform better when features are on a relatively similar scale.
*   Standardization prevents features with naturally large value ranges from dominating the learning process over features with smaller ranges.

### 6. Event-Based Feature Engineering: The CUSUM Filter

**What it does:** This is one of the most financially-motivated steps. A symmetric CUSUM (Cumulative Sum) filter is applied to the main target variable (`resp`) to create a new `event` feature. This feature flags when a significant cumulative deviation from the mean has occurred.

The filter tracks two cumulative sums: one for positive deviations (`s_pos`) and one for negative deviations (`s_neg`).
An event is flagged (`+1` for upward, `-1` for downward) when one of these sums crosses a predefined threshold.

\[
s_{i, pos} = \max(0, s_{i-1, pos} + y_i) \\
s_{i, neg} = \min(0, s_{i-1, neg} + y_i)
\]

An event is triggered at step \(i\) if \(s_{i, pos} > \text{threshold}\) or \(s_{i, neg} < -\text{threshold}\).

```python
# From _add_event_flags and _cusum_filter
resp_val = self.df['resp'].astype(float).to_numpy()
threshold = 3 * resp_val.std()
event_flags = _cusum_filter(resp_val, threshold)
self.df['event'] = event_flags
```

**Why it's important (De Prado's Insights):**
*   This method is a powerful tool for detecting structural breaks or significant events in a time series. De Prado heavily advocates for event-based analysis over fixed-time interval analysis.
*   While De Prado often uses the CUSUM filter as part of the "Triple Barrier Method" for labeling trades (i.e., defining profit-take and stop-loss levels), here it is cleverly used for **feature engineering**.
*   The resulting `event` feature explicitly tells the model that "something significant just happened," which can help it learn the distinct market dynamics that occur around such events. The use of a Numba-optimized function (`@njit`) makes this computationally intensive step feasible on large datasets.

### 7. Structuring for Time Series Models

**What it does:** This final crucial step converts the flat DataFrame into a format suitable for time series models.
1.  **Timestamp Creation**: A proper datetime index is created.
2.  **Padding**: The data for each day is resampled to a regular 1-second frequency. Missing values introduced by this resampling are filled using a forward-fill (`ffill`) and then a backward-fill (`bfill`).
3.  **Filtering**: Days (time series) that are too short to be useful for the model (i.e., cannot generate a minimum number of patches) are filtered out.

```python
# From _structure_as_time_series method
# ...
df_padded = grouped.apply(_pad, include_groups=False)
self.df = df_padded.reset_index(drop=True)
# ...
min_length = patch_size * min_patches
self.df = cast(pd.DataFrame, self.df.groupby("date").filter(lambda x: len(x) >= min_length))
```

**Why it's important:**
*   **Regular Frequency**: Most advanced time series models (Transformers, LSTMs) expect data to be sampled at regular time intervals. Market data is often irregular (based on trade times). This padding step enforces the required structure.
*   **Robust Filling**: The `ffill().bfill()` strategy is a robust way to handle missing data. `ffill` carries the last known value forward, a reasonable assumption for many financial metrics. `bfill` cleans up any remaining NaNs at the beginning of a series.
*   **Minimum Length**: Models that use patching (like Vision Transformers adapted for time series) require input sequences of a certain length. This filtering step ensures that every single item in the final dataset meets this minimum length requirement, preventing errors during training.

## Conclusion

This preprocessing pipeline is a well-architected, robust, and theoretically sound approach to preparing financial data for machine learning. It blends standard best practices (standardization, outlier removal) with sophisticated techniques drawn directly from leading financial ML research (CUSUM filters, correlation-based feature pruning). The resulting dataset is clean, informationally dense, and properly structured to maximize the potential of any downstream forecasting model. 