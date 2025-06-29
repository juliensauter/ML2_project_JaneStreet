import pandas as pd
import polars as pl
import numpy as np
from numba import njit
from typing import Dict, Tuple, Union, cast
from sklearn.feature_selection import VarianceThreshold

from gluonts.dataset.pandas import PandasDataset
from .timing import timeit, timer

@njit
def _cusum_filter(resp_val: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applies a symmetric CUSUM filter to a series of values.
    Optimized with Numba for high performance.
    """
    s_pos = 0.0
    s_neg = 0.0
    event_flags = np.zeros_like(resp_val, dtype=np.int8)

    for i in range(1, len(resp_val)):
        s_pos = max(0, s_pos + resp_val[i])
        s_neg = min(0, s_neg + resp_val[i])
        if s_pos > threshold:
            event_flags[i] = 1
            s_pos = 0.0
        elif s_neg < -threshold:
            event_flags[i] = -1 # Changed to -1 for downward events
            s_pos = 0.0 # Also reset s_pos
            s_neg = 0.0
    return event_flags

class Data:
    df: pd.DataFrame
    feature_cols: list[str]
    target_cols: list[str]
    dataset: PandasDataset
    data: Dict[str, Union[pl.DataFrame, pl.Series]]

    def __init__(self, path: str, patch_size: int, min_patches: int):
        """
        Orchestrates the preprocessing of the Jane Street dataset.
        This involves loading, cleaning, feature engineering, and structuring the data
        into time series format suitable for the Moirai model.
        """
        print("🚀 Starting Data Preprocessing Pipeline")
        print("=" * 50)
        
        # Load and perform initial type casting
        self.data, self.df = self._load_data_from_parquet(path)

        # Ensure chronological order before any processing
        self.df = self.df.sort_values(['date', 'row_id']).reset_index(drop=True)

        # Apply advanced preprocessing steps based on best practices
        with timer("Complete Preprocessing Pipeline"):
            self._select_features()
            self._winsorize_outliers()
            self._remove_low_variance_features()
            self._standardize_features()
            self._add_event_flags()

        # Defragment the DataFrame after numerous column manipulations
        print("🔄 Defragmenting DataFrame for performance...")
        self.df = self.df.copy()
        
        self._structure_as_time_series(patch_size, min_patches)
        print("✅ Advanced preprocessing complete.")

        # --- Data Integrity Check ---
        print("🕵️  Performing data integrity check for NaN/Inf values...")
        # Check for NaNs
        nan_info = self.df.isnull().sum()
        nan_cols = nan_info[nan_info > 0]
        has_nan = not nan_cols.empty

        # Check for Infs
        numeric_df = self.df.select_dtypes(include=np.number)
        inf_info = np.isinf(numeric_df).sum()
        inf_cols = inf_info[inf_info > 0]
        has_inf = not inf_cols.empty

        if has_nan or has_inf:
            print("🔥 CRITICAL: Invalid values detected in the DataFrame after preprocessing:")
            if has_nan:
                print("\n--- NaN Values Found In ---")
                print(nan_cols)
            if has_inf:
                print("\n--- Infinite Values Found In ---")
                print(inf_cols)
            raise ValueError("Invalid values (NaN or Inf) detected in the dataframe, aborting training.")
        else:
            print("✅ Data integrity check passed. No NaN or Inf values found.")
        # --------------------------

        # Final dataset creation for the model
        print("📊 Creating final GluonTS dataset...")
        self.dataset = PandasDataset.from_long_dataframe(
            self.df,
            target=self.target_cols,
            feat_dynamic_real=self.feature_cols,
            item_id="date",
            timestamp="timestamp",
            freq="s",
        )
        print("✅ Dataset creation complete.")
        print("=" * 50)

    @timeit("Data Loading from Parquet")
    def _load_data_from_parquet(self, path: str, include_resp: bool = True, include_weight_feature: bool = False, filter_weight_zero: bool = False) -> Tuple[Dict[str, Union[pl.DataFrame, pl.Series]], pd.DataFrame]:
        """
        Loads data from a Parquet file using Polars for efficiency, performs initial
        cleaning and type casting, and returns both a dictionary of Polars DataFrames
        and a consolidated Pandas DataFrame.
        """
        print("📁 Loading data from Parquet file...")
        
        with timer("Polars Schema Scan"):
            lf = pl.scan_parquet(str(path))
            schema = lf.collect_schema()

        with timer("Feature Column Detection"):
            self.feature_cols = [c for c in schema.names() if c.startswith("feature_")]
            self.target_cols = [f"resp_{i}" for i in range(1, 5)]
            if include_resp:
                self.target_cols += ["resp"]

        with timer("Column Selection and Type Casting"):
            float_cols = self.feature_cols + self.target_cols
            if include_weight_feature:
                float_cols += ["weight"]

            lf = (
                lf
                .select(["ts_id", "date"] + float_cols)
                .with_columns(pl.col(float_cols).cast(pl.Float32))
                .with_columns(pl.col("date").cast(pl.Int32))
            )

            if filter_weight_zero:
                lf = lf.filter(pl.col("weight") != 0)

            if include_weight_feature:
                self.feature_cols = self.feature_cols + ["weight"]
            
        # Slice the LazyFrame to the first 30,000 rows for faster development
        # print("🔪 Slicing DataFrame to first 30,000 rows for development.")
        # lf = lf.head(30000)

        with timer("Polars DataFrame Collection"):
            df: pl.DataFrame = lf.collect()

        with timer("DataFrame Splitting"):
            X = df.select(self.feature_cols)
            y = df.select(self.target_cols)
            
            # Select as Series to ensure correct types for the dictionary
            date_arr: pl.Series = df["date"]
            row_id: pl.Series = df["ts_id"]

        with timer("Pandas DataFrame Creation"):
            polars_dic = {'X': X, 'y': y, 'date': date_arr, 'row_id': row_id}
            
            with timer("Polars to Pandas Conversion"):
                row_id_pd = row_id.to_pandas().rename("row_id")
                date_pd = date_arr.to_pandas()
                X_pd = X.to_pandas()
                y_pd = y.to_pandas()
            
            with timer("DataFrame Concatenation"):
                pandas_df = pd.concat([
                        row_id_pd,
                        date_pd,
                        X_pd,
                        y_pd,
                    ],
                    axis=1,
                )
        
        return polars_dic, pandas_df

    @timeit("Feature Selection")
    def _select_features(self):
        """
        Reduces feature space by removing near-constant and highly correlated features
        to decrease noise and multicollinearity.
        """
        print("🔍 Step 1: Performing feature selection...")
        
        feature_cols = [c for c in self.df.columns if c.startswith('feature_')]

        with timer("Finding Unique Values (for constant and quasi-constant)"):
            nunique_series = self.df[feature_cols].nunique(dropna=False)

        with timer("Constant Feature Removal"):
            constant_mask = nunique_series == 1
            constant_cols = nunique_series[constant_mask].index.tolist() # type: ignore
            if constant_cols:
                self.df.drop(columns=constant_cols, inplace=True)
                print(f"   Dropped {len(constant_cols)} constant features.")
                feature_cols = [c for c in feature_cols if c not in constant_cols]

        with timer("Quasi-constant Feature Removal"):
            # First, get a list of columns that have only 2 unique values
            binary_cols = nunique_series[nunique_series == 2].index.tolist() # type: ignore
            
            # Then, iterate only over this much smaller list to check the frequency
            quasi_constant_cols = []
            for col in binary_cols:
                top_freq = self.df[col].value_counts(normalize=True).iloc[0]
                if top_freq > 0.99:
                    quasi_constant_cols.append(col)

            if quasi_constant_cols:
                self.df.drop(columns=quasi_constant_cols, inplace=True)
                print(f"   Dropped {len(quasi_constant_cols)} quasi-constant features.")
                feature_cols = [c for c in feature_cols if c not in quasi_constant_cols]

        if not feature_cols: 
            print("   No features remaining for correlation analysis.")
            self.feature_cols = feature_cols
            print(f"   Final feature count: {len(self.feature_cols)}")
            return

        with timer("Feature Correlation Calculation"):
            corr_matrix = self.df[feature_cols].corr().abs() # type: ignore
        
        with timer("High Correlation Feature Filtering"):
            upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            high_corr_mask = (corr_matrix > 0.95) & upper_triangle
            high_corr_cols = corr_matrix.columns[high_corr_mask.any(axis=0)].tolist()
            
            if high_corr_cols:
                self.df.drop(columns=high_corr_cols, inplace=True)
                print(f"   Dropped {len(high_corr_cols)} highly correlated features.")
            else:
                print("   No highly correlated features found.")
        
        self.feature_cols = [c for c in self.df.columns if c.startswith('feature_')]
        print(f"   Final feature count: {len(self.feature_cols)}")

    @timeit("Low Variance Feature Removal")
    def _remove_low_variance_features(self):
        """
        Removes continuous features with near-zero variance to prevent model
        instability and reduce noise, a crucial step before standardization.
        """
        print("📉 Step 2a: Removing low-variance features...")
        if not self.feature_cols:
            return

        selector = VarianceThreshold(threshold=0.01)

        numerical_features = self.df[self.feature_cols].select_dtypes(include=np.number)
        
        with timer("VarianceThreshold Fitting"):
            selector.fit(numerical_features)

        with timer("Identifying Low Variance Columns"):
            support = selector.get_support()
            if support is not None:
                low_variance_cols = numerical_features.columns[~support].tolist()
            else:
                low_variance_cols = []

        if low_variance_cols:
            self.df.drop(columns=low_variance_cols, inplace=True)
            print(f"   Dropped {len(low_variance_cols)} low-variance features.")
            self.feature_cols = [c for c in self.feature_cols if c not in low_variance_cols]
        else:
            print("   No low-variance features found.")

    @timeit("Outlier Capping (Winsorization)")
    def _winsorize_outliers(self):
        """
        Caps extreme outliers in features and targets at the 1st and 99th percentiles.
        This is a vectorized operation for performance.
        """
        print("📊 Step 2: Capping outliers using Winsorization...")
        cols_to_winsorize = self.feature_cols + self.target_cols
        
        with timer("Quantile Calculation"):
            # Calculate quantiles for all columns at once
            quantiles = self.df[cols_to_winsorize].quantile([0.01, 0.99])
        
        with timer("Data Clipping"):
            # Clip the columns in a vectorized manner
            self.df[cols_to_winsorize] = self.df[cols_to_winsorize].clip(
                lower=quantiles.loc[0.01],
                upper=quantiles.loc[0.99],
                axis=1
            )

    @timeit("Z-score Standardization")
    def _standardize_features(self):
        """
        Applies Z-score standardization to all feature and target columns in a vectorized operation.
        """
        print("📈 Step 3: Applying Z-score standardization to features and targets...")
        if not self.feature_cols:
            return
        
        cols_to_standardize = self.feature_cols + self.target_cols
        
        with timer("Mean and Std Calculation"):
            means = self.df[cols_to_standardize].mean()
            stds: pd.Series = cast(pd.Series, self.df[cols_to_standardize].std())
            # Avoid division by zero for constant columns that might have slipped through
            stds[stds < 1e-6] = 1.0

        with timer("Standardization Application"):
            self.df[cols_to_standardize] = (self.df[cols_to_standardize] - means) / stds

    @timeit("CUSUM Event Detection")
    def _add_event_flags(self):
        """
        Creates a new 'event' feature by applying a high-performance CUSUM filter.
        """
        print("🎯 Step 4: Creating event flags with a CUSUM filter...")
        
        with timer("Response Value Preparation"):
            resp_val = self.df['resp'].astype(float).to_numpy()
            threshold = 3 * resp_val.std()
        
        with timer("CUSUM Filter Application"):
            # Call the Numba-optimized function
            event_flags = _cusum_filter(resp_val, threshold)
        
        with timer("Event Column Addition"):
            self.df['event'] = event_flags
            
            # Add the new 'event' flag to the list of features for the model.
            if 'event' not in self.feature_cols:
                self.feature_cols.append('event')

    @timeit("Time Series Structuring")
    def _structure_as_time_series(self, patch_size: int, min_patches: int):
        """
        Structures the flat data into a proper time series format by:
        1. Creating a datetime 'timestamp' index.
        2. Padding each day's data to have a consistent one-second frequency.
        3. Filtering out any time series that are too short for the model.
        """
        print("⏰ Step 5: Structuring data into time series format...")
        
        with timer("Timestamp Creation"):
            # Create a proper datetime index
            self.df["timestamp"] = (
                pd.to_datetime(self.df["date"], unit="D", origin="unix")
                + pd.to_timedelta(self.df["row_id"].values, unit="s")
            )
        
        with timer("Data Sorting and Deduplication"):
            # Ensure data is unique and sorted before padding
            self.df = (
                self.df
                .sort_values(["date", "timestamp"])
                .drop_duplicates(subset=["date", "timestamp"], keep="last")
            )

        def _pad(grp: pd.DataFrame) -> pd.DataFrame:
            """
            Pads a group of data (a single day) to have a consistent time frequency.
            Uses forward-fill and then backward-fill to handle any created NaNs robustly.
            """
            # Get the date from the group name, as it's not a column in the group
            date_val = grp.name

            # Create date range without timer to avoid spamming logs
            idx = pd.date_range(
                grp["timestamp"].min(),
                grp["timestamp"].max(),
                freq="s",
                name="timestamp",
            )
            
            # Reindex, fill, and add the date back in
            padded = grp.set_index("timestamp").reindex(idx).ffill().bfill()
            padded["date"] = date_val
            
            return padded.reset_index()

        print(f"   Shape before padding: {self.df.shape}")
        
        with timer("Groupby Operation Setup"):
            # Prepare the groupby operation
            grouped = self.df.groupby("date")
        
        with timer("Time Series Padding"):
            # apply just concatenates the results from _pad. The resulting index is messy.
            df_padded = grouped.apply(_pad, include_groups=False)
            # A final reset_index(drop=True) creates a clean 0..N index.
            self.df = df_padded.reset_index(drop=True)
        print(f"   Shape after padding: {self.df.shape}")

        with timer("Short Series Filtering"):
            # Filter out time series that are too short for the given patch size
            min_length = patch_size * min_patches
            self.df = cast(pd.DataFrame, self.df.groupby("date").filter(lambda x: len(x) >= min_length))
        print(f"   Shape after filtering short series: {self.df.shape}")