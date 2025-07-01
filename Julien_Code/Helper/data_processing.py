import pandas as pd
import numpy as np
from numba import njit
from typing import Tuple, List, cast
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from gluonts.dataset.pandas import PandasDataset

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
            event_flags[i] = -1
            s_pos = 0.0
            s_neg = 0.0
    return event_flags


class Data:
    def __init__(self, path: str, patch_size: int, min_patches: int, train_split_date: int = 400, val_split_date: int = 450):
        print("🚀 Starting Data Processing Pipeline")
        print("=" * 50)

        # 1. Load data from Parquet file
        df = self._load_data(path)

        # 2. Split data chronologically
        self.df_train, self.df_val, self.df_test = self._split_data(df, train_split_date, val_split_date)

        # 3. Preprocess data (fit on train, transform all)
        self.df_train, self.df_val, self.df_test, self.feature_cols, self.target_cols = self._preprocess(self.df_train, self.df_val, self.df_test)

        # 4. Structure data as time series
        print("⏰ Structuring data into time series format...")
        self.df_train = self._structure_as_time_series(self.df_train, patch_size, min_patches, "train")
        self.df_val = self._structure_as_time_series(self.df_val, patch_size, min_patches, "validation")
        self.df_test = self._structure_as_time_series(self.df_test, patch_size, min_patches, "test")

        # 5. Create final GluonTS datasets
        print("📊 Creating final GluonTS datasets...")
        self.train_dataset = self._create_gluonts_dataset(self.df_train)
        self.val_dataset = self._create_gluonts_dataset(self.df_val)
        self.test_dataset = self._create_gluonts_dataset(self.df_test)
        print("✅ Dataset creation complete.")
        print("=" * 50)

    def _load_data(self, path: str) -> pd.DataFrame:
        print("📁 Loading data from Parquet file...")
        df = pd.read_parquet(path)
        df = df.sort_values(['date', 'ts_id']).reset_index(drop=True)
        return df

    def _split_data(self, df: pd.DataFrame, train_split_date: int, val_split_date: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(f"🔪 Splitting data at dates: train < {train_split_date}, val < {val_split_date}")
        df_train = cast(pd.DataFrame, df[df['date'] < train_split_date].copy())
        df_val = cast(pd.DataFrame, df[(df['date'] >= train_split_date) & (df['date'] < val_split_date)].copy())
        df_test = cast(pd.DataFrame, df[df['date'] >= val_split_date].copy())
        print(f"   Train set: {len(df_train)} rows")
        print(f"   Validation set: {len(df_val)} rows")
        print(f"   Test set: {len(df_test)} rows")
        return df_train, df_val, df_test

    def _preprocess(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        print("🛠️  Starting preprocessing...")

        feature_cols = [c for c in df_train.columns if c.startswith("feature_")]
        target_cols = [c for c in df_train.columns if c.startswith("resp")]

        # Feature selection on training data
        print("🔍 Step 1: Performing feature selection...")
        # High correlation removal
        corr_matrix = df_train[feature_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        df_train.drop(columns=to_drop, inplace=True)
        df_val.drop(columns=to_drop, inplace=True)
        df_test.drop(columns=to_drop, inplace=True)
        feature_cols = [c for c in feature_cols if c not in to_drop]
        print(f"   Dropped {len(to_drop)} highly correlated features.")

        # Outlier Capping (Winsorization)
        print("📊 Step 2: Capping outliers...")
        cols_to_winsorize = feature_cols + target_cols
        quantiles = df_train[cols_to_winsorize].quantile([0.01, 0.99])
        for col in cols_to_winsorize:
            low, high = quantiles.loc[0.01, col], quantiles.loc[0.99, col]
            df_train[col] = df_train[col].clip(low, high)
            df_val[col] = df_val[col].clip(low, high)
            df_test[col] = df_test[col].clip(low, high)

        # Low Variance Feature Removal
        print("📉 Step 3: Removing low-variance features...")
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(df_train[feature_cols])
        support_mask = selector.get_support()
        low_variance_cols = (
            [
                col
                for col, supported in zip(feature_cols, support_mask)
                if not supported
            ]
            if support_mask is not None
            else []
        )
        df_train.drop(columns=low_variance_cols, inplace=True)
        df_val.drop(columns=low_variance_cols, inplace=True)
        df_test.drop(columns=low_variance_cols, inplace=True)
        feature_cols = [c for c in feature_cols if c not in low_variance_cols]
        print(f"   Dropped {len(low_variance_cols)} low-variance features.")
        
        # Standardization
        print("📈 Step 4: Applying Z-score standardization...")
        scaler = StandardScaler()
        df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
        df_val[feature_cols] = scaler.transform(df_val[feature_cols])
        df_test[feature_cols] = scaler.transform(df_test[feature_cols])
        
        # CUSUM Event Detection
        print("🎯 Step 5: Creating event flags with a CUSUM filter...")
        resp_std_train = df_train['resp'].std()
        for df in [df_train, df_val, df_test]:
            df['event'] = _cusum_filter(df['resp'].to_numpy(), threshold=3*resp_std_train)
        feature_cols.append('event')

        print(f"   Final feature count: {len(feature_cols)}")
        return df_train, df_val, df_test, feature_cols, target_cols

    def _structure_as_time_series(self, df: pd.DataFrame, patch_size: int, min_patches: int, name: str) -> pd.DataFrame:
        print(f"   Structuring {name} data...")
        df["timestamp"] = (
            pd.to_datetime(df["date"], unit="D", origin="unix")
            + pd.to_timedelta(df["ts_id"].astype(int), unit="s")
        )
        df = df.sort_values(["date", "timestamp"]).drop_duplicates(subset=["date", "timestamp"], keep="last")

        min_length = patch_size * min_patches
        df = cast(pd.DataFrame, df.groupby("date").filter(lambda x: len(x) >= min_length))
        
        # Defragment after filtering
        return df.copy()

    def _create_gluonts_dataset(self, df: pd.DataFrame) -> PandasDataset:
        return PandasDataset.from_long_dataframe(
            df,
            target='resp', # Using single 'resp' as target for simplicity, can be changed to self.target_cols
            feat_dynamic_real=self.feature_cols,
            item_id="date",
            timestamp="timestamp",
            freq="s",
        ) 