import pandas as pd
import numpy as np

DECISION_POINT = 'close'  
ROLL_N = 180              
WINSOR_Q = (0.01, 0.99)   
LOW_CARD_THRESHOLD = 4    

def main():
    print("Starting data processing 4.2 - Retaining full OHLC data")
    print("="*60)
    
    print("Loading raw data...")
    d = pd.read_csv("bitcoin_derivatives_comprehensive_4h.csv")
    s = pd.read_csv("bitcoin_fear_greed_index_4h.csv")
    o = pd.read_csv("bitcoin_onchain_data_4h.csv")
    k = pd.read_csv("BTCUSDT_4h_klines_with_factors.csv")

    print(f"Raw rows - d: {len(d)}, s: {len(s)}, o: {len(o)}, k: {len(k)}")

    for df, name in [(d, 'd'), (s, 's'), (o, 'o')]:
        first_col = df.columns[0]
        if first_col != 'timestamp':
            df.rename(columns={first_col: 'timestamp'}, inplace=True)

    def to_utc(series):
        if np.issubdtype(series.dtype, np.number):
            return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
        return pd.to_datetime(series, utc=True, errors="coerce")

    d["timestamp"] = to_utc(d["timestamp"])
    s["timestamp"] = to_utc(s["timestamp"])
    o["timestamp"] = to_utc(o["timestamp"])

    k["close_time"] = to_utc(k["close_time"])
    k["open_time"]  = to_utc(k["open_time"])
    k["timestamp"]  = k["close_time"]

    for df, name in [(d, 'd'), (s, 's'), (o, 'o'), (k, 'k')]:
        before = len(df)
        df.dropna(subset=["timestamp"], inplace=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        after = len(df)
        if before != after:
            print(f"{name}: Deduplicated/NA dropped {before} -> {after} rows")

    print("\nRetaining full OHLC price data...")
    
    price_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    price_data = k[price_columns].copy()
    
    print("Raw OHLC price stats:")
    for col in ['open', 'high', 'low', 'close']:
        if col in price_data.columns:
            prices = price_data[col]
            print(f"  {col}: {prices.min():.2f} - {prices.max():.2f}")
    
    price_data["y_ret_fwd1"] = np.log(price_data["close"]).shift(-1) - np.log(price_data["close"])
    
    drop_cols = {"open_time", "close_time", "ignore"}
    k_feats = k.drop(columns=[c for c in k.columns if c in drop_cols], errors="ignore").copy()

    d = d.rename(columns={"market_depth_2%":"market_depth_2pct"}, errors='ignore')
    
    if "value_classification" in s.columns:
        s["value_classification"] = s["value_classification"].astype("category")
        s = pd.concat(
            [s.drop(columns=["value_classification"]),
             pd.get_dummies(s["value_classification"], prefix="fgi_cls", dtype=int)],
            axis=1
        )

    def numcols(df):
        cols = ["timestamp"] + [c for c in df.columns if c!="timestamp" and pd.api.types.is_numeric_dtype(df[c])]
        return df[cols].copy()

    k_num = numcols(k_feats)
    d_num = numcols(d)
    s_num = numcols(s)
    o_num = numcols(o)

    print(f"Feature cols - k: {len(k_num.columns)-1}, d: {len(d_num.columns)-1}, s: {len(s_num.columns)-1}, o: {len(o_num.columns)-1}")

    print("\nMerging data...")
    
    df = price_data[["timestamp", "y_ret_fwd1", "open", "high", "low", "close", "volume"]].copy()
    
    df = df.merge(k_num, on="timestamp", how="left", suffixes=("", "_tech"))
    print(f"After merge k: {len(df)} rows, {len(df.columns)} cols")

    df = df.merge(d_num, on="timestamp", how="left")
    print(f"After merge d: {len(df)} rows, {len(df.columns)} cols")

    df = df.merge(s_num, on="timestamp", how="left")
    print(f"After merge s: {len(df)} rows, {len(df.columns)} cols")

    df = df.merge(o_num, on="timestamp", how="left")
    print(f"After merge o: {len(df)} rows, {len(df.columns)} cols")

    price_cols = ["open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df.columns if c not in ["timestamp", "y_ret_fwd1"] + price_cols]
    
    print(f"\nColumn type stats:")
    print(f"  Price cols: {len(price_cols)}")
    print(f"  Feature cols: {len(feature_cols)}")

    if DECISION_POINT == 'open':
        df[feature_cols] = df[feature_cols].shift(1)

    print(f"\nProcessing feature NaNs and outliers...")
    print(f"Feature NaNs before: {df[feature_cols].isna().sum().sum()}")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].ffill().bfill()

    for col in feature_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(med) else med)

    print(f"Feature NaNs after: {df[feature_cols].isna().sum().sum()}")

    q_low = df[feature_cols].quantile(WINSOR_Q[0])
    q_high = df[feature_cols].quantile(WINSOR_Q[1])
    df[feature_cols] = df[feature_cols].clip(q_low, q_high, axis=1)

    def safe_rolling_scale(s: pd.Series, window: int, min_unique: int = LOW_CARD_THRESHOLD):
        uniq = s.dropna().nunique()
        mu = s.rolling(window, min_periods=window).mean()
        
        if uniq < min_unique:
            return (s - mu)
        
        sd = s.rolling(window, min_periods=window).std(ddof=0)
        z = (s - mu) / sd.replace(0, 1.0)
        return z

    print("Applying rolling standardization to features...")
    df_z = df[["timestamp", "y_ret_fwd1"] + price_cols].copy()
    
    normalized_features = {}
    for i, c in enumerate(feature_cols):
        if (i + 1) % 20 == 0 or i == len(feature_cols) - 1:
            print(f"  Processing feature {i+1}/{len(feature_cols)}")
        normalized_features[c] = safe_rolling_scale(df[c], ROLL_N)

    df_z = pd.concat([df_z, pd.DataFrame(normalized_features)], axis=1)
    print(f"After standardization: {len(df_z)} rows")

    print("\nFinal data cleanup...")
    before_rows = len(df_z)
    
    start_idx = ROLL_N
    end_idx = -1
    df_final = df_z.iloc[start_idx:end_idx].copy()

    all_feature_cols = [c for c in df_final.columns if c not in ["timestamp", "y_ret_fwd1"] + price_cols]
    all_nan_cols = [c for c in all_feature_cols if df_final[c].isna().all()]
    
    if len(all_nan_cols) > 0:
        print(f"Removing {len(all_nan_cols)} all-NaN feature columns")
        df_final.drop(columns=all_nan_cols, inplace=True)

    all_feature_cols = [c for c in df_final.columns if c not in ["timestamp", "y_ret_fwd1"] + price_cols]
    
    df_final[all_feature_cols] = df_final[all_feature_cols].fillna(0.0)

    df_final = df_final[df_final["y_ret_fwd1"].notna()].reset_index(drop=True)

    after_rows = len(df_final)
    print(f"Data cleanup: {before_rows} -> {after_rows} rows")
    print(f"Final price cols: {len(price_cols)}")
    print(f"Final feature cols: {len(all_feature_cols)}")

    print("\nFinal data validation:")
    print(f"  Time range: {df_final['timestamp'].min()} to {df_final['timestamp'].max()}")
    print(f"  Sample count: {len(df_final)}")
    
    print(f"  Price column integrity:")
    for col in price_cols:
        if col in df_final.columns:
            values = df_final[col]
            print(f"    {col}: {values.min():.2f} - {values.max():.2f}, NaNs: {values.isna().sum()}")
    
    returns = df_final['y_ret_fwd1'].dropna()
    print(f"  Return stats: Mean={returns.mean():.6f}, Std={returns.std():.6f}")
    print(f"  Positive return ratio: {(returns > 0).mean():.3f}")

    if after_rows > 0:
        out_path = "btc_4h_dataset.parquet"
        df_final.to_parquet(out_path)
        print(f"\nSuccess! Full dataset saved to: {out_path}")
        print(f"   Final shape: {df_final.shape}")
        print(f"   Includes columns: {list(df_final.columns)}")
    else:
        print("\nError: Final dataset is empty!")

if __name__ == "__main__":
    main()