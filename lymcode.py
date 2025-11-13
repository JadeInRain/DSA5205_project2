import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression  # 只用于Platt校准
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost未安装！必须安装才能运行")
    print("   安装命令: pip install xgboost --break-system-packages")
    import sys
    sys.exit(1)

print("""
        完整量化交易策略开发流程
        BTC 4小时K线 | 样本外验证 | Walk-Forward测试
""")

# 全局配置参数

class Config:
    """全局配置类"""
    
    INPUT_CSV = "BTCUSDT_4h_klines_with_factors.csv"
    
    # 时间参数 - 缩短持仓期增加交易
    BAR_HOURS = 4
    HORIZON_DAYS = 3              # 🔥 改成3天
    HORIZON_BARS = 18             # 🔥 改成18
    TRAIN_DAYS = 365
    VAL_DAYS = 90
    STEP_DAYS = 90
    TRAIN_BARS = 2190
    VAL_BARS = 540
    STEP_BARS = 540
    
    # 标签阈值 - 降低标准
    UP_THRESH = 0.01              
    DOWN_THRESH = -0.01           
    
    ROLLING_WINDOW_BARS = 504
    
    TIMESTAMP_CANDIDATES = ["timestamp", "open_time", "time", "datetime"]
    PRICE_COLS = ["open", "high", "low", "close", "volume"]
    LABEL_COLS = ["y_up", "y_dn", "r_future_7d"]
    EXCLUDE_ALSO = ["future_close", "next_open"]
    
    # Holdout期间 - 测试2025年
    HOLDOUT_START = "2025-09-01"  # 
    HOLDOUT_END = "2025-11-05"
    
    RF_CFG = dict(n_estimators=100, max_depth=5, min_samples_split=20,
                  class_weight="balanced", random_state=42, n_jobs=-1)
    XGB_CFG = dict(n_estimators=100, learning_rate=0.1, max_depth=3,
                   subsample=0.8, random_state=42, n_jobs=-1)
    
    # 阈值网格 - 大幅降低
    TAU_UP_GRID = np.round(np.linspace(0.20, 0.40, 5), 3)   # 🔥 0.20-0.40
    TAU_DN_GRID = np.round(np.linspace(0.20, 0.40, 5), 3)   # 🔥 0.20-0.40
    MARGIN_GRID = np.round(np.linspace(0.00, 0.08, 3), 3)   # 🔥 0.00-0.08
    
    LEVERAGE = 3.0
    FEE_PER_SIDE = 0.0005
    ROUND_TRIP = 2 * FEE_PER_SIDE
    
    # 稳健性参数 - 降低要求
    MIN_TRADES_PER_FOLD = 2       
    GLOBAL_MIN_TRADES = 8         
    
    ANN_FACTOR = (365 * 24 / BAR_HOURS) * (7.0 / HORIZON_DAYS)  # 🔥 调整年化因子
    
    RANDOM_STATE = 42
    MODEL_KEYS = ["rf", "xgb"]
    FAST_MODE = True

# ========================
# 工具函数
# ========================
class Utils:
    """工具函数集合"""
    
    @staticmethod
    def create_directories():
        """创建所有必要的目录"""
        dirs = [
            "data/labels",
            "data/prepared",
            "data/splits",
            "data/predictions",
            "data/thresholds",
            "data/signals",
            "data/equity",
            "reports/step8",
            "figs"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    @staticmethod
    def find_timestamp_column(df: pd.DataFrame) -> str:
        """自动识别时间戳列"""
        cols = [c.lower() for c in df.columns]
        for c in Config.TIMESTAMP_CANDIDATES:
            if c in cols:
                return c
        # 尝试转换第一列
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                continue
        return df.columns[0]
    
    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    
    @staticmethod
    def print_step(step_num: int, title: str):
        """打印步骤标题"""
        print(f"\n{'='*80}")
        print(f"  STEP {step_num}: {title}")
        print(f"{'='*80}\n")


# ========================
# STEP 2: 标签构建
# ========================
class LabelBuilder:
    """标签构建类"""
    
    @staticmethod
    def build_labels(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """构建7天未来收益标签"""
        print("🏷️  构建标签...")
        out = df.copy()
        
        # 未来收盘价
        out["future_close"] = out["close"].shift(-Config.HORIZON_BARS)
        
        # 未来7天收益率
        out["r_future_7d"] = (out["future_close"] / out["close"]) - 1.0
        
        # 二分类标签
        out["y_up"] = (out["r_future_7d"] > Config.UP_THRESH).astype("Int64")
        out["y_dn"] = (out["r_future_7d"] < Config.DOWN_THRESH).astype("Int64")
        
        # 下一个开盘价（用于交易执行）
        if "open" in out.columns:
            out["next_open"] = out["open"].shift(-1)
        
        # 统计
        n_total = len(out)
        n_valid = int(out["r_future_7d"].notna().sum())
        up_rate = float(out["y_up"].dropna().mean())
        dn_rate = float(out["y_dn"].dropna().mean())
        
        print(f"   总行数: {n_total:,}")
        print(f"   有效标签: {n_valid:,}")
        print(f"   上涨比例: {up_rate:.2%}")
        print(f"   下跌比例: {dn_rate:.2%}")
        
        return out


# ========================
# STEP 3: 特征预处理
# ========================
class FeaturePreprocessor:
    """特征预处理类 - 防止数据泄露"""
    
    @staticmethod
    def select_features(df: pd.DataFrame, ts_col: str) -> List[str]:
        """选择特征列"""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        exclude = set([ts_col] + Config.PRICE_COLS + Config.LABEL_COLS + Config.EXCLUDE_ALSO)
        exclude |= {c for c in num_cols if c.startswith("future_") or c.startswith("label_")}
        
        features = [c for c in num_cols if c not in exclude]
        return features
    
    @staticmethod
    def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """强制转换为数值型"""
        out = df.copy()
        for c in cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out
    
    @staticmethod
    def rolling_standardize(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """滚动标准化 - 关键：shift(1)防止泄露"""
        print("📊 应用滚动标准化（防泄露）...")
        out = df.copy()
        n = len(out)
        window = Config.ROLLING_WINDOW_BARS
        
        # 如果数据太少，使用expanding
        if n < window + 50:
            print(f"   数据量较小，使用expanding标准化")
            for c in features:
                mean_exp = out[c].expanding(min_periods=20).mean().shift(1)
                std_exp = out[c].expanding(min_periods=20).std(ddof=0).shift(1)
                std_safe = std_exp.where(std_exp > 1e-12, 1e-12)
                out[c] = (out[c] - mean_exp) / std_safe
            return out
        
        # 正常滚动窗口
        minp = max(20, min(window // 4, 200))
        for c in features:
            roll_mean = out[c].rolling(window=window, min_periods=minp).mean().shift(1)
            roll_std = out[c].rolling(window=window, min_periods=minp).std(ddof=0).shift(1)
            std_safe = roll_std.where(roll_std > 1e-12, 1e-12)
            out[c] = (out[c] - roll_mean) / std_safe
        
        print(f"   滚动窗口: {window} bars")
        print(f"   特征数量: {len(features)}")
        
        return out


# ========================
# STEP 4: 时间序列划分
# ========================
class TimeSplitter:
    """时间序列划分类"""
    
    @staticmethod
    def build_rolling_folds(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """构建滚动折叠"""
        print("📅 构建滚动时间窗口...")
        
        # 只在Holdout之前构建fold
        pre_mask = df[ts_col] < pd.to_datetime(Config.HOLDOUT_START)
        pre_df = df.loc[pre_mask].reset_index(drop=True)
        n = len(pre_df)
        
        if n < Config.TRAIN_BARS + Config.VAL_BARS + 1:
            raise ValueError(f"数据不足以构建fold。需要>={Config.TRAIN_BARS + Config.VAL_BARS}行，实际{n}行")
        
        rows = []
        fold_id = 0
        start = 0
        
        while True:
            train_start = start
            train_end = train_start + Config.TRAIN_BARS
            val_start = train_end
            val_end = val_start + Config.VAL_BARS
            
            if val_end > n:
                break
            
            # 训练集
            rows.append({
                "fold_id": fold_id,
                "segment": "train",
                "start_idx": int(train_start),
                "end_idx": int(train_end - 1),
                "start_time": pre_df.loc[train_start, ts_col],
                "end_time": pre_df.loc[train_end - 1, ts_col],
                "length": int(Config.TRAIN_BARS)
            })
            
            # 验证集
            rows.append({
                "fold_id": fold_id,
                "segment": "val",
                "start_idx": int(val_start),
                "end_idx": int(val_end - 1),
                "start_time": pre_df.loc[val_start, ts_col],
                "end_time": pre_df.loc[val_end - 1, ts_col],
                "length": int(Config.VAL_BARS)
            })
            
            fold_id += 1
            start += Config.STEP_BARS
            
            if start + Config.TRAIN_BARS + Config.VAL_BARS > n:
                break
        
        folds_df = pd.DataFrame(rows)
        print(f"   创建了 {fold_id} 个fold")
        print(f"   训练窗口: {Config.TRAIN_DAYS}天 ({Config.TRAIN_BARS}根)")
        print(f"   验证窗口: {Config.VAL_DAYS}天 ({Config.VAL_BARS}根)")
        print(f"   步进: {Config.STEP_DAYS}天 ({Config.STEP_BARS}根)")
        
        return folds_df
    
    @staticmethod
    def build_holdout(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """构建Holdout集"""
        h_start = pd.to_datetime(Config.HOLDOUT_START)
        h_end = pd.to_datetime(Config.HOLDOUT_END)
        mask = (df[ts_col] >= h_start) & (df[ts_col] <= h_end)
        sub = df.loc[mask, [ts_col]].copy()
        
        if sub.empty:
            raise ValueError("Holdout范围无数据")
        
        print(f"   Holdout: {sub[ts_col].iloc[0]} 至 {sub[ts_col].iloc[-1]}")
        print(f"   Holdout行数: {len(sub):,}")
        
        return pd.DataFrame({
            "start_time": [sub[ts_col].iloc[0]],
            "end_time": [sub[ts_col].iloc[-1]],
            "start_idx": [int(sub.index[0])],
            "end_idx": [int(sub.index[-1])],
            "n_rows": [int(len(sub))]
        })


# ========================
# STEP 5: 模型训练和概率校准
# ========================
class ModelTrainer:
    """模型训练和概率校准"""
    
    @staticmethod
    def fit_models(X_tr, y_tr):
        """训练2个模型 - 极速版"""
        m_rf = RandomForestClassifier(**Config.RF_CFG).fit(X_tr, y_tr)
        m_xgb = XGBClassifier(**Config.XGB_CFG, use_label_encoder=False, eval_metric='logloss').fit(X_tr, y_tr)
        return m_rf, m_xgb
    
    @staticmethod
    def score_raw(model, X):
        """获取原始分数"""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            return model.decision_function(X)
        else:
            return model.predict(X)
    
    @staticmethod
    def platt_calibrate(val_scores, y_val):
        """Platt缩放校准"""
        scaler = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
        s = np.asarray(val_scores).reshape(-1, 1)
        y = y_val.astype(int).reshape(-1)
        scaler.fit(s, y)
        return scaler
    
    @staticmethod
    def eval_calibrated(p, y):
        """评估校准后的概率"""
        y = y.astype(int).reshape(-1)
        try:
            auc = roc_auc_score(y, p)
        except:
            auc = np.nan
        try:
            ap = average_precision_score(y, p)
        except:
            ap = np.nan
        try:
            brier = brier_score_loss(y, p)
        except:
            brier = np.nan
        return dict(auc=auc, ap=ap, brier=brier)
    
    @staticmethod
    def train_all_folds(df: pd.DataFrame, folds: pd.DataFrame, features: List[str]):
        """训练所有fold"""
        print("🤖 训练模型和概率校准...")
        
        all_val_outputs = []
        fold_ids = sorted(folds["fold_id"].unique().tolist())
        
        for fid in fold_ids:
            print(f"\n   Fold {fid}/{len(fold_ids)-1}")
            
            # 获取训练和验证索引
            ftrain = folds[(folds["fold_id"] == fid) & (folds["segment"] == "train")].iloc[0]
            fval = folds[(folds["fold_id"] == fid) & (folds["segment"] == "val")].iloc[0]
            
            tr_idx = slice(int(ftrain["start_idx"]), int(ftrain["end_idx"]) + 1)
            va_idx = slice(int(fval["start_idx"]), int(fval["end_idx"]) + 1)
            
            # 提取数据
            X_tr = df.loc[tr_idx, features].values
            X_va = df.loc[va_idx, features].values
            
            y_up_tr = df.loc[tr_idx, "y_up"].astype("Int64")
            y_up_va = df.loc[va_idx, "y_up"].astype("Int64")
            y_dn_tr = df.loc[tr_idx, "y_dn"].astype("Int64")
            y_dn_va = df.loc[va_idx, "y_dn"].astype("Int64")
            
            # 过滤NaN
            tr_mask = np.isfinite(X_tr).all(axis=1) & y_up_tr.notna().values & y_dn_tr.notna().values
            va_mask = np.isfinite(X_va).all(axis=1) & y_up_va.notna().values & y_dn_va.notna().values
            
            X_tr2 = X_tr[tr_mask]
            X_va2 = X_va[va_mask]
            y_up_tr2 = y_up_tr[tr_mask].astype(int).values
            y_up_va2 = y_up_va[va_mask].astype(int).values
            y_dn_tr2 = y_dn_tr[tr_mask].astype(int).values
            y_dn_va2 = y_dn_va[va_mask].astype(int).values
            
            if len(X_tr2) < 100 or len(X_va2) < 20:
                print(f"      样本不足，跳过")
                continue
            
            # 训练UP模型 - 极速版
            up_rf, up_xgb = ModelTrainer.fit_models(X_tr2, y_up_tr2)
            up_s_rf = ModelTrainer.score_raw(up_rf, X_va2)
            up_s_xgb = ModelTrainer.score_raw(up_xgb, X_va2)
            
            up_platt_rf = ModelTrainer.platt_calibrate(up_s_rf, y_up_va2)
            up_platt_xgb = ModelTrainer.platt_calibrate(up_s_xgb, y_up_va2)
            
            up_p_rf = up_platt_rf.predict_proba(np.asarray(up_s_rf).reshape(-1, 1))[:, 1]
            up_p_xgb = up_platt_xgb.predict_proba(np.asarray(up_s_xgb).reshape(-1, 1))[:, 1]
            
            # 训练DOWN模型 - 极速版
            dn_rf, dn_xgb = ModelTrainer.fit_models(X_tr2, y_dn_tr2)
            dn_s_rf = ModelTrainer.score_raw(dn_rf, X_va2)
            dn_s_xgb = ModelTrainer.score_raw(dn_xgb, X_va2)
            
            dn_platt_rf = ModelTrainer.platt_calibrate(dn_s_rf, y_dn_va2)
            dn_platt_xgb = ModelTrainer.platt_calibrate(dn_s_xgb, y_dn_va2)
            
            dn_p_rf = dn_platt_rf.predict_proba(np.asarray(dn_s_rf).reshape(-1, 1))[:, 1]
            dn_p_xgb = dn_platt_xgb.predict_proba(np.asarray(dn_s_xgb).reshape(-1, 1))[:, 1]
            
            # 评估
            up_eval_rf = ModelTrainer.eval_calibrated(up_p_rf, y_up_va2)
            up_eval_xgb = ModelTrainer.eval_calibrated(up_p_xgb, y_up_va2)
            dn_eval_rf = ModelTrainer.eval_calibrated(dn_p_rf, y_dn_va2)
            dn_eval_xgb = ModelTrainer.eval_calibrated(dn_p_xgb, y_dn_va2)
            
            print(f"      UP AUC: rf={up_eval_rf['auc']:.3f} | xgb={up_eval_xgb['auc']:.3f}")
            print(f"      DN AUC: rf={dn_eval_rf['auc']:.3f} | xgb={dn_eval_xgb['auc']:.3f}")
            
            # 保存验证集预测
            ts_col = Utils.find_timestamp_column(df)
            va_df = df.loc[va_idx, [ts_col, "y_up", "y_dn"]].copy()
            va_df = va_df.loc[va_mask].reset_index(drop=True)
            va_df.rename(columns={ts_col: "time"}, inplace=True)
            
            out = pd.DataFrame({
                "fold_id": fid,
                "time": va_df["time"],
                "y_up": va_df["y_up"].astype(int),
                "y_dn": va_df["y_dn"].astype(int),
                "p_up_rf": up_p_rf,
                "p_up_xgb": up_p_xgb,
                "p_dn_rf": dn_p_rf,
                "p_dn_xgb": dn_p_xgb,
            })
            
            out.to_csv(f"data/predictions/val_fold_{fid}.csv", index=False)
            all_val_outputs.append(out)
        
        # 合并所有fold
        if all_val_outputs:
            cat = pd.concat(all_val_outputs, axis=0, ignore_index=True)
            cat.to_csv("data/predictions/val_all_folds.csv", index=False)
            print(f"\n   ✅ 所有fold预测已保存")
        
        return all_val_outputs


# ========================
# STEP 6: 阈值搜索
# ========================
class ThresholdSearcher:
    """阈值搜索类"""
    
    @staticmethod
    def simulate_fold(df_fold, tau_up, tau_dn, margin, model_key):
        """模拟单个fold的交易"""
        p_up = df_fold[f"p_up_{model_key}"].values
        p_dn = df_fold[f"p_dn_{model_key}"].values
        r7 = df_fold["r_future_7d"].values
        
        equity = 1.0
        path = [equity]
        i = 0
        n = len(df_fold)
        trades = 0
        wins = 0
        
        while i < n:
            # 多头信号
            long_sig = (p_up[i] >= tau_up) and (
                (p_dn[i] < tau_dn) or 
                ((p_dn[i] >= tau_dn) and (p_up[i] - p_dn[i] > margin))
            )
            
            # 空头信号
            short_sig = (p_dn[i] >= tau_dn) and (
                (p_up[i] < tau_up) or
                ((p_up[i] >= tau_up) and (p_dn[i] - p_up[i] > margin))
            )
            
            if long_sig and not short_sig:
                net = Config.LEVERAGE * r7[i] - Config.ROUND_TRIP
                equity *= (1.0 + net)
                trades += 1
                wins += (net > 0)
                for _ in range(min(Config.HORIZON_BARS, n - i)):
                    path.append(equity)
                i += Config.HORIZON_BARS
            elif short_sig and not long_sig:
                net = Config.LEVERAGE * (-r7[i]) - Config.ROUND_TRIP
                equity *= (1.0 + net)
                trades += 1
                wins += (net > 0)
                for _ in range(min(Config.HORIZON_BARS, n - i)):
                    path.append(equity)
                i += Config.HORIZON_BARS
            else:
                i += 1
                path.append(equity)
        
        # 计算Sharpe
        ret_series = pd.Series(path).pct_change().dropna()
        mu = ret_series.mean() * Config.ANN_FACTOR
        sd = ret_series.std(ddof=0) * np.sqrt(Config.ANN_FACTOR)
        sharpe = (mu / sd) if (sd and sd > 0) else np.nan
        win_rate = (wins / trades) if trades > 0 else np.nan
        
        return {
            "final_equity": float(equity),
            "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "trades": int(trades),
            "win_rate": float(win_rate) if pd.notna(win_rate) else np.nan
        }
    
    @staticmethod
    def grid_search(df):
        """网格搜索最优阈值"""
        print("🔍 网格搜索最优阈值...")
        print(f"   搜索空间: {len(Config.TAU_UP_GRID)} × {len(Config.TAU_DN_GRID)} × {len(Config.MARGIN_GRID)} = {len(Config.TAU_UP_GRID) * len(Config.TAU_DN_GRID) * len(Config.MARGIN_GRID)}种组合")
        
        rows = []
        total = len(Config.MODEL_KEYS) * len(Config.TAU_UP_GRID) * len(Config.TAU_DN_GRID) * len(Config.MARGIN_GRID)
        count = 0
        
        for mk in Config.MODEL_KEYS:
            for tu in Config.TAU_UP_GRID:
                for td in Config.TAU_DN_GRID:
                    for m in Config.MARGIN_GRID:
                        count += 1
                        if count % 100 == 0:
                            print(f"      进度: {count}/{total}")
                        
                        fold_stats = []
                        for fid, sub in df.groupby("fold_id"):
                            sub = sub.sort_values("time")
                            stats = ThresholdSearcher.simulate_fold(sub, tu, td, m, mk)
                            if stats["trades"] >= Config.MIN_TRADES_PER_FOLD:
                                fold_stats.append(stats)
                        
                        if fold_stats:
                            sharpe_vals = [s["sharpe"] for s in fold_stats if pd.notna(s["sharpe"])]
                            median_sharpe = float(np.median(sharpe_vals)) if sharpe_vals else np.nan
                            geo_equity = float(np.prod([s["final_equity"] for s in fold_stats]) ** (1.0 / len(fold_stats)))
                            avg_trades = float(np.mean([s["trades"] for s in fold_stats]))
                            total_trades = int(np.sum([s["trades"] for s in fold_stats]))
                            avg_win = float(np.mean([s["win_rate"] for s in fold_stats if pd.notna(s["win_rate"])])) if fold_stats else np.nan
                        else:
                            median_sharpe = np.nan
                            geo_equity = np.nan
                            avg_trades = 0.0
                            total_trades = 0
                            avg_win = np.nan
                        
                        rows.append({
                            "model": mk,
                            "tau_up": tu,
                            "tau_dn": td,
                            "margin": m,
                            "median_sharpe": median_sharpe,
                            "geo_equity": geo_equity,
                            "avg_trades": avg_trades,
                            "total_trades": total_trades,
                            "avg_win_rate": avg_win
                        })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def select_best(df_grid):
        """选择最优参数"""
        cand = df_grid[df_grid["total_trades"] >= Config.GLOBAL_MIN_TRADES].copy()
        if cand.empty:
            cand = df_grid.copy()
        
        winners = []
        for mk in Config.MODEL_KEYS:
            sub = cand[cand["model"] == mk].copy()
            sub = sub.sort_values(
                by=["median_sharpe", "geo_equity", "avg_trades"],
                ascending=[False, False, False]
            )
            if len(sub):
                winners.append(sub.iloc[0].to_dict())
        
        best = pd.DataFrame(winners)
        
        # 保存
        for mk in Config.MODEL_KEYS:
            df_grid[df_grid["model"] == mk].to_csv(f"data/thresholds/grid_scores_{mk}.csv", index=False)
        
        best.to_csv("data/thresholds/best_thresholds_per_model.csv", index=False)
        
        print("\n   ✅ 最优阈值:")
        if not best.empty:
            for _, row in best.iterrows():
                print(f"      {row['model']:8s}: tau_up={row['tau_up']:.3f}, tau_dn={row['tau_dn']:.3f}, "
                      f"margin={row['margin']:.3f}, Sharpe={row['median_sharpe']:.3f}")
        
        return best


# ========================
# STEP 7: Walk-Forward测试
# ========================
class WalkForwardTester:
    """Walk-Forward样本外测试"""
    
    @staticmethod
    def load_best_thresholds():
        """加载最优阈值"""
        best = pd.read_csv("data/thresholds/best_thresholds_per_model.csv")
        best.columns = [c.strip().lower() for c in best.columns]
        # 选择Sharpe最高的模型
        best = best.sort_values("median_sharpe", ascending=False).reset_index(drop=True)
        return best.iloc[0]
    
    @staticmethod
    def walk_forward_test_fast(df: pd.DataFrame, features: List[str], folds: pd.DataFrame,
                            holdout_df: pd.DataFrame, segment_name: str):
        """快速Walk-Forward：使用最后fold的模型直接预测（不重训练）"""
        print(f"\n🚶 Walk-Forward测试 - 快速模式 ({segment_name})...")
    
        # 加载最优参数
        best = WalkForwardTester.load_best_thresholds()
        model_key = str(best["model"])
        tau_up = float(best["tau_up"])
        tau_dn = float(best["tau_dn"])
        margin = float(best["margin"])
        
        print(f"   使用模型: {model_key}")
        print(f"   阈值: tau_up={tau_up:.3f}, tau_dn={tau_dn:.3f}, margin={margin:.3f}")
        print(f"   ⚡ 快速模式: 使用最后fold的模型（不重训练）")
        
        # 使用最后一个fold训练最终模型
        last_fold = folds[folds["segment"] == "train"].iloc[-1]
        train_start = int(last_fold["start_idx"])
        train_end = int(last_fold["end_idx"])
        
        print(f"   训练最终模型: {train_start} to {train_end}")
        
        X_train = df.loc[train_start:train_end, features].values
        y_up_train = df.loc[train_start:train_end, "y_up"].astype("Int64")
        y_dn_train = df.loc[train_start:train_end, "y_dn"].astype("Int64")
        
        # 过滤
        train_mask = np.isfinite(X_train).all(axis=1) & y_up_train.notna().values & y_dn_train.notna().values
        X_train_clean = X_train[train_mask]
        y_up_clean = y_up_train[train_mask].astype(int).values
        y_dn_clean = y_dn_train[train_mask].astype(int).values
        
        # 训练模型
        up_rf, up_xgb = ModelTrainer.fit_models(X_train_clean, y_up_clean)
        dn_rf, dn_xgb = ModelTrainer.fit_models(X_train_clean, y_dn_clean)
        
        up_models = {"rf": up_rf, "xgb": up_xgb}
        dn_models = {"rf": dn_rf, "xgb": dn_xgb}
        
        up_model = up_models.get(model_key, up_rf)
        dn_model = dn_models.get(model_key, dn_rf)
        
        # 使用最后的validation window做Platt校准
        last_val_fold = folds[folds["segment"] == "val"].iloc[-1]
        val_start = int(last_val_fold["start_idx"])
        val_end = int(last_val_fold["end_idx"])
        
        X_val = df.loc[val_start:val_end, features].values
        y_up_val = df.loc[val_start:val_end, "y_up"].astype("Int64")
        y_dn_val = df.loc[val_start:val_end, "y_dn"].astype("Int64")
        
        val_mask = np.isfinite(X_val).all(axis=1) & y_up_val.notna().values & y_dn_val.notna().values
        X_val_clean = X_val[val_mask]
        y_up_val_clean = y_up_val[val_mask].astype(int).values
        y_dn_val_clean = y_dn_val[val_mask].astype(int).values
        
        # Platt校准
        up_s_val = ModelTrainer.score_raw(up_model, X_val_clean)
        dn_s_val = ModelTrainer.score_raw(dn_model, X_val_clean)
        
        up_platt = ModelTrainer.platt_calibrate(up_s_val, y_up_val_clean)
        dn_platt = ModelTrainer.platt_calibrate(dn_s_val, y_dn_val_clean)
        
        # 🔥 初始化变量 - 关键！
        trades = []
        equity_path = []
        current_equity = 1.0
        ts_col = Utils.find_timestamp_column(df)
        
        # 在测试集上预测
        test_start_idx = int(holdout_df.iloc[0]["start_idx"])
        test_end_idx = int(holdout_df.iloc[0]["end_idx"])
        
        # 限制测试长度
        MAX_TEST_BARS = 1000
        test_end_idx = min(test_end_idx, test_start_idx + MAX_TEST_BARS)

        # 确保不超边界
        max_valid_idx = len(df) - 1
        if test_end_idx > max_valid_idx:
            print(f"   ⚠️  调整end_idx: {test_end_idx} → {max_valid_idx}")
            test_end_idx = max_valid_idx

        print(f"   📊 测试范围: {test_start_idx} → {test_end_idx} (共{test_end_idx-test_start_idx}条)")
        
        # 🔥 初始化第一个点
        if test_start_idx < len(df):
            equity_path.append({"time": df.loc[test_start_idx, ts_col], "equity": current_equity})
        
        # 主循环
        i = test_start_idx
        while i <= test_end_idx:
            if (i - test_start_idx) % 100 == 0:
                print(f"      进度: {i-test_start_idx}/{test_end_idx-test_start_idx}")
            
            # 边界检查
            if i >= len(df):
                break
            
            # 获取当前bar特征
            X_t = df.loc[[i], features].values
            
            if not np.isfinite(X_t).all():
                i += 1
                if i <= test_end_idx and i < len(df):
                    equity_path.append({"time": df.loc[i, ts_col], "equity": current_equity})
                continue
            
            # 预测
            up_s_t = ModelTrainer.score_raw(up_model, X_t)
            dn_s_t = ModelTrainer.score_raw(dn_model, X_t)
            
            p_up = float(up_platt.predict_proba(np.asarray(up_s_t).reshape(-1, 1))[:, 1][0])
            p_dn = float(dn_platt.predict_proba(np.asarray(dn_s_t).reshape(-1, 1))[:, 1][0])
            
            # 生成信号
            long_sig = (p_up >= tau_up) and ((p_dn < tau_dn) or ((p_dn >= tau_dn) and (p_up - p_dn > margin)))
            short_sig = (p_dn >= tau_dn) and ((p_up < tau_up) or ((p_up >= tau_up) and (p_dn - p_up > margin)))
            
            # 执行交易
            if (long_sig or short_sig) and i + Config.HORIZON_BARS <= test_end_idx:
                r7 = df.loc[i, "r_future_7d"]
                if pd.notna(r7):
                    if long_sig:
                        side = "long"
                        net = Config.LEVERAGE * r7 - Config.ROUND_TRIP
                    else:
                        side = "short"
                        net = Config.LEVERAGE * (-r7) - Config.ROUND_TRIP
                    
                    entry_price = df.loc[i, "next_open"] if "next_open" in df.columns else df.loc[i, "close"]
                    exit_price = entry_price * (1 + r7)
                    
                    trades.append({
                        "time": df.loc[i, ts_col],
                        "side": side,
                        "entry": float(entry_price),
                        "exit": float(exit_price),
                        "return": float(r7),
                        "net": float(net),
                        "p_up": p_up,
                        "p_dn": p_dn
                    })
                    
                    current_equity *= (1 + net)
                    
                    # 持仓期
                    # 🔥 持仓期逐bar mark-to-market
                    entry_price_raw = df.loc[i, "next_open"] if "next_open" in df.columns else df.loc[i, "close"]
                    entry_equity = current_equity

                    for j in range(min(Config.HORIZON_BARS, test_end_idx - i + 1)):
                        idx = i + j
                        if idx <= test_end_idx and idx < len(df):
                            # 获取当前价格
                            current_price = df.loc[idx, "close"]
                            
                            # 计算浮动盈亏
                            if side == "long":
                                unrealized_return = (current_price / entry_price_raw - 1) * Config.LEVERAGE
                            else:  # short
                                unrealized_return = (1 - current_price / entry_price_raw) * Config.LEVERAGE
                            
                            # Mark-to-market净值
                            mtm_equity = entry_equity * (1 + unrealized_return - Config.ROUND_TRIP)
                            equity_path.append({"time": df.loc[idx, ts_col], "equity": mtm_equity})

                    # 最终结算
                    current_equity *= (1 + net)
                    i += Config.HORIZON_BARS
                    for j in range(min(Config.HORIZON_BARS, test_end_idx - i + 1)):
                        idx = i + j
                        if idx <= test_end_idx and idx < len(df):
                            equity_path.append({"time": df.loc[idx, ts_col], "equity": current_equity})
                    i += Config.HORIZON_BARS
                else:
                    i += 1
                    if i <= test_end_idx and i < len(df):
                        equity_path.append({"time": df.loc[i, ts_col], "equity": current_equity})
            else:
                i += 1
                if i <= test_end_idx and i < len(df):
                    equity_path.append({"time": df.loc[i, ts_col], "equity": current_equity})
        
        # 保存结果
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(f"data/signals/{segment_name}_trades.csv", index=False)
            print(f"   ✅ 交易数: {len(trades)}")
            print(f"   ✅ 胜率: {(trades_df['net'] > 0).mean():.2%}")
        else:
            trades_df = pd.DataFrame()
            print(f"   ⚠️  没有产生交易")
        
        if equity_path:
            equity_df = pd.DataFrame(equity_path)
            equity_df.to_csv(f"data/equity/{segment_name}_equity.csv", index=False)
            print(f"   ✅ 最终净值: {current_equity:.4f}")
        else:
            # 如果还是空，创建一个默认的
            equity_df = pd.DataFrame({
                "time": [df.loc[test_start_idx, ts_col]],
                "equity": [1.0]
            })
            equity_df.to_csv(f"data/equity/{segment_name}_equity.csv", index=False)
            print(f"   ⚠️  净值路径为空，已创建默认值")
        
        return trades_df, equity_df
        @staticmethod
        def walk_forward_test(df: pd.DataFrame, features: List[str], holdout_df: pd.DataFrame, 
                            segment_name: str):
            """执行Walk-Forward测试"""
            print(f"\n🚶 Walk-Forward测试 ({segment_name})...")
            
            # 加载最优参数
            best = WalkForwardTester.load_best_thresholds()
            model_key = str(best["model"])
            tau_up = float(best["tau_up"])
            tau_dn = float(best["tau_dn"])
            margin = float(best["margin"])
            
            print(f"   使用模型: {model_key}")
            print(f"   阈值: tau_up={tau_up:.3f}, tau_dn={tau_dn:.3f}, margin={margin:.3f}")
            
            # 确定测试范围
            test_start_idx = int(holdout_df.iloc[0]["start_idx"])
            test_end_idx = int(holdout_df.iloc[0]["end_idx"])
            
            trades = []
            equity_path = []
            
            current_equity = 1.0
            ts_col = Utils.find_timestamp_column(df)
            
            i = test_start_idx
            while i <= test_end_idx:
                if i % 100 == 0:
                    print(f"      进度: {i-test_start_idx}/{test_end_idx-test_start_idx}")
                
                # 训练窗口
                train_start = i - (Config.TRAIN_BARS + Config.VAL_BARS)
                val_start = i - Config.VAL_BARS
                
                if train_start < 0:
                    i += 1
                    if i < len(df):  # 边界检查
                        equity_path.append({"time": df.loc[i, ts_col], "equity": current_equity})
                    continue
                
                # 提取训练和验证数据
                X_tr = df.loc[train_start:val_start-1, features].values
                X_va = df.loc[val_start:i-1, features].values
                X_t = df.loc[[i], features].values
                
                y_up_tr = df.loc[train_start:val_start-1, "y_up"].astype("Int64")
                y_up_va = df.loc[val_start:i-1, "y_up"].astype("Int64")
                y_dn_tr = df.loc[train_start:val_start-1, "y_dn"].astype("Int64")
                y_dn_va = df.loc[val_start:i-1, "y_dn"].astype("Int64")
                
                # 过滤
                tr_mask = np.isfinite(X_tr).all(axis=1) & y_up_tr.notna().values & y_dn_tr.notna().values
                va_mask = np.isfinite(X_va).all(axis=1) & y_up_va.notna().values & y_dn_va.notna().values
                
                if tr_mask.sum() < 50 or va_mask.sum() < 20 or not np.isfinite(X_t).all():
                    i += 1
                    if i < len(df):  # 边界检查
                        equity_path.append({"time": df.loc[i, ts_col], "equity": current_equity})
                    continue
                
                Xtr = X_tr[tr_mask]
                Xva = X_va[va_mask]
                yup_tr = y_up_tr[tr_mask].astype(int).values
                yup_va = y_up_va[va_mask].astype(int).values
                ydn_tr = y_dn_tr[tr_mask].astype(int).values
                ydn_va = y_dn_va[va_mask].astype(int).values
                
                # 训练UP模型 - 极速版
                up_rf, up_xgb = ModelTrainer.fit_models(Xtr, yup_tr)
                up_models = {"rf": up_rf, "xgb": up_xgb}
                up_model = up_models.get(model_key, up_rf)  # 默认RF
                
                up_s_va = ModelTrainer.score_raw(up_model, Xva)
                up_platt = ModelTrainer.platt_calibrate(up_s_va, yup_va)
                up_s_t = ModelTrainer.score_raw(up_model, X_t)
                p_up = float(up_platt.predict_proba(np.asarray(up_s_t).reshape(-1, 1))[:, 1][0])
                
                # 训练DN模型 - 极速版
                dn_rf, dn_xgb = ModelTrainer.fit_models(Xtr, ydn_tr)
                dn_models = {"rf": dn_rf, "xgb": dn_xgb}
                dn_model = dn_models.get(model_key, dn_rf)  # 默认RF
                
                dn_s_va = ModelTrainer.score_raw(dn_model, Xva)
                dn_platt = ModelTrainer.platt_calibrate(dn_s_va, ydn_va)
                dn_s_t = ModelTrainer.score_raw(dn_model, X_t)
                p_dn = float(dn_platt.predict_proba(np.asarray(dn_s_t).reshape(-1, 1))[:, 1][0])
                
                # 生成信号
                long_sig = (p_up >= tau_up) and ((p_dn < tau_dn) or ((p_dn >= tau_dn) and (p_up - p_dn > margin)))
                short_sig = (p_dn >= tau_dn) and ((p_up < tau_up) or ((p_up >= tau_up) and (p_dn - p_up > margin)))
                
                # 执行交易
                if (long_sig or short_sig) and i + Config.HORIZON_BARS <= test_end_idx:
                    r7 = df.loc[i, "r_future_7d"]
                    if pd.notna(r7):
                        if long_sig:
                            side = "long"
                            net = Config.LEVERAGE * r7 - Config.ROUND_TRIP
                        else:
                            side = "short"
                            net = Config.LEVERAGE * (-r7) - Config.ROUND_TRIP
                        
                        entry_price = df.loc[i, "next_open"] if "next_open" in df.columns else df.loc[i, "close"]
                        exit_price = entry_price * (1 + r7)
                        
                        trades.append({
                            "time": df.loc[i, ts_col],
                            "side": side,
                            "entry": float(entry_price),
                            "exit": float(exit_price),
                            "return": float(r7),
                            "net": float(net),
                            "p_up": p_up,
                            "p_dn": p_dn
                        })
                        
                        current_equity *= (1 + net)
                        
                        # 跳过持仓期
                        for j in range(min(Config.HORIZON_BARS, test_end_idx - i + 1)):
                            if i + j < len(df):  # 边界检查
                                equity_path.append({"time": df.loc[i + j, ts_col], "equity": current_equity})
                        i += Config.HORIZON_BARS
                    else:
                        i += 1
                        equity_path.append({"time": df.loc[i, ts_col], "equity": current_equity})
                else:
                    i += 1
                    if i < len(df):  # 边界检查
                        equity_path.append({"time": df.loc[i, ts_col], "equity": current_equity})
            
            # 保存结果
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_csv(f"data/signals/{segment_name}_trades.csv", index=False)
                print(f"   ✅ 交易数: {len(trades)}")
                print(f"   ✅ 胜率: {(trades_df['net'] > 0).mean():.2%}")
            
            if equity_path:
                equity_df = pd.DataFrame(equity_path)
                equity_df.to_csv(f"data/equity/{segment_name}_equity.csv", index=False)
                print(f"   ✅ 最终净值: {current_equity:.4f}")
            
            return trades_df if trades else pd.DataFrame(), equity_df if equity_path else pd.DataFrame()

# ========================
# STEP 8: 结果报告
# ========================
class ReportGenerator:
    """结果报告生成器"""
    
    @staticmethod
    def load_base_prices(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """加载基准价格"""
        if "close" not in df.columns:
            if "open" in df.columns:
                df["close"] = df["open"]
            else:
                raise ValueError("缺少价格列")
        return df[[ts_col, "close"]].rename(columns={ts_col: "time"}).dropna().sort_values("time")
    
    @staticmethod
    def calculate_metrics(eq: pd.DataFrame) -> dict:
        """计算性能指标"""
        eq = eq.dropna().sort_values("time").reset_index(drop=True)
        if eq.empty:
            return dict(final_equity=np.nan, cagr=np.nan, sharpe=np.nan, maxdd=np.nan, bars=0)
        
        r = eq["equity"].pct_change().dropna()
        
        # CAGR
        if len(eq) > 1:
            years = (eq["time"].iloc[-1] - eq["time"].iloc[0]).total_seconds() / (365 * 24 * 3600)
            cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1 / max(years, 1e-9)) - 1 if years > 0 else np.nan
        else:
            cagr = np.nan
        
        # Sharpe
        mu = r.mean() * Config.ANN_FACTOR
        sd = r.std(ddof=0) * np.sqrt(Config.ANN_FACTOR)
        sharpe = (mu / sd) if (sd and sd > 0) else np.nan
        
        # 最大回撤
        cum = eq["equity"].values
        peak = np.maximum.accumulate(cum)
        maxdd = (cum / peak - 1.0).min()
        
        return dict(
            final_equity=float(eq["equity"].iloc[-1]),
            cagr=float(cagr) if pd.notna(cagr) else np.nan,
            sharpe=float(sharpe) if pd.notna(sharpe) else np.nan,
            maxdd=float(maxdd) if pd.notna(maxdd) else np.nan,
            bars=int(len(eq))
        )
    
    @staticmethod
    def trade_stats(trades: pd.DataFrame) -> dict:
        """交易统计"""
        if trades is None or trades.empty:
            return dict(n_trades=0, win_rate=np.nan, avg_net=np.nan, med_net=np.nan)
        wins = (trades["net"] > 0).mean() if "net" in trades.columns else np.nan
        return dict(
            n_trades=int(len(trades)),
            win_rate=float(wins) if pd.notna(wins) else np.nan,
            avg_net=float(trades["net"].mean()) if "net" in trades.columns else np.nan,
            med_net=float(trades["net"].median()) if "net" in trades.columns else np.nan,
        )
    
    @staticmethod
    def bh_equity(base: pd.DataFrame, start_t, end_t, leverage=1.0) -> pd.DataFrame:
        """Buy & Hold基准"""
        px = base[(base["time"] >= start_t) & (base["time"] <= end_t)].copy()
        px = px.sort_values("time").reset_index(drop=True)
        if px.empty:
            return pd.DataFrame(columns=["time", "equity"])
        ret = px["close"].pct_change().fillna(0.0) * leverage
        eq = (1 + ret).cumprod()
        return pd.DataFrame({"time": px["time"], "equity": eq})
    
    @staticmethod
    def align_to_strategy(eq_bh: pd.DataFrame, eq_strat: pd.DataFrame) -> pd.DataFrame:
        """对齐到策略时间戳"""
        a = pd.merge_asof(
            eq_strat[["time"]].sort_values("time"),
            eq_bh.sort_values("time"),
            on="time",
            direction="backward"
        )
        a["equity"] = a["equity"].ffill().bfill()
        return a
    
    @staticmethod
    def plot_equity(eq_strat: pd.DataFrame, eq_bh1: pd.DataFrame, eq_bh3: pd.DataFrame,
                   title: str, out_path: str):
        """绘制净值曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(eq_strat["time"].values.ravel(), eq_strat["equity"].values.ravel(),
         label="Strategy", linewidth=2)
        plt.plot(eq_bh1["time"].values.ravel(), eq_bh1["equity"].values.ravel(),
         label="Buy & Hold 1x", linewidth=2)
        plt.plot(eq_bh3["time"].values.ravel(), eq_bh3["equity"].values.ravel(),
         label="Buy & Hold 3x", linewidth=2)
        ax = plt.gca()
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Time", fontsize=11)
        plt.ylabel("Equity", fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    
    @staticmethod
    def generate_report(df: pd.DataFrame, segment_name: str):
        """生成单个段的报告"""
        print(f"\n📊 生成 {segment_name} 报告...")
        
        eq_path = f"data/equity/{segment_name}_equity.csv"
        trades_path = f"data/signals/{segment_name}_trades.csv"
        
        if not os.path.exists(eq_path):
            print(f"   ⚠️  净值文件不存在，跳过")
            return
        
        # 加载数据
        eq = pd.read_csv(eq_path)
        eq["time"] = pd.to_datetime(eq["time"], errors="coerce")
        eq = eq.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        
        trades = None
        if os.path.exists(trades_path):
            trades = pd.read_csv(trades_path)
            if "time" in trades.columns:
                trades["time"] = pd.to_datetime(trades["time"], errors="coerce")
        
        # 计算指标
        kpis = ReportGenerator.calculate_metrics(eq)
        tstats = ReportGenerator.trade_stats(trades)
        
        # 基准
        ts_col = Utils.find_timestamp_column(df)
        base = ReportGenerator.load_base_prices(df, ts_col)
        start_t, end_t = eq["time"].iloc[0], eq["time"].iloc[-1]
        
        bh1 = ReportGenerator.bh_equity(base, start_t, end_t, leverage=1.0)
        bh3 = ReportGenerator.bh_equity(base, start_t, end_t, leverage=3.0)
        bh1a = ReportGenerator.align_to_strategy(bh1, eq)
        bh3a = ReportGenerator.align_to_strategy(bh3, eq)
        
        # 保存汇总
        rows = [
            dict(model="strategy", **kpis, **tstats),
            dict(model="bh_1x", **ReportGenerator.calculate_metrics(bh1a),
                 n_trades=np.nan, win_rate=np.nan, avg_net=np.nan, med_net=np.nan),
            dict(model="bh_3x", **ReportGenerator.calculate_metrics(bh3a),
                 n_trades=np.nan, win_rate=np.nan, avg_net=np.nan, med_net=np.nan),
        ]
        summary = pd.DataFrame(rows)
        summary.to_csv(f"reports/step8/summary_{segment_name}.csv", index=False)
        
        # 绘图
        ReportGenerator.plot_equity(
            eq, bh1a, bh3a,
            f"{segment_name.capitalize()} Equity (Strategy vs Buy & Hold)",
            f"figs/{segment_name}_equity.png"
        )
        
        # 打印结果
        print(f"\n   📈 {segment_name.upper()} 结果:")
        print(f"      策略最终净值: {kpis['final_equity']:.4f}")
        print(f"      策略CAGR: {kpis['cagr']:.2%}")
        print(f"      策略Sharpe: {kpis['sharpe']:.3f}")
        print(f"      最大回撤: {kpis['maxdd']:.2%}")
        print(f"      交易次数: {tstats['n_trades']}")
        print(f"      胜率: {tstats['win_rate']:.2%}")
        print(f"\n      Buy&Hold 1x净值: {summary[summary['model']=='bh_1x']['final_equity'].iloc[0]:.4f}")
        print(f"      Buy&Hold 3x净值: {summary[summary['model']=='bh_3x']['final_equity'].iloc[0]:.4f}")


# ========================
# 主流程
# ========================
def main():
    """主函数 - 执行完整流程"""
    
    start_time = time.time()
    
    # 创建目录
    Utils.create_directories()
    
    # ==================== STEP 1: 加载数据 ====================
    Utils.print_step(1, "加载原始数据")
    
    if not os.path.exists(Config.INPUT_CSV):
        print(f"❌ 错误: 找不到输入文件 {Config.INPUT_CSV}")
        sys.exit(1)
    
    df = pd.read_csv(Config.INPUT_CSV)
    df = Utils.normalize_columns(df)
    ts_col = Utils.find_timestamp_column(df)
    
    print(f"✅ 数据加载成功")
    print(f"   文件: {Config.INPUT_CSV}")
    print(f"   行数: {len(df):,}")
    print(f"   列数: {len(df.columns)}")
    print(f"   时间列: {ts_col}")
    
    # 排序
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col).reset_index(drop=True)
    
    print(f"   时间范围: {df[ts_col].min()} 至 {df[ts_col].max()}")
    
    # ==================== STEP 2: 构建标签 ====================
    Utils.print_step(2, "构建标签")
    
    df_labeled = LabelBuilder.build_labels(df, ts_col)
    df_labeled.to_csv("data/labels/btc_4h_with_labels.csv", index=False)
    print(f"✅ 标签已保存: data/labels/btc_4h_with_labels.csv")
    
    # ==================== STEP 3: 特征预处理 ====================
    Utils.print_step(3, "特征预处理（防泄露）")
    
    features = FeaturePreprocessor.select_features(df_labeled, ts_col)
    print(f"📋 选择了 {len(features)} 个特征")
    if len(features) <= 20:
        print(f"   特征列表: {features}")
    else:
        print(f"   前20个特征: {features[:20]} ...")
    
    df_numeric = FeaturePreprocessor.coerce_numeric(df_labeled, features)
    df_prep = FeaturePreprocessor.rolling_standardize(df_numeric, features)
    
    df_prep.to_csv("data/prepared/btc_4h_preprocessed.csv", index=False)
    print(f"✅ 预处理数据已保存: data/prepared/btc_4h_preprocessed.csv")
    
    # ==================== STEP 4: 时间序列划分 ====================
    Utils.print_step(4, "时间序列划分")
    
    folds = TimeSplitter.build_rolling_folds(df_prep, ts_col)
    folds.to_csv("data/splits/rolling_folds.csv", index=False)
    print(f"✅ 折叠定义已保存: data/splits/rolling_folds.csv")
    
    holdout = TimeSplitter.build_holdout(df_prep, ts_col)
    holdout.to_csv("data/splits/holdout.csv", index=False)
    print(f"✅ Holdout定义已保存: data/splits/holdout.csv")
    
    # ==================== STEP 5: 模型训练 ====================
    Utils.print_step(5, "模型训练和概率校准")
    
    val_outputs = ModelTrainer.train_all_folds(df_prep, folds, features)
    
    if not val_outputs:
        print("❌ 错误: 没有生成验证集预测")
        sys.exit(1)
    
    # ==================== STEP 6: 阈值搜索 ====================
    Utils.print_step(6, "阈值搜索")
    
    val_all = pd.read_csv("data/predictions/val_all_folds.csv")
    val_all.columns = [c.strip().lower() for c in val_all.columns]
    val_all["time"] = pd.to_datetime(val_all["time"], errors="coerce")
    
    # 合并r_future_7d
    base_for_merge = df_prep[[ts_col, "r_future_7d"]].rename(columns={ts_col: "time"})
    val_all = val_all.merge(base_for_merge, on="time", how="left")
    val_all = val_all.dropna(subset=["r_future_7d"]).sort_values(["fold_id", "time"]).reset_index(drop=True)
    
    grid = ThresholdSearcher.grid_search(val_all)
    best = ThresholdSearcher.select_best(grid)
    
    # ==================== STEP 7: Walk-Forward测试 ====================
    Utils.print_step(7, "Walk-Forward样本外测试")
    
    holdout_info = pd.read_csv("data/splits/holdout.csv")
    
    # Test集（Holdout之前的最后部分）
    test_start_idx = folds[folds["segment"] == "val"].iloc[-1]["end_idx"] + 1
    test_end_idx = holdout_info.iloc[0]["start_idx"] - 1
    
    if Config.FAST_MODE:
        print("\n⚡ 使用快速模式（不重训练每个bar）")
        
        if test_start_idx < test_end_idx:
            test_holdout_df = pd.DataFrame({
                "start_idx": [test_start_idx],
                "end_idx": [test_end_idx]
            })
            
            test_trades, test_equity = WalkForwardTester.walk_forward_test_fast(
                df_prep, features, folds, test_holdout_df, "test"
            )
        else:
            print("   ⚠️  没有足够数据构建Test集")
        
        # Holdout集
        holdout_test_df = pd.DataFrame({
            "start_idx": [holdout_info.iloc[0]["start_idx"]],
            "end_idx": [holdout_info.iloc[0]["end_idx"]]
        })
        
        holdout_trades, holdout_equity = WalkForwardTester.walk_forward_test_fast(
            df_prep, features, folds, holdout_test_df, "holdout"
        )
    else:
        print("\n🐌 使用完整Walk-Forward模式（每个bar重训练，很慢！）")
        
        if test_start_idx < test_end_idx:
            test_holdout_df = pd.DataFrame({
                "start_idx": [test_start_idx],
                "end_idx": [test_end_idx]
            })
            
            test_trades, test_equity = WalkForwardTester.walk_forward_test(
                df_prep, features, test_holdout_df, "test"
            )
        else:
            print("   ⚠️  没有足够数据构建Test集")
        
        # Holdout集
        holdout_test_df = pd.DataFrame({
            "start_idx": [holdout_info.iloc[0]["start_idx"]],
            "end_idx": [holdout_info.iloc[0]["end_idx"]]
        })
        
        holdout_trades, holdout_equity = WalkForwardTester.walk_forward_test(
            df_prep, features, holdout_test_df, "holdout"
        )
    
    # ==================== STEP 8: 结果报告 ====================
    Utils.print_step(8, "生成最终报告")
    
    ReportGenerator.generate_report(df_prep, "test")
    ReportGenerator.generate_report(df_prep, "holdout")
    
    # ==================== 完成 ====================
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"  ✅ 完整流程执行完毕")
    print(f"  ⏱️  总耗时: {elapsed/60:.1f}分钟")
    print(f"{'='*80}\n")
    
    print("📁 输出文件:")
    print("   - data/labels/btc_4h_with_labels.csv")
    print("   - data/prepared/btc_4h_preprocessed.csv")
    print("   - data/splits/rolling_folds.csv")
    print("   - data/splits/holdout.csv")
    print("   - data/predictions/val_all_folds.csv")
    print("   - data/thresholds/best_thresholds_per_model.csv")
    print("   - data/signals/test_trades.csv")
    print("   - data/signals/holdout_trades.csv")
    print("   - data/equity/test_equity.csv")
    print("   - data/equity/holdout_equity.csv")
    print("   - reports/step8/summary_test.csv")
    print("   - reports/step8/summary_holdout.csv")
    print("   - figs/test_equity.png")
    print("   - figs/holdout_equity.png")
    
    print(f"\n🎉 完成！请查看 reports/step8/ 和 figs/ 目录查看结果")
if __name__ == "__main__":
    main()