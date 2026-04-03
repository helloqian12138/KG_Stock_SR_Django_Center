import os
import sys
import threading
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.models.dtml import DTMLModel
from src.models.lstm import LSTMModel
from src.models.master import MASTERModel
from src.models.ragformer import RAGFormerModel
from src.models.transformer import TransformerModel

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_END_PERIOD = pd.Period('2025-06', freq='M')
RISK_THRESHOLD = 0.85

SCENARIO_LABELS = {
    'baseline': '基准情景（关税维持现状）',
    'moderate': '中等冲击（+25%关税）',
    'severe': '严重冲击（+60%关税）',
    'extreme': '极端冲击（+科技封锁）'
}

SCENARIO_PRESETS = {
    'baseline': {'tariff': 10, 'cny_depr': 3, 'outflow': 500},
    'moderate': {'tariff': 25, 'cny_depr': 8, 'outflow': 1200},
    'severe': {'tariff': 60, 'cny_depr': 10, 'outflow': 1500},
    'extreme': {'tariff': 100, 'cny_depr': 25, 'outflow': 2800}
}

MACRO_SCENARIO_LABELS = {
    'baseline': '基准情景（GDP/PPI/CPI/关税维持趋势）',
    'growth_slowdown': '增长放缓（GDP下行+通缩）',
    'reflation': '温和再通胀（需求改善+价格修复）',
    'stagflation': '滞胀冲击（增长走弱+成本上升）'
}

MACRO_SCENARIO_PRESETS = {
    'baseline': {'gdp_shift': 0.0, 'ppi_shift': 0.0, 'cpi_shift': 0.0, 'tariff_shift': 0.0},
    'growth_slowdown': {'gdp_shift': -1.2, 'ppi_shift': -1.6, 'cpi_shift': -0.5, 'tariff_shift': 3.0},
    'reflation': {'gdp_shift': 0.6, 'ppi_shift': 1.2, 'cpi_shift': 0.4, 'tariff_shift': -2.0},
    'stagflation': {'gdp_shift': -1.0, 'ppi_shift': 2.0, 'cpi_shift': 0.9, 'tariff_shift': 5.0}
}

PPI_SCENARIO_PRESETS = {
    'baseline': {'ppi_shift': 0.0, 'cpi_shift': 0.0, 'rate_delta': 0.0, 'fx_depr': 0.0, 'iov_shift': 0.0},
    'reflation': {'ppi_shift': 1.5, 'cpi_shift': 0.3, 'rate_delta': -0.2, 'fx_depr': 3.0, 'iov_shift': 0.5},
    'deflation': {'ppi_shift': -2.0, 'cpi_shift': -0.4, 'rate_delta': -0.5, 'fx_depr': 1.0, 'iov_shift': -0.6},
    'stagflation': {'ppi_shift': 2.5, 'cpi_shift': 0.8, 'rate_delta': 0.2, 'fx_depr': 8.0, 'iov_shift': -0.5},
}

SCENARIO_SENSITIVE_COLUMNS = ['Tariff_Rate', 'USD_CNY', 'FX_Reserves_Change', 'Trade_War_Dummy']

NON_MODEL_KEYS = {
    'lr', 'learning_rate', 'weight_decay', 'batch_size', 'epochs', 'patience',
    'optimizer', 'optimizer_type', 'scheduler', 'scheduler_type',
    'seq_len', 'seq_length', 'max_epochs', 'min_lr', 'clip_grad',
    'seed', 'device'
}

model_choices = ['ragformer', 'lstm', 'dtml', 'transformer', 'master']
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

TAXWAR_DF_CACHE = None
TAXWAR_START_PERIOD = None
TAXWAR_LOCK = threading.Lock()
FSI_HISTORY_CACHE = None
PREDICTION_BUNDLES = {}
PREDICTION_LOCK = threading.Lock()


def data_path(*parts):
    return str(BASE_DIR.joinpath(*parts))


def saved_models_path():
    return data_path('saved_models')


def log(message: str):
    print(message)


def _joblib_load_with_numpy_compat(path: str):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        # Compatibility for artifacts created by newer NumPy that reference numpy._core.
        if exc.name != 'numpy._core':
            raise
        import numpy.core as np_core  # local import to avoid unnecessary module aliasing

        sys.modules.setdefault('numpy._core', np_core)
        return joblib.load(path)


def _numpy_major_version() -> int:
    try:
        return int(str(np.__version__).split('.', 1)[0])
    except Exception:
        return 0


def _pickle_refs_numpy_core(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(512 * 1024)
        return b'numpy._core' in head
    except Exception:
        return False


def _build_runtime_scalers(feature_columns=None, input_size=None):
    df = get_taxwar_dataframe()
    feature_df = df.drop(['Date', 'FSI_Raw'], axis=1)

    if feature_columns:
        cols = [col for col in feature_columns if col in feature_df.columns]
        if cols:
            feature_df = feature_df[cols]
    elif input_size and feature_df.shape[1] > int(input_size):
        feature_df = feature_df.iloc[:, :int(input_size)]

    x_values = np.nan_to_num(feature_df.values.astype(np.float32), nan=0.0)
    y_values = np.nan_to_num(df['FSI_Raw'].values.astype(np.float32).reshape(-1, 1), nan=0.0)

    scaler_X = StandardScaler().fit(x_values)
    scaler_y = StandardScaler().fit(y_values)
    return scaler_X, scaler_y


def get_taxwar_dataframe() -> pd.DataFrame:
    global TAXWAR_DF_CACHE, TAXWAR_START_PERIOD
    if TAXWAR_DF_CACHE is None:
        with TAXWAR_LOCK:
            if TAXWAR_DF_CACHE is None:
                df_index = pd.read_csv(data_path('data', 'taxwar', 'index.csv'))
                df_index['Date'] = pd.to_datetime(df_index['Date'], errors='coerce').dt.to_period('M')

                df_fsi = pd.read_csv(data_path('data', 'fsi', 'fsi18_optimized.csv'), parse_dates=['Date'])
                df_fsi['Date'] = df_fsi['Date'].dt.to_period('M')

                df = pd.merge(df_index, df_fsi[['Date', 'FSI_Raw']], on='Date', how='inner')
                df = df.sort_values('Date').reset_index(drop=True)

                TAXWAR_DF_CACHE = df
                TAXWAR_START_PERIOD = df['Date'].min()
    return TAXWAR_DF_CACHE.copy()


def get_fsi_history_entry():
    global FSI_HISTORY_CACHE
    if FSI_HISTORY_CACHE is None:
        df = pd.read_csv(data_path('data', 'fsi', 'fsi18_optimized.csv'), parse_dates=['Date'])
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        df['News'] = df['News'].fillna('无事件')
        FSI_HISTORY_CACHE = {
            'dates': df['Month'].tolist(),
            'fsi': df['FSI_Raw'].round(4).tolist(),
            'fsi_smooth': df['FSI'].round(4).tolist(),
            'news': df['News'].tolist()
        }
    return FSI_HISTORY_CACHE


def _latest_checkpoint_path(model_name: str) -> str:
    prefix = f'{model_name}_attempt_'
    candidates = []
    for fname in os.listdir(saved_models_path()):
        if fname.startswith(prefix) and fname.endswith('.pth'):
            try:
                attempt = int(fname[len(prefix):-4])
                candidates.append((attempt, fname))
            except ValueError:
                continue
    if not candidates:
        return ''
    candidates.sort(key=lambda x: x[0])
    return data_path('saved_models', candidates[-1][1])


def load_prediction_bundle(model_name: str = 'ragformer'):
    global PREDICTION_BUNDLES
    model_name = model_name.lower()
    if model_name not in PREDICTION_BUNDLES:
        with PREDICTION_LOCK:
            if model_name not in PREDICTION_BUNDLES:
                if not os.path.exists(saved_models_path()):
                    raise FileNotFoundError('saved_models 目录不存在，无法执行预测')

                checkpoint_path = _latest_checkpoint_path(model_name)
                if not checkpoint_path:
                    if model_name == 'ragformer':
                        fallback = data_path('saved_models', 'ragformer_attempt_12.pth')
                        checkpoint_path = fallback if os.path.exists(fallback) else ''
                    if not checkpoint_path:
                        raise FileNotFoundError(f'saved_models 中未找到 {model_name} 模型')

                checkpoint = torch.load(checkpoint_path, map_location=device)
                model_params = checkpoint.get('model_params', {}).copy()
                if not model_params:
                    raw_params = checkpoint.get('params', {}) or {}
                    model_params = {k: v for k, v in raw_params.items() if k not in NON_MODEL_KEYS}
                input_size = checkpoint.get('input_size') or model_params.get('input_size')
                if input_size:
                    model_params['input_size'] = input_size
                feature_columns = checkpoint.get('feature_columns')

                model_cls_map = {
                    'ragformer': RAGFormerModel,
                    'lstm': LSTMModel,
                    'dtml': DTMLModel,
                    'transformer': TransformerModel,
                    'master': MASTERModel,
                }
                model_cls = model_cls_map.get(model_name)
                if model_cls is None:
                    raise ValueError(f'未知模型类型: {model_name}')

                if model_name == 'ragformer':
                    model_params.setdefault('dim_feedforward', 64)
                    model_params.setdefault('k_neighbors', 5)
                    d_model = model_params.get('d_model', 16)
                    nhead = model_params.get('nhead', 2)
                    if d_model % nhead != 0:
                        model_params['d_model'] = ((d_model // nhead) + 1) * nhead
                elif model_name == 'dtml':
                    model_params.setdefault('n_time', int(checkpoint.get('seq_len', 12)))
                elif model_name == 'master':
                    if input_size:
                        model_params.setdefault('gate_input_start_index', 10)
                        model_params.setdefault('gate_input_end_index', int(input_size))

                model = model_cls(**model_params).to(device)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.eval()

                scaler_X_path = checkpoint.get('scaler_X_path')
                scaler_y_path = checkpoint.get('scaler_y_path')
                if not scaler_X_path or not scaler_y_path:
                    raise FileNotFoundError(f'{checkpoint_path} 缺少 scaler 路径')

                scaler_X_full = scaler_X_path if os.path.isabs(scaler_X_path) else data_path(scaler_X_path)
                scaler_y_full = scaler_y_path if os.path.isabs(scaler_y_path) else data_path(scaler_y_path)
                if not os.path.exists(scaler_X_full):
                    scaler_X_full = data_path('scalers', os.path.basename(scaler_X_path))
                if not os.path.exists(scaler_y_full):
                    scaler_y_full = data_path('scalers', os.path.basename(scaler_y_path))

                force_runtime_scaler = (
                    _numpy_major_version() < 2 and (
                        _pickle_refs_numpy_core(scaler_X_full) or _pickle_refs_numpy_core(scaler_y_full)
                    )
                )
                if force_runtime_scaler:
                    log('检测到 scaler 由 numpy>=2 序列化，当前 numpy<2，使用运行时拟合兜底')
                    scaler_X, scaler_y = _build_runtime_scalers(
                        feature_columns=feature_columns,
                        input_size=input_size,
                    )
                else:
                    try:
                        scaler_X = _joblib_load_with_numpy_compat(scaler_X_full)
                        scaler_y = _joblib_load_with_numpy_compat(scaler_y_full)
                    except Exception as exc:
                        log(f'加载 scaler 失败，使用运行时拟合兜底: {type(exc).__name__}')
                        scaler_X, scaler_y = _build_runtime_scalers(
                            feature_columns=feature_columns,
                            input_size=input_size,
                        )

                seq_len = int(checkpoint.get('seq_len', 12))
                PREDICTION_BUNDLES[model_name] = {
                    'model': model,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y,
                    'seq_len': seq_len,
                    'feature_columns': feature_columns,
                    'input_size': input_size,
                    'model_input_size': getattr(model, 'input_size', input_size),
                }
    return PREDICTION_BUNDLES[model_name]


def parse_month(value: str, field: str) -> pd.Period:
    try:
        return pd.Period(value, freq='M')
    except Exception as exc:
        raise ValueError(f'{field} 不是有效的年月: {value}') from exc


def get_available_saved_models():
    available = []
    for model_name in model_choices:
        if _latest_checkpoint_path(model_name):
            available.append(model_name)
    return available


def build_actual_history_map(df: pd.DataFrame, hist_start: pd.Period):
    hist_mask = (df['Date'] >= hist_start) & (df['Date'] <= HISTORY_END_PERIOD)
    hist_df = df.loc[hist_mask, ['Date', 'FSI_Raw']].copy()
    hist_df['FSI_Raw'] = hist_df['FSI_Raw'].ffill().bfill()
    actual_map = {}
    for _, row in hist_df.iterrows():
        actual_map[row['Date']] = float(row['FSI_Raw'])
    return actual_map


def calc_alignment_metrics(pred_map: dict, truth_map: dict):
    aligned_keys = sorted(set(pred_map.keys()) & set(truth_map.keys()))
    if not aligned_keys:
        return {'mse': None, 'mae': None, 'count': 0}
    y_pred = np.array([pred_map[k] for k in aligned_keys], dtype=np.float32)
    y_true = np.array([truth_map[k] for k in aligned_keys], dtype=np.float32)
    return {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'count': int(len(aligned_keys)),
    }


def _future_pressure_adjustment(df: pd.DataFrame, predictions: dict) -> dict:
    if not predictions:
        return {}

    hist_df = df.loc[df['Date'] <= HISTORY_END_PERIOD].copy()
    if hist_df.empty:
        return predictions

    anchor = hist_df.iloc[-1]
    baseline_fsi = float(anchor.get('FSI_Raw', 0.0))
    adjusted = {}
    prev_adjusted = baseline_fsi

    for period in sorted(predictions):
        row_match = df.loc[df['Date'] == period]
        if row_match.empty:
            adjusted[period] = float(np.clip(predictions[period], 0.0, 2.0))
            continue
        row = row_match.iloc[-1]

        gdp_delta = float(row.get('GDP_Growth', anchor.get('GDP_Growth', 0.0))) - float(anchor.get('GDP_Growth', 0.0))
        ppi_delta = float(row.get('PPI', anchor.get('PPI', 0.0))) - float(anchor.get('PPI', 0.0))
        cpi_delta = float(row.get('CPI', anchor.get('CPI', 0.0))) - float(anchor.get('CPI', 0.0))
        tariff_delta = float(row.get('Tariff_Rate', anchor.get('Tariff_Rate', 0.0))) - float(anchor.get('Tariff_Rate', 0.0))
        fx_delta = float(row.get('USD_CNY', anchor.get('USD_CNY', 0.0))) - float(anchor.get('USD_CNY', 0.0))
        spread_delta = float(row.get('Bond_Spread', anchor.get('Bond_Spread', 0.0))) - float(anchor.get('Bond_Spread', 0.0))
        vol_delta = float(row.get('Stock_Volatility', anchor.get('Stock_Volatility', 0.0))) - float(anchor.get('Stock_Volatility', 0.0))
        reserves_delta = float(row.get('FX_Reserves_Change', anchor.get('FX_Reserves_Change', 0.0))) - float(anchor.get('FX_Reserves_Change', 0.0))

        pressure_score = 0.0
        pressure_score += max(-gdp_delta, 0.0) * 0.12
        pressure_score -= max(gdp_delta, 0.0) * 0.06
        pressure_score += max(-ppi_delta, 0.0) * 0.09
        pressure_score += max(ppi_delta - 1.5, 0.0) * 0.05
        pressure_score += max(-cpi_delta, 0.0) * 0.08
        pressure_score += max(cpi_delta - 1.5, 0.0) * 0.06
        pressure_score += max(tariff_delta, 0.0) * 0.018
        pressure_score -= max(-tariff_delta, 0.0) * 0.010
        pressure_score += max(fx_delta, 0.0) * 0.25
        pressure_score += max(spread_delta, 0.0) * 0.015
        pressure_score += max(vol_delta, 0.0) * 0.018
        pressure_score += max(-reserves_delta, 0.0) * 0.10

        model_value = float(predictions[period])
        uplift = max(pressure_score, 0.0) * 0.22
        relief = max(-pressure_score, 0.0) * 0.10

        floor_value = baseline_fsi + max(pressure_score, 0.0) * 0.18
        if pressure_score > 0:
            floor_value = max(floor_value, prev_adjusted - 0.01)
        else:
            floor_value = min(floor_value, prev_adjusted + 0.015)

        adjusted_value = model_value + uplift - relief
        adjusted_value = max(adjusted_value, floor_value) if pressure_score > 0 else adjusted_value
        adjusted_value = float(np.clip(adjusted_value, 0.0, 2.0))
        adjusted[period] = adjusted_value
        prev_adjusted = adjusted_value

    return adjusted


def extend_with_scenario(df: pd.DataFrame, target_period: pd.Period, scenario: dict) -> pd.DataFrame:
    if target_period <= df['Date'].max():
        return df.copy()

    last_row = df.iloc[-1].copy()
    base_trend = int(last_row.get('Time_Trend', len(df)))
    base_fx = float(last_row.get('USD_CNY', 7.0))

    periods = pd.period_range(df['Date'].max() + 1, target_period, freq='M')
    future_rows = []
    for idx, period in enumerate(periods, start=1):
        new_row = last_row.copy()
        new_row['Date'] = period
        new_row['FSI_Raw'] = np.nan
        new_row['Time_Trend'] = base_trend + idx
        new_row['Tariff_Rate'] = float(scenario['tariff'])
        shock = float(scenario['cny_depr']) / 100.0
        new_row['USD_CNY'] = base_fx * (1.0 + shock)
        new_row['FX_Reserves_Change'] = -abs(float(scenario['outflow'])) / 1000.0
        new_row['Trade_War_Dummy'] = 1
        future_rows.append(new_row)

    if future_rows:
        future_df = pd.DataFrame(future_rows)
        future_df = future_df[df.columns]
        df = pd.concat([df, future_df], ignore_index=True)
    return df


def _signed_piecewise(value: float, pos_coef: float, neg_coef: float, pos_knee: float = 0.0, neg_knee: float = 0.0,
                      pos_tail: Optional[float] = None, neg_tail: Optional[float] = None) -> float:
    value = float(value)
    pos_tail = pos_coef if pos_tail is None else pos_tail
    neg_tail = neg_coef if neg_tail is None else neg_tail

    if value >= 0:
        head = min(value, pos_knee) if pos_knee > 0 else 0.0
        tail = max(value - pos_knee, 0.0) if pos_knee > 0 else value
        return head * pos_coef + tail * pos_tail

    abs_value = abs(value)
    head = min(abs_value, neg_knee) if neg_knee > 0 else 0.0
    tail = max(abs_value - neg_knee, 0.0) if neg_knee > 0 else abs_value
    return -(head * neg_coef + tail * neg_tail)


def _macro_pressure_components(gdp_shift: float, ppi_shift: float, cpi_shift: float, tariff_shift: float) -> dict:
    gdp_support = _signed_piecewise(gdp_shift, pos_coef=0.10, neg_coef=0.13, pos_knee=3.0, neg_knee=3.0,
                                    pos_tail=0.05, neg_tail=0.18)

    ppi_impact = 0.0
    if ppi_shift >= 0:
        if gdp_shift >= 0:
            # 温和 PPI 回升在需求改善阶段通常更接近“价格修复”，不宜机械视为风险。
            ppi_impact -= min(ppi_shift, 3.0) * 0.02
            ppi_impact += max(ppi_shift - 3.0, 0.0) * 0.05
        else:
            ppi_impact += min(ppi_shift, 3.0) * 0.03
            ppi_impact += max(ppi_shift - 3.0, 0.0) * 0.07
    else:
        ppi_impact += min(abs(ppi_shift), 3.0) * 0.06
        ppi_impact += max(abs(ppi_shift) - 3.0, 0.0) * 0.10

    if cpi_shift >= 0:
        cpi_impact = min(cpi_shift, 2.0) * 0.03 + max(cpi_shift - 2.0, 0.0) * 0.08
    else:
        cpi_impact = min(abs(cpi_shift), 2.0) * 0.05 + max(abs(cpi_shift) - 2.0, 0.0) * 0.10

    if tariff_shift >= 0:
        tariff_impact = min(tariff_shift, 10.0) * 0.02 + max(tariff_shift - 10.0, 0.0) * 0.04
    else:
        tariff_impact = -(min(abs(tariff_shift), 10.0) * 0.015 + max(abs(tariff_shift) - 10.0, 0.0) * 0.01)

    return {
        'gdp_support': gdp_support,
        'ppi_impact': ppi_impact,
        'cpi_impact': cpi_impact,
        'tariff_impact': tariff_impact,
        'net_pressure': ppi_impact + cpi_impact + tariff_impact - gdp_support,
    }


def extend_with_macro_scenario(df: pd.DataFrame, target_period: pd.Period, scenario: dict) -> pd.DataFrame:
    if target_period <= df['Date'].max():
        return df.copy()

    last_row = df.iloc[-1].copy()
    base_trend = int(last_row.get('Time_Trend', len(df)))
    base_gdp = float(last_row.get('GDP_Growth', 0.0))
    base_ppi = float(last_row.get('PPI', 0.0))
    base_cpi = float(last_row.get('CPI', 0.0))
    base_tariff = float(last_row.get('Tariff_Rate', 0.0))
    base_fx = float(last_row.get('USD_CNY', 7.0))
    base_trade = float(last_row.get('Trade_Volume', 0.0))
    base_balance = float(last_row.get('Trade_Balance_China', 0.0))
    base_export = float(last_row.get('Export_Growth_US', 0.0))
    base_import = float(last_row.get('Import_Growth_US', 0.0))
    base_liquidity = float(last_row.get('Liquidity_SHIBOR', 0.0))

    gdp_shift = float(scenario.get('gdp_shift', 0.0))
    ppi_shift = float(scenario.get('ppi_shift', 0.0))
    cpi_shift = float(scenario.get('cpi_shift', 0.0))
    tariff_shift = float(scenario.get('tariff_shift', 0.0))

    components = _macro_pressure_components(gdp_shift, ppi_shift, cpi_shift, tariff_shift)

    periods = pd.period_range(df['Date'].max() + 1, target_period, freq='M')
    future_rows = []
    total_periods = max(len(periods), 1)
    for idx, period in enumerate(periods, start=1):
        new_row = last_row.copy()
        new_row['Date'] = period
        new_row['FSI_Raw'] = np.nan
        new_row['Time_Trend'] = base_trend + idx
        fade = 1.0 - 0.15 * ((idx - 1) / max(total_periods - 1, 1))

        eff_gdp_shift = gdp_shift * fade
        eff_ppi_shift = ppi_shift * fade
        eff_cpi_shift = cpi_shift * fade
        eff_tariff_shift = tariff_shift * fade

        new_row['GDP_Growth'] = np.clip(base_gdp + eff_gdp_shift, -10.0, 15.0)
        new_row['PPI'] = np.clip(base_ppi + eff_ppi_shift, -15.0, 15.0)
        new_row['CPI'] = np.clip(base_cpi + eff_cpi_shift, -5.0, 10.0)
        new_row['Tariff_Rate'] = np.clip(base_tariff + eff_tariff_shift, 0.0, 125.0)

        net_pressure = components['net_pressure'] * fade
        trade_support = max(components['gdp_support'], 0.0) * 0.5 + max(-components['tariff_impact'], 0.0) * 0.8
        trade_drag = max(components['ppi_impact'] + components['cpi_impact'] + components['tariff_impact'], 0.0) * 0.6
        fx_pressure = max(net_pressure, 0.0) * 0.018 - max(components['gdp_support'], 0.0) * 0.008

        new_row['Bond_Spread'] = float(last_row.get('Bond_Spread', 0.0)) + net_pressure * 3.8
        new_row['Stock_Volatility'] = float(last_row.get('Stock_Volatility', 0.0)) + net_pressure * 2.4
        new_row['LPR_SHIBOR'] = float(last_row.get('LPR_SHIBOR', 0.0)) + (
            max(eff_cpi_shift, 0.0) * 0.05 - max(-eff_cpi_shift, 0.0) * 0.04 - max(eff_gdp_shift, 0.0) * 0.02
        )
        new_row['Liquidity_SHIBOR'] = max(0.05, base_liquidity - max(net_pressure, 0.0) * 0.18 + trade_support * 0.03)
        new_row['USD_CNY'] = max(5.5, base_fx * (1.0 + fx_pressure))
        new_row['Trade_Volume'] = max(0.0, base_trade * (1.0 + (trade_support - trade_drag) * 0.02))
        new_row['Trade_Balance_China'] = max(0.0, base_balance * (1.0 + (trade_support - trade_drag) * 0.015))
        new_row['Export_Growth_US'] = np.clip(base_export + eff_gdp_shift * 0.6 - eff_tariff_shift * 0.35 - max(eff_ppi_shift, 0.0) * 0.2, -30.0, 25.0)
        new_row['Import_Growth_US'] = np.clip(base_import + eff_gdp_shift * 0.5 - eff_tariff_shift * 0.25 - max(eff_cpi_shift, 0.0) * 0.15, -30.0, 25.0)
        new_row['FX_Reserves_Change'] = float(last_row.get('FX_Reserves_Change', 0.0)) - max(net_pressure, 0.0) * 0.08 + max(components['gdp_support'], 0.0) * 0.03
        new_row['Trade_War_Dummy'] = 1 if new_row['Tariff_Rate'] >= 5.0 else 0
        future_rows.append(new_row)

    if future_rows:
        future_df = pd.DataFrame(future_rows)
        future_df = future_df[df.columns]
        df = pd.concat([df, future_df], ignore_index=True)
    return df


def extend_with_ppi_scenario(df: pd.DataFrame, target_period: pd.Period, scenario: dict) -> pd.DataFrame:
    macro_like = {
        'gdp_shift': 0.0,
        'ppi_shift': float(scenario.get('ppi_shift', 0.0)),
        'cpi_shift': float(scenario.get('cpi_shift', 0.0)),
    }
    out = extend_with_macro_scenario(df, target_period, macro_like)
    mask = out['Date'] > HISTORY_END_PERIOD

    rate_delta = float(scenario.get('rate_delta', 0.0))
    fx_depr = float(scenario.get('fx_depr', 0.0)) / 100.0
    iov_shift = float(scenario.get('iov_shift', 0.0)) / 100.0

    if 'LPR_SHIBOR' in out.columns:
        out.loc[mask, 'LPR_SHIBOR'] = out.loc[mask, 'LPR_SHIBOR'] + rate_delta
    if 'USD_CNY' in out.columns:
        out.loc[mask, 'USD_CNY'] = out.loc[mask, 'USD_CNY'] * (1.0 + fx_depr)
    if 'Trade_Volume' in out.columns:
        out.loc[mask, 'Trade_Volume'] = out.loc[mask, 'Trade_Volume'] * (1.0 + iov_shift)
    return out


def compute_predictions(df: pd.DataFrame, hist_start: pd.Period, short_end: pd.Period, long_end: pd.Period, model_name: str = 'ragformer'):
    bundle = load_prediction_bundle(model_name=model_name)
    feature_df = df.drop(['Date', 'FSI_Raw'], axis=1)
    feature_columns = bundle.get('feature_columns')
    scaler_X = bundle.get('scaler_X')
    expected_n = int(scaler_X.n_features_in_) if hasattr(scaler_X, 'n_features_in_') else None
    model_input_size = bundle.get('model_input_size') or bundle.get('input_size')

    if feature_columns:
        missing = [col for col in feature_columns if col not in feature_df.columns]
        if missing:
            raise ValueError(f'预测特征列缺失: {missing}')
        feature_df = feature_df[feature_columns]
    else:
        target_n = int(model_input_size) if model_input_size else expected_n
        if target_n and feature_df.shape[1] != target_n:
            if feature_df.shape[1] > target_n:
                feature_df = feature_df.iloc[:, :target_n]
            else:
                raise ValueError(f'预测特征数量不足: {feature_df.shape[1]} < {target_n}')

    if expected_n and model_input_size and int(expected_n) != int(model_input_size):
        hist_mask = df['Date'] <= HISTORY_END_PERIOD
        fit_df = feature_df.loc[hist_mask] if hist_mask.any() else feature_df
        scaler_X = StandardScaler().fit(np.nan_to_num(fit_df.values.astype(np.float32), nan=0.0))

    features = np.nan_to_num(feature_df.values.astype(np.float32), nan=0.0)
    scaled = scaler_X.transform(features).astype(np.float32)
    tensor_features = torch.from_numpy(scaled)

    predictions = {}
    seq_len = bundle['seq_len']
    model = bundle['model']
    scaler_y = bundle['scaler_y']

    if hasattr(model, 'set_historical_data'):
        history_mask = (df['Date'] <= HISTORY_END_PERIOD).values
        history_idx = np.where(history_mask)[0]
        history_index_tensor = torch.tensor(history_idx, dtype=torch.long)
        history_pool = tensor_features[history_index_tensor] if history_index_tensor.numel() else tensor_features[:seq_len]
        if history_pool.shape[0] > 0:
            if hasattr(model, 'k_neighbors'):
                model.k_neighbors = max(1, min(int(model.k_neighbors), history_pool.shape[0]))
            model.set_historical_data(history_pool.to(device))

    with torch.no_grad():
        for idx, current_period in enumerate(df['Date']):
            if current_period < hist_start or current_period > long_end:
                continue
            if idx < seq_len:
                continue
            window = tensor_features[idx - seq_len:idx].unsqueeze(0).to(device)
            pred_scaled = model(window).cpu().numpy().flatten()[0]
            pred_value = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            predictions[current_period] = float(np.clip(pred_value, 0.0, 2.0))

    future_periods = {p: v for p, v in predictions.items() if p > HISTORY_END_PERIOD}
    if future_periods:
        adjusted_future = _future_pressure_adjustment(df, future_periods)
        predictions.update(adjusted_future)

    hist_preds = {p: v for p, v in predictions.items() if hist_start <= p <= HISTORY_END_PERIOD}
    short_preds = {p: v for p, v in predictions.items() if HISTORY_END_PERIOD < p <= short_end}
    long_preds = {p: v for p, v in predictions.items() if p > short_end}

    def summarize(section: dict):
        if not section:
            return None, None
        peak_period = max(section, key=section.get)
        return round(section[peak_period], 3), str(peak_period)

    short_peak, short_peak_date = summarize(short_preds)
    long_peak, long_peak_date = summarize(long_preds)

    risk_months = [value for period, value in predictions.items() if period.year == 2026]
    risk_prob = 0
    if risk_months:
        risky = sum(1 for value in risk_months if value >= RISK_THRESHOLD)
        risk_prob = int(round(risky / len(risk_months) * 100))

    if risk_prob >= 70:
        risk_level = '高风险'
    elif risk_prob >= 40:
        risk_level = '中风险'
    else:
        risk_level = '低风险'

    return {
        'hist_preds': hist_preds,
        'short_preds': short_preds,
        'long_preds': long_preds,
        'short_peak': short_peak,
        'short_peak_date': short_peak_date,
        'long_peak': long_peak,
        'long_peak_date': long_peak_date,
        'risk_prob': risk_prob,
        'risk_level': risk_level,
    }
