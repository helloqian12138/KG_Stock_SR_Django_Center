import json
import traceback
from typing import Optional

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .services import (
    HISTORY_END_PERIOD,
    MACRO_SCENARIO_LABELS,
    MACRO_SCENARIO_PRESETS,
    PPI_SCENARIO_PRESETS,
    SCENARIO_LABELS,
    SCENARIO_PRESETS,
    build_actual_history_map,
    calc_alignment_metrics,
    compute_predictions,
    extend_with_macro_scenario,
    extend_with_ppi_scenario,
    extend_with_scenario,
    get_available_saved_models,
    get_fsi_history_entry,
    get_taxwar_dataframe,
    parse_month,
)


def _json_body(request):
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode('utf-8'))
    except Exception:
        return {}


def _matches_preset(scenario: dict, preset_values: dict, tolerance: float = 1e-9) -> bool:
    for key, preset_value in preset_values.items():
        if abs(float(scenario.get(key, 0.0)) - float(preset_value)) > tolerance:
            return False
    return True


def _format_signed_pct(value: float) -> str:
    return f'{value:+.1f}%'


def _macro_scenario_label(preset: Optional[str], scenario: dict) -> str:
    if preset in MACRO_SCENARIO_PRESETS and _matches_preset(scenario, MACRO_SCENARIO_PRESETS[preset]):
        return MACRO_SCENARIO_LABELS[preset]
    return (
        '自定义联合情景（'
        f"GDP{_format_signed_pct(float(scenario.get('gdp_shift', 0.0)))}, "
        f"CPI{_format_signed_pct(float(scenario.get('cpi_shift', 0.0)))}, "
        f"PPI{_format_signed_pct(float(scenario.get('ppi_shift', 0.0)))}, "
        f"关税{_format_signed_pct(float(scenario.get('tariff_shift', 0.0)))}）"
    )


def taxwar_page(request):
    return render(request, 'taxwar.html', {'active_tab': 'taxwar'})


def taxwar_compare_page(request):
    return render(request, 'taxwar_model_compare.html')


def macroshock_page(request):
    return render(request, 'taxwar.html', {'active_tab': 'macroshock'})


def ppi_page(request):
    return render(request, 'ppi.html')


def api_fsi_history(request):
    return JsonResponse(get_fsi_history_entry())


@csrf_exempt
def api_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)

    data = _json_body(request)
    try:
        hist_start = parse_month(data.get('hist_start', '2018-01'), '历史起点')
        short_end = parse_month(data.get('short_end', '2025-12'), '短期终点')
        long_end = parse_month(data.get('long_end', '2026-12'), '长期终点')
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)

    if short_end <= HISTORY_END_PERIOD:
        return JsonResponse({'error': '短期终点需晚于 2025-06'}, status=400)
    if long_end < short_end:
        return JsonResponse({'error': '长期终点需不早于短期终点'}, status=400)

    df = get_taxwar_dataframe()
    min_period = df['Date'].min()
    if hist_start < min_period:
        hist_start = min_period
    if hist_start >= HISTORY_END_PERIOD:
        return JsonResponse({'error': '历史区间必须早于 2025-06'}, status=400)

    scenario = {
        'tariff': float(data.get('tariff', 0)),
        'cny_depr': float(data.get('cny_depr', 0)),
        'outflow': max(0.0, float(data.get('outflow', 0))),
    }
    preset = data.get('preset')
    scenario_label = SCENARIO_LABELS.get(preset) if preset in SCENARIO_PRESETS and _matches_preset(scenario, SCENARIO_PRESETS[preset]) else '自定义压力情景'

    try:
        df_extended = extend_with_scenario(df, long_end, scenario)
        prediction_payload = compute_predictions(df_extended, hist_start, short_end, long_end, model_name='ragformer')
    except FileNotFoundError as exc:
        return JsonResponse({'error': str(exc)}, status=500)
    except Exception as exc:
        traceback.print_exc()
        return JsonResponse({'error': '模型推理失败', 'detail': str(exc)}, status=500)

    hist_dates_sorted = sorted(prediction_payload['hist_preds'].keys())
    if not hist_dates_sorted:
        return JsonResponse({'error': '无法在该历史区间生成预测'}, status=400)

    short_dates_sorted = sorted(prediction_payload['short_preds'].keys())
    long_dates_sorted = sorted(prediction_payload['long_preds'].keys())

    hist_actual_mask = (df_extended['Date'] >= hist_start) & (df_extended['Date'] <= HISTORY_END_PERIOD)
    hist_actual_df = df_extended.loc[hist_actual_mask, ['Date', 'FSI_Raw']].copy()
    hist_actual_df['FSI_Raw'] = hist_actual_df['FSI_Raw'].ffill().bfill()
    full_history = get_fsi_history_entry()

    response = {
        'dates_hist': [str(p) for p in hist_dates_sorted],
        'fsi_hist': [round(prediction_payload['hist_preds'][p], 3) for p in hist_dates_sorted],
        'dates_hist_actual': hist_actual_df['Date'].astype(str).tolist(),
        'fsi_hist_actual': [round(float(v), 3) for v in hist_actual_df['FSI_Raw'].tolist()],
        'dates_hist_actual_full': full_history.get('dates', []),
        'fsi_hist_actual_full': full_history.get('fsi', []),
        'fsi_hist_smooth_full': full_history.get('fsi_smooth', []),
        'hist_news_full': full_history.get('news', []),
        'dates_short': [str(p) for p in short_dates_sorted],
        'fsi_short': [round(prediction_payload['short_preds'][p], 3) for p in short_dates_sorted],
        'dates_long': [str(p) for p in long_dates_sorted],
        'fsi_long': [round(prediction_payload['long_preds'][p], 3) for p in long_dates_sorted],
        'short_peak': prediction_payload['short_peak'],
        'short_peak_date': prediction_payload['short_peak_date'],
        'long_peak': prediction_payload['long_peak'],
        'long_peak_date': prediction_payload['long_peak_date'],
        'risk_prob': prediction_payload['risk_prob'],
        'risk_level': prediction_payload['risk_level'],
        'scenario_label': scenario_label,
    }
    return JsonResponse(response)


@csrf_exempt
def api_macroshock_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)

    data = _json_body(request)
    try:
        hist_start = parse_month(data.get('hist_start', '2018-01'), '历史起点')
        short_end = parse_month(data.get('short_end', '2025-12'), '短期终点')
        long_end = parse_month(data.get('long_end', '2026-12'), '长期终点')
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)

    if short_end <= HISTORY_END_PERIOD:
        return JsonResponse({'error': '短期终点需晚于 2025-06'}, status=400)
    if long_end < short_end:
        return JsonResponse({'error': '长期终点需不早于短期终点'}, status=400)

    df = get_taxwar_dataframe()
    min_period = df['Date'].min()
    if hist_start < min_period:
        hist_start = min_period
    if hist_start >= HISTORY_END_PERIOD:
        return JsonResponse({'error': '历史区间必须早于 2025-06'}, status=400)

    preset = data.get('preset')
    scenario = MACRO_SCENARIO_PRESETS.get(preset, {
        'gdp_shift': float(data.get('gdp_shift', 0)),
        'ppi_shift': float(data.get('ppi_shift', 0)),
        'cpi_shift': float(data.get('cpi_shift', 0)),
        'tariff_shift': float(data.get('tariff_shift', 0)),
    })
    scenario_label = _macro_scenario_label(preset, scenario)

    try:
        df_extended = extend_with_macro_scenario(df, long_end, scenario)
        prediction_payload = compute_predictions(df_extended, hist_start, short_end, long_end, model_name='ragformer')
    except FileNotFoundError as exc:
        return JsonResponse({'error': str(exc)}, status=500)
    except Exception as exc:
        traceback.print_exc()
        return JsonResponse({'error': '模型推理失败', 'detail': str(exc)}, status=500)

    hist_dates_sorted = sorted(prediction_payload['hist_preds'].keys())
    if not hist_dates_sorted:
        return JsonResponse({'error': '无法在该历史区间生成预测'}, status=400)

    short_dates_sorted = sorted(prediction_payload['short_preds'].keys())
    long_dates_sorted = sorted(prediction_payload['long_preds'].keys())

    hist_actual_mask = (df_extended['Date'] >= hist_start) & (df_extended['Date'] <= HISTORY_END_PERIOD)
    hist_actual_df = df_extended.loc[hist_actual_mask, ['Date', 'FSI_Raw']].copy()
    hist_actual_df['FSI_Raw'] = hist_actual_df['FSI_Raw'].ffill().bfill()
    full_history = get_fsi_history_entry()

    response = {
        'dates_hist': [str(p) for p in hist_dates_sorted],
        'fsi_hist': [round(prediction_payload['hist_preds'][p], 3) for p in hist_dates_sorted],
        'dates_hist_actual': hist_actual_df['Date'].astype(str).tolist(),
        'fsi_hist_actual': [round(float(v), 3) for v in hist_actual_df['FSI_Raw'].tolist()],
        'dates_hist_actual_full': full_history.get('dates', []),
        'fsi_hist_actual_full': full_history.get('fsi', []),
        'fsi_hist_smooth_full': full_history.get('fsi_smooth', []),
        'hist_news_full': full_history.get('news', []),
        'dates_short': [str(p) for p in short_dates_sorted],
        'fsi_short': [round(prediction_payload['short_preds'][p], 3) for p in short_dates_sorted],
        'dates_long': [str(p) for p in long_dates_sorted],
        'fsi_long': [round(prediction_payload['long_preds'][p], 3) for p in long_dates_sorted],
        'short_peak': prediction_payload['short_peak'],
        'short_peak_date': prediction_payload['short_peak_date'],
        'long_peak': prediction_payload['long_peak'],
        'long_peak_date': prediction_payload['long_peak_date'],
        'risk_prob': prediction_payload['risk_prob'],
        'risk_level': prediction_payload['risk_level'],
        'scenario_label': scenario_label,
    }
    return JsonResponse(response)


@csrf_exempt
def api_ppi_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)

    data = _json_body(request)
    try:
        hist_start = parse_month(data.get('hist_start', '2015-01'), '历史起点')
        short_end = parse_month(data.get('short_end', '2025-12'), '短期终点')
        long_end = parse_month(data.get('long_end', '2026-12'), '长期终点')
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)

    if short_end <= HISTORY_END_PERIOD:
        return JsonResponse({'error': '短期终点需晚于 2025-06'}, status=400)
    if long_end < short_end:
        return JsonResponse({'error': '长期终点需不早于短期终点'}, status=400)

    df = get_taxwar_dataframe()
    min_period = df['Date'].min()
    if hist_start < min_period:
        hist_start = min_period
    if hist_start >= HISTORY_END_PERIOD:
        return JsonResponse({'error': '历史区间必须早于 2025-06'}, status=400)

    preset = data.get('preset')
    scenario = PPI_SCENARIO_PRESETS.get(preset, {
        'ppi_shift': float(data.get('ppi_shift', 0)),
        'cpi_shift': float(data.get('cpi_shift', 0)),
        'rate_delta': float(data.get('rate_delta', 0)),
        'fx_depr': float(data.get('fx_depr', 0)),
        'iov_shift': float(data.get('iov_shift', 0)),
    })

    try:
        df_extended = extend_with_ppi_scenario(df, long_end, scenario)
        prediction_payload = compute_predictions(df_extended, hist_start, short_end, long_end, model_name='ragformer')
    except FileNotFoundError as exc:
        return JsonResponse({'error': str(exc)}, status=500)
    except Exception as exc:
        traceback.print_exc()
        return JsonResponse({'error': '模型推理失败', 'detail': str(exc)}, status=500)

    hist_dates_sorted = sorted(prediction_payload['hist_preds'].keys())
    if not hist_dates_sorted:
        return JsonResponse({'error': '无法在该历史区间生成预测'}, status=400)

    short_dates_sorted = sorted(prediction_payload['short_preds'].keys())
    long_dates_sorted = sorted(prediction_payload['long_preds'].keys())

    hist_actual_mask = (df_extended['Date'] >= hist_start) & (df_extended['Date'] <= HISTORY_END_PERIOD)
    hist_actual_df = df_extended.loc[hist_actual_mask, ['Date', 'FSI_Raw']].copy()
    hist_actual_df['FSI_Raw'] = hist_actual_df['FSI_Raw'].ffill().bfill()
    full_history = get_fsi_history_entry()

    response = {
        'dates_hist': [str(p) for p in hist_dates_sorted],
        'fsi_hist': [round(prediction_payload['hist_preds'][p], 3) for p in hist_dates_sorted],
        'dates_hist_actual': hist_actual_df['Date'].astype(str).tolist(),
        'fsi_hist_actual': [round(float(v), 3) for v in hist_actual_df['FSI_Raw'].tolist()],
        'dates_hist_actual_full': full_history.get('dates', []),
        'fsi_hist_actual_full': full_history.get('fsi', []),
        'fsi_hist_smooth_full': full_history.get('fsi_smooth', []),
        'hist_news_full': full_history.get('news', []),
        'dates_short': [str(p) for p in short_dates_sorted],
        'fsi_short': [round(prediction_payload['short_preds'][p], 3) for p in short_dates_sorted],
        'dates_long': [str(p) for p in long_dates_sorted],
        'fsi_long': [round(prediction_payload['long_preds'][p], 3) for p in long_dates_sorted],
        'short_peak': prediction_payload['short_peak'],
        'short_peak_date': prediction_payload['short_peak_date'],
        'long_peak': prediction_payload['long_peak'],
        'long_peak_date': prediction_payload['long_peak_date'],
        'risk_prob': prediction_payload['risk_prob'],
        'risk_level': prediction_payload['risk_level'],
        'scenario_label': f'PPI 情景（{preset or "custom"}）',
    }
    return JsonResponse(response)


@csrf_exempt
def api_taxwar_model_compare(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)

    data = _json_body(request)
    try:
        hist_start = parse_month(data.get('hist_start', '2018-01'), '历史起点')
        short_end = parse_month(data.get('short_end', '2025-12'), '短期终点')
        long_end = parse_month(data.get('long_end', '2026-12'), '长期终点')
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)

    if short_end <= HISTORY_END_PERIOD:
        return JsonResponse({'error': '短期终点需晚于 2025-06'}, status=400)
    if long_end < short_end:
        return JsonResponse({'error': '长期终点需不早于短期终点'}, status=400)

    scenario_keys = data.get('scenarios') or list(SCENARIO_PRESETS.keys())
    scenario_keys = [s for s in scenario_keys if s in SCENARIO_PRESETS]
    if not scenario_keys:
        scenario_keys = list(SCENARIO_PRESETS.keys())
    if 'baseline' not in scenario_keys:
        scenario_keys = ['baseline'] + scenario_keys

    requested_models = data.get('models') or []
    available_models = get_available_saved_models()
    if requested_models:
        model_names = [m for m in requested_models if m in available_models]
    else:
        model_names = available_models
    if not model_names:
        return JsonResponse({'error': 'saved_models 中未找到可用模型'}, status=400)

    df = get_taxwar_dataframe()
    min_period = df['Date'].min()
    if hist_start < min_period:
        hist_start = min_period
    if hist_start >= HISTORY_END_PERIOD:
        return JsonResponse({'error': '历史区间必须早于 2025-06'}, status=400)

    scenario_dfs = {key: extend_with_scenario(df, long_end, SCENARIO_PRESETS[key]) for key in scenario_keys}

    actual_hist_map = build_actual_history_map(df, hist_start)
    metrics_rows = []
    path_payload = {key: {} for key in scenario_keys}
    model_errors = []

    for model_name in model_names:
        baseline_payload = None
        baseline_future_map = {}

        try:
            baseline_payload = compute_predictions(
                scenario_dfs['baseline'], hist_start, short_end, long_end, model_name=model_name
            )
            baseline_future_map.update(baseline_payload['short_preds'])
            baseline_future_map.update(baseline_payload['long_preds'])
        except Exception as exc:
            model_errors.append({'model': model_name, 'scenario': 'baseline', 'error': str(exc)})
            continue

        for scenario_key in scenario_keys:
            payload = baseline_payload if scenario_key == 'baseline' else None
            if payload is None:
                try:
                    payload = compute_predictions(
                        scenario_dfs[scenario_key], hist_start, short_end, long_end, model_name=model_name
                    )
                except Exception as exc:
                    model_errors.append({'model': model_name, 'scenario': scenario_key, 'error': str(exc)})
                    continue

            hist_metrics = calc_alignment_metrics(payload['hist_preds'], actual_hist_map)
            scenario_future_map = {}
            scenario_future_map.update(payload['short_preds'])
            scenario_future_map.update(payload['long_preds'])
            future_metrics = calc_alignment_metrics(scenario_future_map, baseline_future_map)

            metrics_rows.append({
                'scenario': scenario_key,
                'scenario_label': SCENARIO_LABELS.get(scenario_key, scenario_key),
                'model': model_name,
                'hist_mse_vs_fsi': hist_metrics['mse'],
                'hist_mae_vs_fsi': hist_metrics['mae'],
                'hist_points': hist_metrics['count'],
                'future_mse_vs_baseline': future_metrics['mse'],
                'future_mae_vs_baseline': future_metrics['mae'],
                'future_points': future_metrics['count'],
                'short_peak': payload['short_peak'],
                'short_peak_date': payload['short_peak_date'],
                'long_peak': payload['long_peak'],
                'long_peak_date': payload['long_peak_date'],
                'risk_prob': payload['risk_prob'],
                'risk_level': payload['risk_level'],
            })

            hist_sorted = sorted(payload['hist_preds'].keys())
            short_sorted = sorted(payload['short_preds'].keys())
            long_sorted = sorted(payload['long_preds'].keys())
            path_payload[scenario_key][model_name] = {
                'dates_hist': [str(p) for p in hist_sorted],
                'fsi_hist': [round(float(payload['hist_preds'][p]), 4) for p in hist_sorted],
                'dates_short': [str(p) for p in short_sorted],
                'fsi_short': [round(float(payload['short_preds'][p]), 4) for p in short_sorted],
                'dates_long': [str(p) for p in long_sorted],
                'fsi_long': [round(float(payload['long_preds'][p]), 4) for p in long_sorted],
            }

    full_history = get_fsi_history_entry()
    hist_series = []
    for d, v in zip(full_history.get('dates', []), full_history.get('fsi', [])):
        p = parse_month(d, '日期')
        if hist_start <= p <= HISTORY_END_PERIOD:
            hist_series.append({'date': d, 'fsi': float(v)})

    response = {
        'hist_start': str(hist_start),
        'short_end': str(short_end),
        'long_end': str(long_end),
        'scenarios': scenario_keys,
        'models': model_names,
        'metrics': metrics_rows,
        'paths': path_payload,
        'fsi_actual': hist_series,
        'errors': model_errors,
    }
    return JsonResponse(response)
