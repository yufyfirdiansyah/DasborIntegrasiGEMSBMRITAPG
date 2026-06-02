import os
import sys
import time
import json
import datetime
import threading
import traceback
import pandas as pd
import numpy as np

# Automatically install required packages
try:
    import joblib
except ImportError:
    joblib = None

try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError:
    print("[INFO] Flask is not installed. Installing via pip...")
    os.system("pip install flask")
    from flask import Flask, jsonify, render_template_string, request

try:
    import yfinance as yf
except ImportError:
    print("[INFO] yfinance is not installed. Installing via pip...")
    os.system("pip install yfinance")
    from flask import Flask, jsonify, render_template_string, request
    import yfinance as yf

try:
    import requests
except ImportError:
    print("[INFO] requests is not installed. Installing via pip...")
    os.system("pip install requests")
    import requests

# Initialize Flask
app = Flask(__name__)

BASE_DIR = os.getcwd()
IS_VERCEL = os.environ.get('VERCEL') == '1' or os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

# Unified Paths Mapping
ASSETS = ['GEMS', 'TAPG', 'BMRI']

PATHS = {
    'GEMS': {
        'MODEL': os.path.join(BASE_DIR, 'Project_GEMS_Modelling', 'gems_logistic_regression.joblib'),
        'SCALER': os.path.join(BASE_DIR, 'Project_GEMS_Modelling', 'gems_scaler.joblib'),
        'PARAMS': os.path.join(BASE_DIR, 'Project_GEMS_Modelling', 'gems_model_parameters.json'),
        'CACHE': '/tmp/gems_latest_prediction.json' if IS_VERCEL else os.path.join(BASE_DIR, 'Project_GEMS_Modelling', 'latest_prediction.json')
    },
    'TAPG': {
        'MODEL': os.path.join(BASE_DIR, 'Project_TAPG_Modelling', 'tapg_logistic_regression.joblib'),
        'SCALER': os.path.join(BASE_DIR, 'Project_TAPG_Modelling', 'tapg_scaler.joblib'),
        'PARAMS': os.path.join(BASE_DIR, 'Project_TAPG_Modelling', 'tapg_model_parameters.json'),
        'CACHE': '/tmp/tapg_latest_prediction.json' if IS_VERCEL else os.path.join(BASE_DIR, 'Project_TAPG_Modelling', 'latest_prediction.json')
    },
    'BMRI': {
        'MODEL': os.path.join(BASE_DIR, 'Project_BMRI_Modelling', 'bmri_logistic_regression.joblib'),
        'SCALER': os.path.join(BASE_DIR, 'Project_BMRI_Modelling', 'bmri_scaler.joblib'),
        'PARAMS': os.path.join(BASE_DIR, 'Project_BMRI_Modelling', 'bmri_model_parameters.json'),
        'CACHE': '/tmp/bmri_latest_prediction.json' if IS_VERCEL else os.path.join(BASE_DIR, 'Project_BMRI_Modelling', 'latest_prediction.json')
    }
}

CONFIG_PATH = '/tmp/unified_dashboard_config.json' if IS_VERCEL else os.path.join(BASE_DIR, 'dashboard_config.json')
LOG_PATH = '/tmp/unified_dashboard_logs.txt' if IS_VERCEL else os.path.join(BASE_DIR, 'dashboard_logs.txt')

system_logs = []

def log_message(msg):
    wib_now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    timestamp = wib_now.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    system_logs.append(formatted)
    if len(system_logs) > 60:
        system_logs.pop(0)
    try:
        with open(LOG_PATH, 'a') as f:
            f.write(formatted + "\n")
    except:
        pass

# Initialize log file
try:
    with open(LOG_PATH, 'w') as f:
        f.write("=== LOG STARTUP SYSTEM DASHBOARD MULTI-EMITEN ===\n")
except:
    pass

# Dynamic Config Helpers
def get_webhook_url():
    webhook_url = os.environ.get('WEBHOOK_URL', '').strip()
    if not webhook_url and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config_data = json.load(f)
                webhook_url = config_data.get('webhook_url', '').strip()
        except:
            pass
    return webhook_url

# =====================================================================
# CORE QUANT ENGINE: CALCULATIONS & PREDICTIONS
# =====================================================================

def fetch_yesterday_vwaps(ticker):
    log_message(f"[{ticker}] Fetching Yesterday's completed EOD VWAP from 5m candles...")
    try:
        df_i = yf.download(ticker, period='5d', interval='5m', progress=False)
        if not df_i.empty:
            if isinstance(df_i.columns, pd.MultiIndex):
                df_i.columns = df_i.columns.get_level_values(0)
            
            df_i['DateOnly'] = df_i.index.date
            dates = df_i['DateOnly'].unique()
            if len(dates) >= 2:
                latest_date = dates[-1]  # Yesterday (since we EOD lock and today isn't closed)
                prev_date = dates[-2]    # Day before yesterday
                
                df_latest = df_i[df_i['DateOnly'] == latest_date]
                df_prev = df_i[df_i['DateOnly'] == prev_date]
                
                vwap_latest = (df_latest['Close'] * df_latest['Volume']).sum() / df_latest['Volume'].sum()
                vwap_prev = (df_prev['Close'] * df_prev['Volume']).sum() / df_prev['Volume'].sum()
                
                log_message(f"[{ticker}] VWAP Yesterday: Rp {vwap_latest:.2f}, Prev: Rp {vwap_prev:.2f}")
                return float(vwap_latest), float(vwap_prev)
            elif len(dates) == 1:
                df_latest = df_i
                vwap_latest = (df_latest['Close'] * df_latest['Volume']).sum() / df_latest['Volume'].sum()
                return float(vwap_latest), float(vwap_latest * 0.995)
    except Exception as e:
        log_message(f"[{ticker} WARNING] Failed to compute Yesterday's VWAP: {e}")
    return None, None

def run_prediction_for_asset(asset_key):
    log_message(f"\n[{asset_key}] Starting daily prediction pipeline...")
    
    asset_paths = PATHS[asset_key]
    
    # 1. Fetch Model parameters (prefer JSON first, fall back to Joblib)
    gems_params = None
    if os.path.exists(asset_paths['PARAMS']):
        try:
            with open(asset_paths['PARAMS'], 'r') as f:
                gems_params = json.load(f)
            log_message(f"[{asset_key}] Standard JSON parameters loaded successfully.")
        except Exception as e:
            log_message(f"[{asset_key} WARNING] Failed to load JSON parameters: {e}")
            
    if gems_params is None:
        if not os.path.exists(asset_paths['MODEL']) or not os.path.exists(asset_paths['SCALER']):
            raise Exception(f"Model files (.json or .joblib) for {asset_key} not available. Please run training first.")
        if joblib is None:
            raise Exception("Library 'joblib' is not available to load .joblib files.")
        model = joblib.load(asset_paths['MODEL'])
        scaler = joblib.load(asset_paths['SCALER'])
        log_message(f"[{asset_key}] Fallback Joblib model & scaler loaded successfully.")
    else:
        model = None
        scaler = None
        
    # 2. Config tickers
    if asset_key == 'GEMS':
        primary_ticker = 'GEMS.JK'
        macros = {
            'FXI': 'FXI',
            'NG': 'NG=F',
            'USDIDR': 'USDIDR=X',
            'BTU': 'BTU',
            'WHC': 'WHC.AX'
        }
    elif asset_key == 'TAPG':
        primary_ticker = 'TAPG.JK'
        macros = {
            'WTI': 'CL=F',
            'Brent': 'BZ=F',
            'Soybean': 'ZS=F',
            'EWM': 'EWM',
            'CPO': 'FCPO=F',
            'USDIDR': 'USDIDR=X'
        }
    else:  # BMRI
        primary_ticker = 'BMRI.JK'
        macros = {
            'EIDO': 'EIDO',
            'NIKKEI': '^N225',
            'USDIDR': 'USDIDR=X'
        }
        
    # 3. Pull daily prices
    tickers_list = [primary_ticker] + list(macros.values())
    raw_data = {}
    log_message(f"[{asset_key}] Fetching daily data from Yahoo Finance...")
    try:
        all_df = yf.download(" ".join(tickers_list), period='60d', interval='1d', progress=False, group_by='ticker')
    except Exception as ex:
        log_message(f"[{asset_key} ERROR] Failed to fetch bulk data: {ex}")
        all_df = pd.DataFrame()
        
    for k, ticker in ([(asset_key, primary_ticker)] + list(macros.items())):
        try:
            if not all_df.empty:
                df_t = all_df[ticker].dropna(subset=['Close'])
                if isinstance(df_t.columns, pd.MultiIndex):
                    df_t.columns = df_t.columns.get_level_values(0)
            else:
                df_t = pd.DataFrame()
            raw_data[k] = df_t
        except Exception as ex:
            log_message(f"[{asset_key} WARNING] Missing data for ticker {ticker}: {ex}")
            raw_data[k] = pd.DataFrame()
            
    # Validate primary asset
    primary_df = raw_data[asset_key]
    if primary_df.empty:
        raise Exception(f"Primary asset {primary_ticker} data is empty! Sync failed.")
        
    primary_df = primary_df.reset_index()
    
    # 4. Ingest and Align Dates (IDX anchor)
    aligned = pd.DataFrame({'Date': primary_df['Date']})
    aligned[f'{asset_key}_Close'] = primary_df['Close'].values
    aligned[f'{asset_key}_Open'] = primary_df['Open'].values
    aligned[f'{asset_key}_Volume'] = primary_df['Volume'].values
    aligned[f'{asset_key}_High'] = primary_df['High'].values
    aligned[f'{asset_key}_Low'] = primary_df['Low'].values
    
    for k in macros.keys():
        df_m = raw_data[k]
        if not df_m.empty:
            df_m = df_m.reset_index()
            df_m = df_m[['Date', 'Close']].rename(columns={'Close': f'{k}_Close'})
            aligned = pd.merge(aligned, df_m, on='Date', how='left')
        else:
            aligned[f'{k}_Close'] = np.nan
            
    # Apply forward-fill / backward-fill & replace remaining NaNs
    aligned = aligned.sort_values('Date').reset_index(drop=True)
    aligned = aligned.ffill().bfill()
    aligned = aligned.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    # [EOD LOCK] Ignore today's volatile dynamic bar if market is active (before 16:15 WIB)
    if not aligned.empty:
        wib_now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
        today_wib_str = wib_now.strftime('%Y-%m-%d')
        latest_row_date_str = aligned.iloc[-1]['Date'].strftime('%Y-%m-%d')
        
        if latest_row_date_str == today_wib_str:
            if wib_now.time() < datetime.time(16, 15):
                log_message(f"[{asset_key} EOD LOCK] Skipping today's active bar ({latest_row_date_str}) before close. Using yesterday's candle.")
                aligned = aligned.iloc[:-1].reset_index(drop=True)
            else:
                log_message(f"[{asset_key} EOD LOCK] Standard EOD prediction utilizing today's closed daily candle ({latest_row_date_str}).")
                
    # Calculate returns
    for k in macros.keys():
        aligned[f'{k}_Return'] = aligned[f'{k}_Close'].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
    # Calculate tech indicators
    aligned[f'{asset_key}_MA20'] = aligned[f'{asset_key}_Close'].rolling(window=20).mean()
    aligned[f'{asset_key}_Volume_MA20'] = aligned[f'{asset_key}_Volume'].rolling(window=20).mean()
    aligned[f'{asset_key}_Return'] = aligned[f'{asset_key}_Close'].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    aligned[f'{asset_key}_Volume_Change'] = aligned[f'{asset_key}_Volume'].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    aligned = aligned.ffill().bfill().fillna(0.0)
    
    # Fetch latest EOD locked row for features
    latest_row = aligned.iloc[-1]
    latest_date = latest_row['Date'].strftime('%Y-%m-%d')
    log_message(f"[{asset_key}] Selected feature date for tomorrow's prediction: {latest_date}")
    
    # 5. Calculate yesterday's EOD VWAP (dynamically fetched from 5m bars)
    vwap_latest, vwap_prev = fetch_yesterday_vwaps(primary_ticker)
    if vwap_latest is None:
        # Fallback to Close price if intraday calculations fail
        vwap_latest = float(latest_row[f'{asset_key}_Close'])
        vwap_prev = float(latest_row[f'{asset_key}_Close']) * 0.998
        
    # 6. Extract Standard Binary Representation features
    # Standard rule: 0 if return > 0, 1 if return <= 0
    binary_features = []
    global_sentiment = {}
    
    for k in macros.keys():
        ret_val = float(latest_row[f'{k}_Return'])
        bin_val = 0 if ret_val > 0 else 1
        binary_features.append(bin_val)
        global_sentiment[k] = ret_val * 100
        
    # Standard technical internal feature: Volume_Change
    vol_change_val = float(latest_row[f'{asset_key}_Volume_Change'])
    vol_change_bin = 0 if vol_change_val > 0 else 1
    binary_features.append(vol_change_bin)
    
    # 7. Predict using standardized coefficients (math representation)
    if gems_params is not None:
        coef = gems_params['coef']
        intercept = gems_params['intercept']
        mean = gems_params['mean']
        scale = gems_params['scale']
        
        # StandardScaler: (x - mean) / scale
        scaled_x = [(x - m) / s for x, m, s in zip(binary_features, mean, scale)]
        
        # Logistic: z = sum(w * x) + b
        z = sum(w * x for w, x in zip(coef, scaled_x)) + intercept
        
        # Sigmoid: prob of class 1 (SELL)
        prob_class_1 = 1.0 / (1.0 + np.exp(-z))
        
        pred = 1 if prob_class_1 >= 0.5 else 0
        proba = [1.0 - prob_class_1, prob_class_1]
    else:
        # Fallback to Joblib prediction
        feature_cols = gems_params.get('feature_names') if gems_params else [f'{k}_Bin' for k in macros.keys()] + [f'{asset_key}_Volume_Change_Bin']
        X_latest = pd.DataFrame([binary_features], columns=feature_cols)
        X_scaled = scaler.transform(X_latest)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
    # 8. Standardized signals mapping (0 = BUY, 1 = SELL)
    decision = "BUY" if pred == 0 else "SELL"
    confidence = proba[0] if pred == 0 else proba[1]
    
    # Calculate bullish ratios
    bullish_count = 0
    for k in macros.keys():
        if float(latest_row[f'{k}_Return']) > 0:
            bullish_count += 1
            
    result = {
        'asset': asset_key,
        'last_sync': (datetime.datetime.utcnow() + datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB"),
        'target_date': latest_date,
        'decision': decision,
        'signal_code': int(pred),
        'confidence': float(confidence),
        'bullish_ratio': f"{bullish_count}/{len(macros)} BULLISH",
        'price': {
            'close': float(latest_row[f'{asset_key}_Close']),
            'open': float(latest_row[f'{asset_key}_Open']),
            'volume': int(latest_row[f'{asset_key}_Volume']),
            'high': float(latest_row[f'{asset_key}_High']),
            'low': float(latest_row[f'{asset_key}_Low'])
        },
        'raw_prices': {k: float(latest_row[f'{k}_Close']) for k in macros.keys()},
        'technical': {
            'ma20': float(latest_row[f'{asset_key}_MA20']),
            'vwap': vwap_latest,
            'prev_vwap': vwap_prev,
            'volume_ma20': float(latest_row[f'{asset_key}_Volume_MA20']),
            'change_pct': float(latest_row[f'{asset_key}_Return'] * 100),
            'volume_change_pct': float(latest_row[f'{asset_key}_Volume_Change'] * 100)
        },
        'global_sentiment': global_sentiment
    }
    
    # Save to local cache
    try:
        with open(asset_paths['CACHE'], 'w') as f:
            json.dump(result, f, indent=4)
        log_message(f"[{asset_key} SUCCESS] Predicted: {decision} ({confidence*100:.2f}%) successfully cached.")
    except Exception as e:
        log_message(f"[{asset_key} WARNING] Failed to cache prediction: {e}")
        
    # Push update to Google Sheets Webhook
    push_daily_row_to_sheets(result)
    
    return result

def push_daily_row_to_sheets(result):
    webhook_url = get_webhook_url()
    if not webhook_url:
        log_message(f"[{result['asset']} SHEETS] Webhook URL not configured. Skipping Sheets logging.")
        return
        
    log_message(f"[{result['asset']} SHEETS] Sending real-time daily prediction to Google Sheet via Webhook...")
    try:
        response = requests.post(
            webhook_url, 
            json=result, 
            headers={"Content-Type": "application/json"}, 
            timeout=15
        )
        if response.status_code == 200:
            res_data = response.json()
            if res_data.get('status') == 'success':
                log_message(f"[{result['asset']} SHEETS SUCCESS] Successfully logged to Google Sheet!")
            else:
                log_message(f"[{result['asset']} SHEETS WARNING] Google Sheet webhook returned error: {res_data.get('message')}")
        else:
            log_message(f"[{result['asset']} SHEETS ERROR] HTTP fail: {response.status_code}")
    except Exception as e:
        log_message(f"[{result['asset']} SHEETS ERROR] Connection failed: {e}")

# =====================================================================
# BACKGROUND AUTOMATED SCHEDULER (08:45 WIB Daily Run)
# =====================================================================

def run_multi_scheduler_loop():
    log_message("Background multi-ticker scheduler initiated for 08:45 WIB WIB...")
    last_run_date = None
    
    while True:
        try:
            now = datetime.datetime.now()
            current_time_str = now.strftime("%H:%M")
            current_date_str = now.strftime("%Y-%m-%d")
            
            if current_time_str == "08:45" and last_run_date != current_date_str:
                log_message("--- AUTOMATED DAILY MULTI-EMITEN TRADING SCHEDULER STARTING (08:45 WIB) ---")
                for asset in ASSETS:
                    try:
                        run_prediction_for_asset(asset)
                    except Exception as e:
                        log_message(f"[SCHEDULER ERROR] Failed predicting {asset}: {e}")
                last_run_date = current_date_str
                log_message("Automated daily trading scheduler finished syncing all assets.")
        except Exception as e:
            log_message(f"[SCHEDULER EXCEPTION] Error in scheduler loop: {e}")
            
        time.sleep(30)

# Run background thread
scheduler_thread = threading.Thread(target=run_multi_scheduler_loop, daemon=True)
scheduler_thread.start()

# Load initial predictions on startup
def load_startup_predictions():
    log_message("Startup loading predictions for GEMS, TAPG, and BMRI...")
    for asset in ASSETS:
        try:
            cache_path = PATHS[asset]['CACHE']
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                log_message(f"[{asset}] Loaded predictions from local cache.")
            else:
                run_prediction_for_asset(asset)
        except Exception as e:
            log_message(f"[STARTUP WARNING] Failed initializing GEMS/TAPG/BMRI: {e}")

load_startup_predictions()

# =====================================================================
# API SINK ENDPOINTS FOR FRONTEND AND REAL-TIME LOGGING
# =====================================================================

@app.route('/api/prediction/<asset>', methods=['GET'])
def get_prediction(asset):
    asset = asset.upper()
    if asset not in ASSETS:
        return jsonify({'success': False, 'message': f'Invalid asset: {asset}'})
        
    try:
        cache_path = PATHS[asset]['CACHE']
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Map legacy cache price keys to standardized 'price'
            legacy_price_key = f"{asset.lower()}_price"
            if legacy_price_key in data and 'price' not in data:
                data['price'] = data[legacy_price_key]
                
            return jsonify({'success': True, 'data': data, 'logs': system_logs})
        else:
            data = run_prediction_for_asset(asset)
            return jsonify({'success': True, 'data': data, 'logs': system_logs})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'logs': system_logs})

@app.route('/api/sync/<asset>', methods=['POST'])
def force_sync(asset):
    asset = asset.upper()
    if asset not in ASSETS:
        return jsonify({'success': False, 'message': f'Invalid asset: {asset}'})
        
    try:
        result = run_prediction_for_asset(asset)
        return jsonify({'success': True, 'data': result, 'logs': system_logs})
    except Exception as e:
        log_message(f"[{asset} SYNC FAILED] Manual sync crashed: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'message': str(e), 'logs': system_logs})

@app.route('/api/cron', methods=['GET'])
def trigger_daily_cron():
    log_message("=== VERCEL CRON TRIGGERED: DAILY MULTI-EMITEN TRADING SCHEDULER STARTING (08:45 WIB) ===")
    results = {}
    errors = []
    
    for asset in ASSETS:
        try:
            log_message(f"[CRON] Running predictive pipeline for {asset}...")
            result = run_prediction_for_asset(asset)
            results[asset] = {
                'success': True,
                'decision': result.get('decision', 'HOLD'),
                'confidence': result.get('confidence', 0.0),
                'price': result.get('price', 0.0)
            }
        except Exception as e:
            err_msg = f"[CRON ERROR] Failed predicting {asset}: {str(e)}"
            log_message(err_msg)
            errors.append(err_msg)
            results[asset] = {'success': False, 'message': str(e)}
            
    status_code = 200 if not errors else 500
    return jsonify({
        'success': len(errors) == 0,
        'message': 'Cron sync job completed' if len(errors) == 0 else f'Cron sync completed with errors: {len(errors)}',
        'results': results,
        'errors': errors,
        'logs': system_logs[-10:]
    }), status_code

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        try:
            data = request.json or {}
            webhook_url = data.get('webhook_url', '').strip()
            
            config_data = {}
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    config_data = json.load(f)
                    
            config_data['webhook_url'] = webhook_url
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config_data, f, indent=4)
                
            log_message(f"Google Sheet Webhook URL saved successfully: {webhook_url}")
            return jsonify({'success': True, 'message': 'Google Sheet Webhook URL saved successfully!'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    else:
        try:
            webhook_url = get_webhook_url()
            return jsonify({'success': True, 'webhook_url': webhook_url})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

# =====================================================================
# PREMIUM SHADCN/UI STYLE INTEGRATED QUANTITATIVE TERMINAL (CSS/HTML)
# =====================================================================

UNIFIED_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integrated Quantitative Terminal (GEMS • TAPG • BMRI)</title>
    <!-- Google Fonts Inter & JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-app: #fafafa;
            --bg-card: #ffffff;
            --bg-card-hover: #f4f4f5;
            --border-color: #e4e4e7;
            --text-main: #09090b;
            --text-muted: #71717a;
            --accent: #ff5500;
            --accent-hover: #e04b00;
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            
            --color-buy: #10b981;
            --color-buy-bg: #ecfdf5;
            --color-sell: #ef4444;
            --color-sell-bg: #fef2f2;
            
            --transition: all 0.2s ease-in-out;
        }
        
        body.dark {
            --bg-app: #09090b;
            --bg-card: #121215;
            --bg-card-hover: #181820;
            --border-color: #27221f;
            --text-main: #f3f3f6;
            --text-muted: #a1a1aa;
            --accent: #ff5500;
            --accent-hover: #ff7733;
            --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.4);
            
            --color-buy: #10b981;
            --color-buy-bg: rgba(16, 185, 129, 0.12);
            --color-sell: #ef4444;
            --color-sell-bg: rgba(239, 68, 68, 0.12);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-app);
            color: var(--text-main);
            transition: var(--transition);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding-bottom: 2rem;
            letter-spacing: -0.01em;
        }
        
        /* HEADER */
        header {
            padding: 1.25rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            background-color: var(--bg-card);
            z-index: 50;
            box-shadow: var(--shadow);
        }
        
        .header-title-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .header-title {
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            text-transform: uppercase;
        }
        
        .header-title span {
            color: var(--accent);
        }
        
        .header-badge {
            font-size: 0.65rem;
            font-weight: 700;
            color: var(--text-main);
            border: 1px solid var(--text-main);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .sync-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            margin-left: 1rem;
            text-transform: uppercase;
        }
        
        .header-controls {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .theme-toggle {
            cursor: pointer;
            width: 40px;
            height: 40px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: var(--bg-card);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-main);
            transition: var(--transition);
            box-shadow: var(--shadow);
        }
        
        .theme-toggle:hover {
            background: var(--bg-card-hover);
            transform: translateY(-1px);
        }
        
        /* TERMINAL TICKERS SELECTOR TABS */
        .tabs-container {
            display: flex;
            background-color: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 0.5rem 2rem;
            gap: 0.5rem;
        }
        
        .tab-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 0.6rem 1.5rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            text-transform: uppercase;
        }
        
        .tab-btn:hover {
            color: var(--text-main);
            background-color: var(--bg-card-hover);
        }
        
        .tab-btn.active {
            color: #ffffff;
            background-color: var(--accent);
            border-color: var(--accent);
            box-shadow: var(--shadow);
        }
        
        .tab-btn.active span.lbl-ticker {
            color: rgba(255,255,255,0.7);
        }
        
        .tab-btn span.lbl-ticker {
            font-size: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            background-color: rgba(0,0,0,0.06);
            padding: 0.1rem 0.35rem;
            border-radius: 4px;
            color: var(--text-muted);
        }
        
        body.dark .tab-btn span.lbl-ticker {
            background-color: rgba(255,255,255,0.06);
        }
        
        /* DASHBOARD CONTENT CONTAINER */
        .dashboard-container {
            max-width: 1400px;
            margin: 1.5rem auto;
            width: 100%;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 1.5rem;
            flex-grow: 1;
        }
        
        @media (max-width: 1024px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
        }
        
        /* LEFT COLUMN */
        .panel-left {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }
        
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            transition: var(--transition);
        }
        
        .card-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }
        
        .card-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.05em;
        }
        
        .card-badge {
            font-size: 0.65rem;
            font-weight: 800;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            text-transform: uppercase;
        }
        
        .card-badge.up {
            color: var(--accent);
            border: 1px solid var(--accent);
            background-color: rgba(255, 85, 0, 0.08);
        }
        
        /* SENTIMENT LIST */
        .vector-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .vector-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            background-color: rgba(0, 0, 0, 0.02);
            border: 1px solid var(--border-color);
            transition: var(--transition);
        }
        
        body.dark .vector-item {
            background-color: rgba(255, 255, 255, 0.01);
        }
        
        .vector-item:hover {
            background-color: var(--bg-card-hover);
        }
        
        .vector-name {
            font-weight: 700;
            font-size: 0.85rem;
        }
        
        .vector-name span {
            font-size: 0.65rem;
            color: var(--text-muted);
            font-weight: 500;
            display: block;
            margin-top: 0.1rem;
            text-transform: uppercase;
        }
        
        .vector-meta {
            text-align: right;
        }
        
        .vector-change {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 0.15rem 0.35rem;
            border-radius: 4px;
        }
        
        .vector-price {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            margin-top: 0.15rem;
        }
        
        .vector-logic-box {
            margin-top: 1.25rem;
            padding: 0.85rem 1rem;
            border-radius: 10px;
            border: 1px dashed rgba(255, 85, 0, 0.3);
            background-color: rgba(255, 85, 0, 0.02);
            font-size: 0.75rem;
            line-height: 1.4;
            color: var(--text-muted);
            font-style: italic;
        }
        
        /* VOLUME FLOW */
        .vol-flow-card {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .vol-value {
            font-size: 2.25rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: -0.04em;
        }
        
        .vol-progress-container {
            width: 100%;
            height: 6px;
            background-color: var(--border-color);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 0.25rem;
        }
        
        .vol-progress-bar {
            height: 100%;
            background-color: var(--accent);
            border-radius: 3px;
            width: 0%;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
        
        .vol-ma-label {
            display: flex;
            justify-content: flex-end;
            font-size: 0.7rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
        }
        
        /* CENTER PANEL */
        .panel-right {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }
        
        /* HERO HERO CARD */
        .asset-hero-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2.25rem;
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: center;
            overflow: hidden;
            box-shadow: var(--shadow);
            min-height: 280px;
        }
        
        .asset-hero-left {
            z-index: 10;
        }
        
        .asset-sub-lbl {
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: 800;
            color: var(--accent);
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .asset-title {
            font-size: 4.5rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.04em;
            margin-bottom: 0.75rem;
            color: var(--text-main);
        }
        
        .asset-desc-lbl {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        
        /* WATERMARK */
        .asset-hero-card::after {
            content: 'QUANT';
            position: absolute;
            right: 25%;
            top: 5%;
            font-size: 15rem;
            font-weight: 900;
            color: rgba(255, 85, 0, 0.02);
            line-height: 1;
            pointer-events: none;
            z-index: 1;
        }
        
        /* CLOSE PRICE BLOCK */
        .close-metrics-box {
            border: 1px solid var(--border-color);
            background-color: rgba(0, 0, 0, 0.02);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: left;
            min-width: 250px;
            z-index: 10;
            box-shadow: var(--shadow);
        }
        
        body.dark .close-metrics-box {
            background-color: rgba(255, 255, 255, 0.01);
        }
        
        .close-title {
            font-size: 0.65rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .close-price {
            font-size: 2.5rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: -0.04em;
            line-height: 1;
            margin-bottom: 0.75rem;
        }
        
        .close-details {
            display: flex;
            gap: 1.25rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
        }
        
        .close-details span.change {
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* TECH SUB-GRID */
        .asset-sub-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }
        
        @media (max-width: 640px) {
            .asset-sub-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .sub-tech-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.75rem;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 170px;
        }
        
        .sub-tech-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .sub-tech-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.05em;
        }
        
        .sub-tech-badge {
            font-size: 0.65rem;
            font-weight: 800;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            text-transform: uppercase;
            border: 1px solid transparent;
        }
        
        .sub-tech-badge.bullish {
            color: var(--color-buy);
            border-color: var(--color-buy);
            background-color: rgba(16, 185, 129, 0.08);
        }
        
        .sub-tech-badge.bearish {
            color: var(--color-sell);
            border-color: var(--color-sell);
            background-color: rgba(239, 68, 68, 0.08);
        }
        
        .sub-tech-val {
            font-size: 2.25rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: -0.03em;
            margin: 0.5rem 0;
            line-height: 1;
        }
        
        .sub-tech-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 500;
        }
        
        .sub-tech-desc span {
            font-family: 'JetBrains Mono', monospace;
            margin-left: 0.25rem;
        }
        
        /* BOTTOM CONTROLS & DEV CONSOLE */
        .console-collapsible {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            box-shadow: var(--shadow);
        }
        
        .console-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        
        .console-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .console-body {
            margin-top: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .webhook-flex {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }
        
        .webhook-input {
            flex-grow: 1;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: rgba(0, 0, 0, 0.02);
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            outline: none;
            transition: var(--transition);
        }
        
        body.dark .webhook-input {
            background-color: rgba(255, 255, 255, 0.02);
        }
        
        .webhook-input:focus {
            border-color: var(--accent);
        }
        
        .btn {
            background-color: var(--text-main);
            color: var(--bg-app);
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.25rem;
            font-weight: 700;
            font-size: 0.75rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: var(--shadow);
            transition: var(--transition);
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        
        .btn:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-accent {
            background-color: var(--accent);
            color: #ffffff;
        }
        
        .btn-accent:hover {
            background-color: var(--accent-hover);
        }
        
        .log-console {
            background-color: #050508;
            border: 1px solid #14141a;
            border-radius: 10px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #a1a1aa;
            height: 140px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .log-line {
            line-height: 1.4;
        }
        
        .log-line.error { color: var(--color-sell); }
        .log-line.warning { color: #fbbf24; }
        .log-line.success { color: var(--color-buy); }
        
        /* VALUE UTILITIES */
        .val-up {
            color: var(--color-buy);
            background-color: rgba(16, 185, 129, 0.08);
        }
        
        .val-down {
            color: var(--color-sell);
            background-color: rgba(239, 68, 68, 0.08);
        }
    </style>
</head>
<body class="dark">

    <header>
        <div class="header-title-container">
            <h1 class="header-title"><span>QUANTITATIVE</span> GATEWAY</h1>
            <div class="header-badge" id="asset-sector-badge">SECTOR</div>
            <span class="sync-label" id="sync-time-lbl">SYNC: LOADING...</span>
        </div>
        
        <div class="header-controls">
            <!-- Live Predictive Signal Badge next to theme toggle -->
            <div id="header-signal-badge" style="display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.85rem; border-radius: 8px; font-weight: 800; font-size: 0.85rem; border: 1px solid var(--border-color); background-color: var(--bg-card); box-shadow: var(--shadow); transition: var(--transition);">
                <i class="fa-solid fa-chart-line"></i>
                <span id="header-signal-lbl" style="text-transform: uppercase;">SIGNAL: --</span>
                <span id="header-confidence-lbl" style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 600; opacity: 0.9;">--%</span>
            </div>
            
            <div class="theme-toggle" id="theme-btn" title="Ubah Tema">
                <i class="fa-solid fa-sun" id="theme-icon"></i>
            </div>
        </div>
    </header>
    
    <!-- TERMINAL SELECTOR TABS -->
    <nav class="tabs-container">
        <button class="tab-btn active" onclick="switchAsset('GEMS')">
            GEMS <span class="lbl-ticker">GEMS.JK</span>
        </button>
        <button class="tab-btn" onclick="switchAsset('TAPG')">
            TAPG <span class="lbl-ticker">TAPG.JK</span>
        </button>
        <button class="tab-btn" onclick="switchAsset('BMRI')">
            BMRI <span class="lbl-ticker">BMRI.JK</span>
        </button>
    </nav>

    <main class="dashboard-container">
        
        <!-- LEFT COLUMN -->
        <section class="panel-left">
            
            <!-- GLOBAL sentiment VECTORS -->
            <div class="card">
                <div class="card-header-row">
                    <div class="card-title">Macro Sentiment Vectors</div>
                    <div class="card-badge up" id="bullish-ratio-badge">LOADING...</div>
                </div>
                
                <div class="vector-list" id="vector-sentiment-list">
                    <div style="text-align: center; color: var(--text-muted); padding: 2rem 0;">
                        <i class="fa-solid fa-spinner fa-spin" style="font-size: 1.25rem; margin-bottom: 0.5rem;"></i>
                        <p>Mengambil Data Makro...</p>
                    </div>
                </div>
                

            </div>
            
            <!-- VOLUME FLOW -->
            <div class="card">
                <div class="card-header-row">
                    <div class="card-title">Volume Flow</div>
                    <div class="card-badge" id="vol-flow-badge" style="background-color: var(--color-sell-bg); color: var(--color-sell);">LOW VOL</div>
                </div>
                
                <div class="vol-flow-card">
                    <div class="vol-value" id="vol-val-lbl">0</div>
                    <div class="vol-progress-container">
                        <div class="vol-progress-bar" id="vol-progress-bar" style="width: 0%;"></div>
                    </div>
                    <div class="vol-ma-label" id="vol-ma-lbl">MA: 0</div>
                </div>
            </div>
            
        </section>
        
        <!-- CENTER PANEL -->
        <section class="panel-right">
            
            <!-- HERO & CLOSE PRICE -->
            <div class="asset-hero-card">
                <div class="asset-hero-left">
                    <div class="asset-sub-lbl" id="hero-sector-lbl">IDX SECTOR</div>
                    <h2 class="asset-title" id="hero-title-lbl">GEMS</h2>
                    <div class="asset-desc-lbl" id="hero-company-lbl">INDONESIA</div>
                </div>
                
                <!-- CURRENT CLOSE BLOCK -->
                <div class="close-metrics-box">
                    <div class="close-title" id="close-date-lbl">Current Close</div>
                    <div class="close-price" id="close-price-lbl">0</div>
                    <div class="close-details">
                        <div>OPEN <span style="color: var(--text-main);" id="open-price-lbl">0</span></div>
                        <div>CHANGE <span class="change" id="change-pct-lbl" style="color: var(--color-buy);">0.00%</span></div>
                    </div>
                </div>
            </div>
            
            <!-- SUB-GRID DETAILS -->
            <div class="asset-sub-grid">
                
                <!-- TREND MA20 -->
                <div class="sub-tech-card">
                    <div class="sub-tech-header">
                        <div class="sub-tech-title">Baseline Trend [MA20]</div>
                        <div class="sub-tech-badge bearish" id="trend-badge">Bearish</div>
                    </div>
                    <div class="sub-tech-val" id="trend-ma20-lbl">Rp 0</div>
                    <div class="sub-tech-desc">Moving Average baseline</div>
                </div>
                
                <!-- YESTERDAY'S VWAP -->
                <div class="sub-tech-card">
                    <div class="sub-tech-header">
                        <div class="sub-tech-title">EOD VWAP Yesterday</div>
                        <div class="sub-tech-badge bearish" id="vwap-badge">Below</div>
                    </div>
                    <div class="sub-tech-val" id="vwap-lbl">Rp 0</div>
                    <div class="sub-tech-desc" id="vwap-prev-desc">Prev EOD: Rp 0 <i class="fa-solid fa-arrow-trend-down" id="vwap-arrow" style="font-size: 0.75rem; color: var(--color-sell);"></i></div>
                </div>
                
            </div>
            
            <!-- CONFIG & LOGS PANEL -->
            <div class="console-collapsible">
                <div class="console-header" id="console-toggle-btn">
                    <div class="console-title">
                        <i class="fa-solid fa-terminal" style="color: var(--accent);"></i>
                        <span>Configuration & System Console</span>
                    </div>
                    <i class="fa-solid fa-chevron-down" id="console-arrow"></i>
                </div>
                
                <div class="console-body" id="console-body">
                    <div class="webhook-flex">
                        <input type="text" id="webhook-input" class="webhook-input" placeholder="Google Sheet Webhook URL (https://script.google.com/macros/s/.../exec)">
                        <button class="btn btn-accent" id="save-webhook-btn">
                            <i class="fa-solid fa-floppy-disk"></i> Simpan
                        </button>
                        <button class="btn" id="sync-btn">
                            <i class="fa-solid fa-arrows-rotate" id="sync-icon"></i> Sync Ticker
                        </button>
                    </div>
                    
                    <div class="log-console" id="log-console">
                        <!-- Log lines -->
                    </div>
                </div>
            </div>
            
        </section>
        
    </main>

    <script>
        let currentAsset = 'GEMS';
        
        // DOM Elements
        const bodyEl = document.body;
        const themeBtn = document.getElementById('theme-btn');
        const themeIcon = document.getElementById('theme-icon');
        const syncBtn = document.getElementById('sync-btn');
        const syncIcon = document.getElementById('sync-icon');
        const logConsole = document.getElementById('log-console');
        const webhookInput = document.getElementById('webhook-input');
        const saveWebhookBtn = document.getElementById('save-webhook-btn');
        const consoleToggleBtn = document.getElementById('console-toggle-btn');
        const consoleBody = document.getElementById('console-body');
        const consoleArrow = document.getElementById('console-arrow');
        
        // Collapsible Handler
        let isCollapsed = false;
        consoleToggleBtn.addEventListener('click', () => {
            isCollapsed = !isCollapsed;
            if (isCollapsed) {
                consoleBody.style.display = "none";
                consoleArrow.className = "fa-solid fa-chevron-right";
            } else {
                consoleBody.style.display = "flex";
                consoleArrow.className = "fa-solid fa-chevron-down";
            }
        });
        
        // Dark/Light Theme Handler
        themeBtn.addEventListener('click', () => {
            if (bodyEl.classList.contains('dark')) {
                bodyEl.classList.remove('dark');
                themeIcon.className = "fa-solid fa-moon";
            } else {
                bodyEl.classList.add('dark');
                themeIcon.className = "fa-solid fa-sun";
            }
        });
        
        // Tab switching
        function switchAsset(assetName) {
            currentAsset = assetName;
            
            // Toggle active button
            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => {
                if (btn.innerText.includes(assetName)) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            
            // Refresh prediction data immediately
            fetchPrediction();
        }
        
        // Helper formatting
        function formatNumber(num) {
            return new Intl.NumberFormat('id-ID').format(num);
        }
        
        function formatRupiah(num) {
            return "Rp " + new Intl.NumberFormat('id-ID', { minimumFractionDigits: 1, maximumFractionDigits: 2 }).format(num);
        }
        
        // Update dashboard UI
        function updateUI(data) {
            // Header Sync Date & Badge Ticker
            document.getElementById('sync-time-lbl').innerText = `SYNC: ${data.last_sync.replace(' WIB', '')}`;
            
            const sectorMapping = {
                'GEMS': 'COAL SECTOR • IDX',
                'TAPG': 'PALM OIL SECTOR • IDX',
                'BMRI': 'BANKING SECTOR • IDX'
            };
            const nameMapping = {
                'GEMS': 'Golden Energy & Mines',
                'TAPG': 'Triputra Agro Persada',
                'BMRI': 'Bank Mandiri Tbk'
            };
            const watermarkMapping = {
                'GEMS': 'COAL',
                'TAPG': 'CPO',
                'BMRI': 'BANK'
            };
            
            const activeAsset = data.asset || currentAsset;
            document.getElementById('asset-sector-badge').innerText = sectorMapping[activeAsset] || 'SECTOR';
            document.getElementById('hero-sector-lbl').innerText = sectorMapping[activeAsset] || 'SECTOR';
            document.getElementById('hero-title-lbl').innerText = activeAsset;
            document.getElementById('hero-company-lbl').innerText = `• ${nameMapping[activeAsset] || 'Not Yet Calibrated'} • INDONESIA`;
            
            // Update Top Header Signal & Confidence Badge
            const headerSignalLbl = document.getElementById('header-signal-lbl');
            const headerConfidenceLbl = document.getElementById('header-confidence-lbl');
            const headerSignalBadge = document.getElementById('header-signal-badge');
            
            if (data.decision && data.confidence) {
                const confPct = (data.confidence * 100).toFixed(2);
                headerSignalLbl.innerText = `${data.decision}`;
                headerConfidenceLbl.innerText = `${confPct}%`;
                headerSignalBadge.style.display = "flex";
                
                if (data.decision === "BUY") {
                    headerSignalBadge.style.backgroundColor = "var(--color-buy-bg)";
                    headerSignalBadge.style.color = "var(--color-buy)";
                    headerSignalBadge.style.borderColor = "var(--color-buy)";
                } else {
                    headerSignalBadge.style.backgroundColor = "var(--color-sell-bg)";
                    headerSignalBadge.style.color = "var(--color-sell)";
                    headerSignalBadge.style.borderColor = "var(--color-sell)";
                }
            } else {
                headerSignalBadge.style.display = "none";
            }
            
            // SOP Decision & glow indicator
            const vwapBadge = document.getElementById('vwap-badge');
            const heroCard = document.querySelector('.asset-hero-card');
            
            // Set dynamic background glow based on BUY (green) or SELL (red)
            if (data.decision === "BUY") {
                heroCard.style.borderLeft = "6px solid var(--color-buy)";
            } else {
                heroCard.style.borderLeft = "6px solid var(--color-sell)";
            }
            
            // Global Sentiment Vectors Ratio
            const bullishRatio = document.getElementById('bullish-ratio-badge');
            const ratioStr = data.bullish_ratio || "0/5 NEUTRAL";
            bullishRatio.innerText = ratioStr;
            const parts = ratioStr.split(' ')[0].split('/');
            const numerator = parseInt(parts[0]) || 0;
            const denominator = parseInt(parts[1]) || 5;
            const pct = numerator / denominator;
            
            if (pct >= 0.7) {
                bullishRatio.style.borderColor = "var(--color-buy)";
                bullishRatio.style.color = "var(--color-buy)";
                bullishRatio.style.backgroundColor = "rgba(16, 185, 129, 0.08)";
            } else if (pct <= 0.3) {
                bullishRatio.style.borderColor = "var(--color-sell)";
                bullishRatio.style.color = "var(--color-sell)";
                bullishRatio.style.backgroundColor = "rgba(239, 68, 68, 0.08)";
            } else {
                bullishRatio.style.borderColor = "var(--accent)";
                bullishRatio.style.color = "var(--accent)";
                bullishRatio.style.backgroundColor = "rgba(255, 85, 0, 0.08)";
            }
            
            // Global macro sentiment list
            const vectorSentimentList = document.getElementById('vector-sentiment-list');
            vectorSentimentList.innerHTML = "";
            
            const detailedNames = {
                'FXI': 'FXI [CHINA]',
                'NG': 'NG=F [GAS]',
                'USDIDR': 'USD/IDR',
                'BTU': 'BTU [NYSE]',
                'WHC': 'WHC.AX [ASX]',
                'WTI': 'WTI Crude Oil',
                'Brent': 'Brent Crude',
                'Soybean': 'Soybean Meal',
                'EWM': 'EWM [MALAYSIA]',
                'CPO': 'CPO Futures',
                'EIDO': 'EIDO [INDO ETF]',
                'NIKKEI': '^N225 [NIKKEI]'
            };
            
            const mappingNames = {
                'FXI': 'China Large Cap',
                'NG': 'Natural Gas Commodities',
                'USDIDR': 'Currency Margin',
                'BTU': 'Coal Sector NYSE',
                'WHC': 'Australian Coal Proxy',
                'WTI': 'WTI Oil Commodity',
                'Brent': 'Brent Oil Commodity',
                'Soybean': 'Agricultural Index',
                'EWM': 'Malaysia Index ETF',
                'CPO': 'Palm Oil Commodities',
                'EIDO': 'MSCI Indonesia Index',
                'NIKKEI': 'Japan Equity Benchmark'
            };
            
            if (data.global_sentiment) {
                Object.keys(data.global_sentiment).forEach(key => {
                    const returnVal = data.global_sentiment[key];
                    const price = data.raw_prices ? data.raw_prices[key] : 0;
                    
                    const item = document.createElement('div');
                    item.className = "vector-item";
                    
                    const sign = returnVal >= 0 ? "+" : "";
                    const valClass = returnVal >= 0 ? "val-up" : "val-down";
                    
                    let formattedPrice = formatNumber(price);
                    if (key === "USDIDR") {
                        formattedPrice = formatNumber(Math.round(price));
                    } else if (price % 1 !== 0) {
                        formattedPrice = price.toFixed(2);
                    }
                    
                    item.innerHTML = `
                        <div class="vector-name">
                            ${detailedNames[key] || key}
                            <span>${mappingNames[key] || ''}</span>
                        </div>
                        <div class="vector-meta">
                            <span class="vector-change ${valClass}">${sign}${returnVal.toFixed(2)}%</span>
                            <div class="vector-price">${formattedPrice}</div>
                        </div>
                    `;
                    vectorSentimentList.appendChild(item);
                });
            }
            
            // Volume Flow Card
            if (data.price) {
                document.getElementById('vol-val-lbl').innerText = formatNumber(data.price.volume);
                document.getElementById('vol-ma-lbl').innerText = `MA: ${formatNumber(Math.round(data.technical.volume_ma20))}`;
                
                const volFlowBadge = document.getElementById('vol-flow-badge');
                const volProgressBar = document.getElementById('vol-progress-bar');
                
                const ratio = (data.price.volume / data.technical.volume_ma20) * 100;
                volProgressBar.style.width = `${Math.min(ratio, 100)}%`;
                
                if (data.price.volume >= data.technical.volume_ma20) {
                    volFlowBadge.innerText = "HIGH VOL";
                    volFlowBadge.style.backgroundColor = "rgba(16, 185, 129, 0.08)";
                    volFlowBadge.style.color = "var(--color-buy)";
                    volProgressBar.style.backgroundColor = "var(--color-buy)";
                } else {
                    volFlowBadge.innerText = "LOW VOL";
                    volFlowBadge.style.backgroundColor = "rgba(239, 68, 68, 0.08)";
                    volFlowBadge.style.color = "var(--color-sell)";
                    volProgressBar.style.backgroundColor = "var(--accent)";
                }
                
                // Hero Block Close open high
                let formattedClose = formatNumber(data.price.close);
                if (data.price.close % 1 !== 0) {
                    formattedClose = data.price.close.toFixed(3);
                }
                document.getElementById('close-price-lbl').innerText = formattedClose;
                
                let formattedOpen = formatNumber(data.price.open);
                if (data.price.open % 1 !== 0) {
                    formattedOpen = data.price.open.toFixed(3);
                }
                document.getElementById('open-price-lbl').innerText = "Rp " + formattedOpen;
            }
            
            document.getElementById('close-date-lbl').innerText = `Locked EOD Price (${data.target_date || 'N/A'})`;
            
            if (data.technical) {
                const changePct = data.technical.change_pct;
                const changePctEl = document.getElementById('change-pct-lbl');
                changePctEl.innerText = (changePct >= 0 ? "+" : "") + changePct.toFixed(2) + "%";
                changePctEl.style.color = changePct >= 0 ? "var(--color-buy)" : "var(--color-sell)";
                
                // Trend MA20
                document.getElementById('trend-ma20-lbl').innerText = formatRupiah(data.technical.ma20);
                const trendBadge = document.getElementById('trend-badge');
                if (data.price && data.price.close >= data.technical.ma20) {
                    trendBadge.innerText = "Bullish";
                    trendBadge.className = "sub-tech-badge bullish";
                } else {
                    trendBadge.innerText = "Bearish";
                    trendBadge.className = "sub-tech-badge bearish";
                }
                
                // Yesterday's VWAP
                document.getElementById('vwap-lbl').innerText = formatRupiah(data.technical.vwap);
                document.getElementById('vwap-prev-desc').innerHTML = `Prev EOD: Rp ${formatNumber(data.technical.prev_vwap.toFixed(2))} <i class="fa-solid ${data.technical.vwap >= data.technical.prev_vwap ? 'fa-arrow-trend-up' : 'fa-arrow-trend-down'}" id="vwap-arrow" style="font-size: 0.75rem; color: ${data.technical.vwap >= data.technical.prev_vwap ? 'var(--color-buy)' : 'var(--color-sell)'};"></i>`;
                
                const vwapBadgeState = document.getElementById('vwap-badge');
                if (data.price && data.price.close >= data.technical.vwap) {
                    vwapBadgeState.innerText = "Above";
                    vwapBadgeState.className = "sub-tech-badge bullish";
                } else {
                    vwapBadgeState.innerText = "Below";
                    vwapBadgeState.className = "sub-tech-badge bearish";
                }
            }
        }
        
        // Reset UI when asset is untrained
        function showUntrainedWarning(msg) {
            document.getElementById('asset-sector-badge').innerText = "UNTRAINED";
            document.getElementById('hero-sector-lbl').innerText = "MODEL UNTRAINED / PENDING";
            document.getElementById('hero-title-lbl').innerText = currentAsset;
            document.getElementById('hero-company-lbl').innerText = "• Model calibration required • INDONESIA";
            
            const heroCard = document.querySelector('.asset-hero-card');
            heroCard.style.borderLeft = "6px solid var(--border-color)";
            
            document.getElementById('close-price-lbl').innerText = "---";
            document.getElementById('open-price-lbl').innerText = "Rp ---";
            document.getElementById('change-pct-lbl').innerText = "0.00%";
            document.getElementById('change-pct-lbl').style.color = "var(--text-muted)";
            document.getElementById('close-date-lbl').innerText = "No locked candle available";
            
            document.getElementById('trend-ma20-lbl').innerText = "Rp ---";
            const trendBadge = document.getElementById('trend-badge');
            trendBadge.innerText = "Unknown";
            trendBadge.className = "sub-tech-badge";
            
            document.getElementById('vwap-lbl').innerText = "Rp ---";
            document.getElementById('vwap-prev-desc').innerHTML = "Prev EOD: Rp ---";
            const vwapBadgeState = document.getElementById('vwap-badge');
            vwapBadgeState.innerText = "Unknown";
            vwapBadgeState.className = "sub-tech-badge";
            
            document.getElementById('bullish-ratio-badge').innerText = "N/A";
            document.getElementById('bullish-ratio-badge').style.borderColor = "var(--border-color)";
            document.getElementById('bullish-ratio-badge').style.color = "var(--text-muted)";
            document.getElementById('bullish-ratio-badge').style.backgroundColor = "transparent";
            
            document.getElementById('vector-sentiment-list').innerHTML = `
                <div style="text-align: center; color: var(--text-muted); padding: 2.5rem 0;">
                    <i class="fa-solid fa-triangle-exclamation" style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--accent);"></i>
                    <p style="font-size: 0.8rem; font-weight:600;">Model calibration is required to generate vectors</p>
                </div>
            `;
            
            document.getElementById('vol-val-lbl').innerText = "---";
            document.getElementById('vol-ma-lbl').innerText = "MA: ---";
            document.getElementById('vol-progress-bar').style.width = "0%";
            const volFlowBadge = document.getElementById('vol-flow-badge');
            volFlowBadge.innerText = "NO DATA";
            volFlowBadge.style.backgroundColor = "transparent";
            volFlowBadge.style.color = "var(--text-muted)";
            volFlowBadge.style.borderColor = "var(--border-color)";
        }
        
        // Fetch Cache
        async function fetchPrediction() {
            try {
                const response = await fetch(`/api/prediction/${currentAsset}`);
                const result = await response.json();
                if (result.success) {
                    updateUI(result.data);
                    updateLogs(result.logs);
                } else {
                    showUntrainedWarning(result.message);
                    updateLogs(result.logs);
                }
            } catch (err) {
                console.error("Gagal menarik data prediksi:", err);
            }
        }
        
        // Fetch Config
        async function fetchConfig() {
            try {
                const response = await fetch('/api/config');
                const result = await response.json();
                if (result.success && result.webhook_url) {
                    webhookInput.value = result.webhook_url;
                }
            } catch (err) {
                console.error("Gagal menarik konfigurasi:", err);
            }
        }
        
        // Save Webhook URL
        saveWebhookBtn.addEventListener('click', async () => {
            saveWebhookBtn.disabled = true;
            const webhookUrl = webhookInput.value.trim();
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ webhook_url: webhookUrl })
                });
                const result = await response.json();
                if (result.success) {
                    alert("Sukses! " + result.message);
                } else {
                    alert("Gagal: " + result.message);
                }
            } catch (err) {
                alert("Gagal menyimpan konfigurasi.");
            } finally {
                saveWebhookBtn.disabled = false;
            }
        });
        
        // Force Sync Manual Ticker
        syncBtn.addEventListener('click', async () => {
            syncBtn.disabled = true;
            syncIcon.classList.add('fa-spin');
            
            const logLine = document.createElement('div');
            logLine.className = "log-line success";
            logLine.innerText = `[${new Date().toLocaleTimeString()}] Memulai sinkronisasi manual & kalkulasi model untuk ${currentAsset}...`;
            logConsole.appendChild(logLine);
            logConsole.scrollTop = logConsole.scrollHeight;
            
            try {
                const response = await fetch(`/api/sync/${currentAsset}`, { method: 'POST' });
                const result = await response.json();
                if (result.success) {
                    updateUI(result.data);
                    updateLogs(result.logs);
                } else {
                    alert("Error Sync: " + result.message);
                }
            } catch (err) {
                alert("Gagal melakukan sync manual.");
            } finally {
                syncBtn.disabled = false;
                syncIcon.classList.remove('fa-spin');
            }
        });
        
        // Update Console Logs
        function updateLogs(logs) {
            logConsole.innerHTML = "";
            logs.forEach(log => {
                const line = document.createElement('div');
                line.className = "log-line";
                if (log.includes("[ERROR]")) line.classList.add("error");
                else if (log.includes("[WARNING]")) line.classList.add("warning");
                else if (log.includes("SUCCESS") || log.includes("sukses") || log.includes("Berhasil")) line.classList.add("success");
                
                line.innerText = log;
                logConsole.appendChild(line);
            });
            logConsole.scrollTop = logConsole.scrollHeight;
        }
        
        // Startup Polling
        fetchPrediction();
        fetchConfig();
        setInterval(fetchPrediction, 20000); // Polling harian every 20s
        
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(UNIFIED_HTML_TEMPLATE)

if __name__ == '__main__':
    log_message("=== STARTING UNIFIED TRADING DASHBOARD INTEGRATION SYSTEM ===")
    log_message(f"Active workspace models: GEMS, TAPG, BMRI")
    # Bind to port 5001 (or custom Vercel settings)
    app.run(host='127.0.0.1', port=5001, debug=False)
