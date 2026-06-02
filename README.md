# Unified Quantitative Trading Terminal 📈
### GEMS • TAPG • BMRI | Predictive Analytics & Automated IDX Trading Signal Hub

A premium, serverless-optimized quantitative terminal designed to forecast directional IDX market trends and automate daily log updates to Google Sheets. Powered by mathematical parameter inference (ultra-lightweight, zero cold-start latency), it runs natively on Vercel Serverless Functions.

---

## 🚀 Key Features

*   **Inter Premium UI Layout**: Clean modern fintech styling (glassmorphism tabs, dark theme, interactive tables, responsive charts).
*   **Predictive Signal Badge**: Top-level header indicator presenting real-time trading decisions (**BUY / SELL**) and model **Confidence %**.
*   **Pure Math Inference**: Built-in weight matrix coefficients from Logistic Regression. Eliminates dependencies on `scikit-learn` and `joblib` for Vercel size limit compliance.
*   **Automated EOD Cron Job**: Seamlessly updates Google Sheets every IDX trading day at **08:45 WIB** via Vercel Cron.
*   **Unified Multi-Ticker Architecture**: Centralized Flask dashboard managing **GEMS (Golden Energy & Mines)**, **TAPG (Triputra Agro Persada)**, and **BMRI (Bank Mandiri)** simultaneously.

---

## 🛠️ Local Installation & Setup

To run and preview the terminal locally on your machine:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yufyfirdiansyah/DasborIntegrasiGEMSBMRITAPG.git
    cd DasborIntegrasiGEMSBMRITAPG
    ```

2.  **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the Flask Server**:
    ```bash
    python app.py
    ```

4.  **Open in Browser**:
    Navigate to **[http://127.0.0.1:5001](http://127.0.0.1:5001)** to view the live dashboard.

---

## ☁️ Deploying to Vercel (Step-by-Step)

The repository is pre-configured and serverless-ready. To launch it globally:

1.  Log in to **[Vercel](https://vercel.com/)** with your GitHub account.
2.  Click **Add New** → **Project**.
3.  Find `DasborIntegrasiGEMSBMRITAPG` from your imported repository list and click **Import**.
4.  Leave the build and environment settings at their default values (Vercel automatically detects `vercel.json` and setups the Python runtime).
5.  Click **Deploy**.
6.  Once deployed, your live production terminal URL is ready!

---

## ⏰ Cron Job Schedule & Webhook Sync

The automated daily market calculations are governed by the cron job inside `vercel.json`:

*   **Schedule**: Runs at `01:45 UTC` (which matches **08:45 WIB** Jakarta time) every Monday to Friday.
*   **Trigger**: Hits the `/api/cron` route of your serverless app.
*   **Action**: Calls `yfinance` to grab the latestIDX candle, extracts technical features (Volume MA20, MA20, EOD VWAP), feeds them into the calibrated model parameters, and submits the results directly to your Google Sheets webhook.

---

## 📂 Core Production Directory Structure

```text
DasborIntegrasiGEMSBMRITAPG/
│
├── Project_BMRI_Modelling/           
│   └── bmri_model_parameters.json    # Calibrated mathematical weights for BMRI
│
├── Project_GEMS_Modelling/           
│   └── gems_model_parameters.json    # Calibrated mathematical weights for GEMS
│
├── Project_TAPG_Modelling/           
│   └── tapg_model_parameters.json    # Calibrated mathematical weights for TAPG
│
├── app.py                            # Unified Flask engine (HTML template, predictions, config API)
├── vercel.json                       # Vercel serverless functions routing and 08:45 WIB cron trigger
├── requirements.txt                  # Minimum serverless dependencies (under 1MB memory footprint)
└── .gitignore                        # Prevents pushing binary .joblib weights, logs, and pdf files
```

*Created with ❤️ by **Antigravity AI** for yufyfirdiansyah.*
