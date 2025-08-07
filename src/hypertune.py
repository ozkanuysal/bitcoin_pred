import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import logging
from .data import load_data # Changed to relative import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the parameter grid - Based on Prophet documentation recommendations
param_grid = {
    'changepoint_prior_scale': [0.0001, 0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

# Metrics to calculate (use mean across horizon)
METRICS = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']

# Explanations for metrics
METRIC_EXPLANATIONS = {
    'mse': "Mean Squared Error: Average of the squares of the errors. Lower is better. Sensitive to large errors.",
    'rmse': "Root Mean Squared Error: Square root of MSE. In the same units as the target variable. Lower is better.",
    'mae': "Mean Absolute Error: Average of the absolute errors. Less sensitive to outliers than MSE/RMSE. Lower is better.",
    'mape': "Mean Absolute Percentage Error: Average of absolute percentage errors. Lower is better. Undefined if actual value is zero.",
    'mdape': "Median Absolute Percentage Error: Median of absolute percentage errors. Robust to outliers. Lower is better.",
    'smape': "Symmetric Mean Absolute Percentage Error: Percentage error based on the average of actual and forecast values. Ranges from 0% to 200%. Lower is better.",
    'coverage': "Coverage: Proportion of actual values that fall within the predicted uncertainty interval (yhat_lower, yhat_upper). Higher is better (closer to the nominal interval width, e.g., 0.8 for 80% interval)."
}

# Pass CV params (as integers representing days) to the cached function
@st.cache_data 
def run_hyperparameter_tuning(df_log, initial_days, period_days, horizon_days):
    results = []
    total_combinations = len(all_params)
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Starting hyperparameter tuning for {total_combinations} combinations...") 

    # Format integers into strings for Prophet
    initial_str = f"{initial_days} days"
    period_str = f"{period_days} days"
    horizon_str = f"{horizon_days} days"

    for i, params in enumerate(all_params):
        status_text.text(f"Testing combination {i+1}/{total_combinations}: {params}")
        logging.info(f"Testing parameters: {params} with CV params: initial={initial_str}, period={period_str}, horizon={horizon_str}")
        
        current_metrics = {'params': params}
        for metric in METRICS:
            current_metrics[metric] = float('inf')

        try:
            m = Prophet(**params).fit(df_log)
            df_cv = cross_validation(m, initial=initial_str, period=period_str, horizon=horizon_str, parallel="processes", disable_tqdm=True)
            df_p = performance_metrics(df_cv, metrics=METRICS, rolling_window=0.9)

            if not df_p.empty:
                for metric in METRICS:
                    if metric in df_p.columns and not df_p[metric].isnull().all():
                        current_metrics[metric] = df_p[metric].mean()
                    else: 
                        current_metrics[metric] = float('inf')
            
            logging.info(f"Parameters: {params} - Metrics: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in current_metrics.items() if k != 'params'} }")

        except Exception as e:
            logging.error(f"Failed for parameters {params} with CV params: {initial_str}, {period_str}, {horizon_str}: {e}")

        results.append(current_metrics)
        progress_bar.progress((i + 1) / total_combinations)

    status_text.text("Hyperparameter tuning function complete.")
    progress_bar.empty()
    
    results_df = pd.DataFrame(results)
    for metric in METRICS:
        results_df[metric] = pd.to_numeric(results_df[metric], errors='coerce').fillna(float('inf'))
         
    results_df = results_df.sort_values(by='rmse')
    
    if not results_df.empty and results_df.iloc[0]['rmse'] != float('inf'):
        best_params = results_df.iloc[0]['params']
        best_rmse = results_df.iloc[0]['rmse']
        logging.info(f"Best Parameters (based on RMSE): {best_params} with RMSE: {best_rmse:.4f}")
    else:
        best_params = "N/A"
        best_rmse = float('inf')
        logging.warning("Could not find best parameters. All runs may have failed or produced invalid metrics.")
    
    return results_df, best_params, best_rmse

def hypertune_app(): 
    st.title("Hyperparameter Tuning")

    st.subheader("Model Hyperparameters")

    st.markdown("**`changepoint_prior_scale`**: Parameter modulating the flexibility of the automatic changepoint selection. Large values will allow many changepoints, small values will allow few changepoints.")
    st.write(f"Values tested: `{param_grid['changepoint_prior_scale']}`")
    
    st.markdown("**`seasonality_prior_scale`**: Parameter modulating the strength of the seasonality model. Larger values allow the model to fit larger seasonal fluctuations, smaller values dampen the seasonality.")
    st.write(f"Values tested: `{param_grid['seasonality_prior_scale']}`")
    
    st.subheader("Cross-Validation Settings (in Days)")
    col1, col2, col3 = st.columns(3)
    with col1:
        # Use number_input for days
        initial_cv_days = st.number_input(
            "Initial Training Period (Days)", 
            min_value=1, 
            value=4000, # Default as integer
            max_value=4500,
            step=10,
            help="The amount of historical data (in days) to use for the first training cutoff."
        )
    with col2:
        period_cv_days = st.number_input(
            "Cutoff Period (Days)", 
            min_value=1,
            value=180, # Default as integer
            max_value=365,
            step=10,
            help="The spacing (in days) between subsequent training cutoffs."
        )
    with col3:
        horizon_cv_days = st.number_input(
            "Forecast Horizon (Days)", 
            min_value=1,
            value=365, # Default as integer
            max_value=365,
            step=1,
            help="The length of the forecast (in days) to make after each cutoff."
        )

    st.subheader("Start Tuning Process")
    st.info("This process computes multiple metrics across several parameter combinations using the specified cross-validation settings. It can take several minutes.")
    
    start_tuning = st.button("Start Hyperparameter Tuning")
    if not start_tuning:
        return 
    # No need for string validation anymore
    with st.spinner("Running cross-validation... Please wait."):
        # Pass the integer day values to the tuning function
        df = load_data()
        df = df.copy()
        df = df.rename(columns={'price': 'y', 'timestamp': 'ds'})
        df['y_orig'] = df['y'] 
        df['y'] = np.log(df['y'])
        df = df[['ds', 'y']] # Prepare data subset for tuning function
        results_display, best_params, best_rmse = run_hyperparameter_tuning(
            df, 
            initial_days=initial_cv_days, 
            period_days=period_cv_days, 
            horizon_days=horizon_cv_days
        )
    
    st.subheader("Tuning Results (Sorted by RMSE)")
    
    if not results_display.empty and 'params' in results_display.columns and isinstance(results_display['params'].iloc[0], dict):
        params_df = results_display['params'].apply(pd.Series)
        results_display = pd.concat([params_df, results_display.drop(['params'], axis=1)], axis=1)
        
    cols_order = list(param_grid.keys()) + METRICS
    cols_order = [col for col in cols_order if col in results_display.columns] 
    results_display = results_display[cols_order]
    
    float_cols = results_display.select_dtypes(include=['float']).columns
    column_config = {col: st.column_config.NumberColumn(format="%.4f") for col in float_cols}
    for metric, explanation in METRIC_EXPLANATIONS.items():
        if metric in column_config:
                column_config[metric] = st.column_config.NumberColumn(format="%.4f", help=explanation)

    st.dataframe(results_display, column_config=column_config)

    st.subheader("Best Parameters Found (based on lowest RMSE)")
    if best_rmse != float('inf') and best_params != "N/A":
        st.success(f"Lowest Mean RMSE: {best_rmse:.4f}")
        st.json(best_params)
    else:
        st.warning("Hyperparameter tuning failed to find optimal parameters based on RMSE.")