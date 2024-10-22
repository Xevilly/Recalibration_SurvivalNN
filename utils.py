import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LinearRegression
# -----------------------------------------------------------

def cut_groups(risk,n_bins):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.percentile(risk, quantiles * 100)
    bins[-1] = bins[-1] + 1e-8
    group = np.digitize(risk, bins) - 1
    return group

def calibration_plot(df_plot, t0):
    kmf = KaplanMeierFitter()
    df_plot_table = np.empty(shape=(0, 5))
    int_group = df_plot["group"].unique().tolist()
    for i in int_group:
        temp = df_plot[df_plot["group"] == i]
        kmf.fit(temp["time"], event_observed=temp["outcome"], timeline=[t0])
        mean_risk = np.mean(temp["risk"])
        line = np.append(1 - kmf.survival_function_.values, 1 - kmf.confidence_interval_.values)
        line = np.append(line, mean_risk)
        line = np.append(line, i)
        df_plot_table = np.vstack([df_plot_table, line])
    df_plot_table = pd.DataFrame(df_plot_table, columns=["obsrisk", "low", "upper", "predrisk", "group"])
    df_plot_table = df_plot_table.sort_values(by="group", ascending=True)
    return df_plot_table

def EO_ratio(cali_df):
    model = LinearRegression().fit(cali_df["obsrisk"].values.reshape((-1,1)), cali_df["predrisk"])
    cons=model.intercept_
    beta=model.coef_
    return abs(beta-1), abs(cons-0)

def EO_ratio_subgp(cali_df,group):
    model = LinearRegression().fit(cali_df["obsrisk"].values.reshape((-1, 1)), cali_df["predrisk"])
    cons = model.intercept_
    beta = model.coef_
    subgp = cali_df[cali_df['group'] == group]
    beta_sub = subgp["predrisk"]/subgp["obsrisk"]
    return abs(beta-1), abs(cons-0), abs(beta_sub-1)

def prepare_data(pred, duration, events):
    return pd.DataFrame({
        "lp": pred[:, 0],
        "time": duration,
        "outcome": events
    })

def compute_metrics(df, time):
    outcome = df[["outcome", "time"]].to_numpy()
    outcome = np.array([(bool(e1), e2) for e1, e2 in outcome], dtype=[('status', '?'), ('survival_in_years', '<f8')])
    cindex = concordance_index(df['time'], 1-df['risk'], df['outcome'])
    score = brier_score(outcome, outcome, 1 - df['risk'].values, time)[1][0]
    return cindex, score
