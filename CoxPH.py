from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

class BaseCox():
    @staticmethod
    def compute_s0(data, lp, time):
        partial_hazards = np.exp(lp)
        predicted_partial_hazards = (
            partial_hazards
                .to_frame(name="P")
                .reindex(data.index)
                .assign(T=data["month_fp_whocvd"], E=data["who_cvd"], W=1)
                .set_index(data.index)
        )
        ind_hazards = predicted_partial_hazards
        ind_hazards_summed_over_durations = ind_hazards.groupby("T")[["P", "E"]].sum()
        ind_hazards_summed_over_durations["P"] = ind_hazards_summed_over_durations["P"].loc[::-1].cumsum()
        baseline_hazard = pd.DataFrame(
            ind_hazards_summed_over_durations["E"] / ind_hazards_summed_over_durations["P"]
        )
        cumulative = baseline_hazard.cumsum()
        survival_df = np.exp(-cumulative)
        s0_compute = survival_df.iloc[time].values
        return s0_compute
    
    @staticmethod
    def compute_lp(data, coef):
        lp = (data.drop(['who_cvd', 'month_fp_whocvd'], axis=1) * coef).sum(axis=1)
        return lp

    @staticmethod
    def fit(data, time):
        cph = CoxPHFitter()
        cph.fit(data, duration_col='month_fp_whocvd', event_col='who_cvd')
        coef = cph.summary["coef"]
        lp = BaseCox.compute_lp(data, coef)
        s0_compute = BaseCox.compute_s0(data, lp, time)

        return cph, coef, s0_compute

    @staticmethod
    def predict_risk(data, coef, s0_compute):
        lp = data.drop(['who_cvd', 'month_fp_whocvd'], axis=1) * coef
        lp = np.sum(lp, axis=1)
        risk = 1- s0_compute**np.exp(lp)
        return risk
