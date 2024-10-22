# first run import_data.py and CoxPH.py and then import data using run.py (until discrete time)
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import brier_score_loss
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import torch # For building the networks
from pycox.models import CoxPH
from lifelines.utils import concordance_index
import torchtuples as tt
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
np.random.seed(1234)
_ = torch.manual_seed(123)
import torchtuples as tt


def objective(trial):
    val = (x_val, y_val)
    in_features = x_train.shape[1]
    batch_size = trial.suggest_categorical('batch_size',[1024])
    epochs = 2048
    dropout = trial.suggest_categorical('dropout', [0.1, 0.3, 0.5])
    hid_layers = trial.suggest_categorical('hid_layers', [3,4])
    nodes = [trial.suggest_categorical('n_units1', [7, 14, 21, 28]), trial.suggest_categorical('n_units2',[7,14,21,28]),trial.suggest_categorical('n_units3',[7,14,21 ,28]),trial.suggest_categorical('n_units4',[7,14,21,28])]

    num_nodes=[]
    for i in range(0, hid_layers):
        num_nodes.append(nodes[i])

    lr_rate = trial.suggest_loguniform('lr_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-2)
  
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam','SGD'])
    alpha = trial.suggest_categorical('alpha', [0.2,0.4,0.6])
    momentum = trial.suggest_uniform('momentum', 0, 1)

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features=labtrans.out_features, batch_norm=True, dropout=dropout, output_bias=False)

    if optimizer_name == 'Adam':
        model = DeepHitSingle(net, tt.optim.Adam(lr=lr_rate,  weight_decay=weight_decay),alpha=alpha, sigma=10, duration_index=labtrans.cuts)
    else:
        model = DeepHitSingle(net, tt.optim.SGD(lr=lr_rate,  weight_decay=weight_decay, momentum=momentum),alpha=alpha, sigma=10, duration_index=labtrans.cuts)

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks=[tt.callbacks.EarlyStopping(min_delta=0.001)], verbose=True, val_data=val, val_batch_size=batch_size)

    surv = model.predict_surv_df(x_val)
    ev = EvalSurv(surv, duration_val, events_val, censor_surv='km')
    risk = np.clip(1 - ev.surv_at_times(120).squeeze(), 1e-10, 1 - 1e-10)
    cox_val = pd.DataFrame({"risk": risk,
                            "time": duration_val,
                            "outcome": events_val})

    outcome = cox_val[["outcome", "time"]].to_numpy()  
    aux = [(bool(e1), e2) for e1, e2 in outcome]  
    outcome = np.array(aux, dtype=[('status', '?'), ('survival_in_years', '<f8')])

    cindex = concordance_index(cox_val['time'], 1-cox_val['risk'], cox_val['outcome'])
    score =brier_score(outcome, outcome, 1-cox_val['risk'].values, 120)[1][0]

    cox_val['group'] = cut_groups(cox_val["risk"], 10)
    cali_cox = calibration_plot(cox_val[["time", "outcome", "risk", "group"]],120)
    beta, cons, beta_sub = EO_ratio_subgp(cali_cox, group=9)

    return cindex, beta, cons, beta_sub


n_startup_trials = 11 * x_train.shape[1]- 1
sampler = optuna.samplers.MOTPESampler(
    n_startup_trials=n_startup_trials,  seed=1234
)

if __name__ == "__main__":


    study = optuna.create_study(sampler=sampler,
                                directions=["maximize", "minimize","minimize","minimize"],
    storage="sqlite:///hit_Sep_newcut_EOsub.db", study_name="hit_Sep_newcut_EOsub", load_if_exists=True)  
    study.optimize(objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trials

    for i in range(len(trial)):
        print("  Value: ", trial[i].values)

    print("  Params: ")
    for key, value in trial[0].params.items():
        print("    {}: {}".format(key, value))