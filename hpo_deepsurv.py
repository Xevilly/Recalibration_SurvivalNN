import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import brier_score_loss
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import torch # For building the networks
from pycox.models import CoxPH
from lifelines.utils import concordance_index
np.random.seed(1234)
torch.manual_seed(123)
torch.cuda.manual_seed(123)       
import torchtuples as tt
from sksurv.metrics import brier_score

def objective(trial):
    val = (x_val, y_val)
    in_features = x_train.shape[1]
    batch_size = trial.suggest_categorical('batch_size',[2048])
    epochs = 2048
    dropout =trial.suggest_categorical('dropout', [0.1, 0.3, 0.5])
    hid_layers = trial.suggest_int('hid_layers', 1, 2)
    nodes = [trial.suggest_categorical('n_units1',[3,5,7]),trial.suggest_categorical('n_units2',[3,5,7])]
    num_nodes=[]
    for i in range(0,hid_layers):
        num_nodes.append(nodes[i])

    lr_rate = trial.suggest_loguniform('lr_rate', 1e-4, 1e-2)
    weight_decay= trial.suggest_loguniform('weight_decay', 1e-4, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam','SGD'])
    momentum = trial.suggest_uniform('momentum', 0, 1)

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features=1, batch_norm=True, dropout=dropout, output_bias=False)

    if optimizer_name == 'Adam':
        model = CoxPH(net, tt.optim.Adam(lr=lr_rate,  weight_decay=weight_decay))
    else:
        model = CoxPH(net, tt.optim.SGD(lr=lr_rate,  weight_decay=weight_decay, momentum=momentum))

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks=[tt.callbacks.EarlyStopping()], verbose=True, val_data=val, val_batch_size=batch_size)
    pred = model.predict(x_train)
    cox_train = pd.DataFrame({"lp": pred[:, 0],
                              "month_fp_whocvd": duration_train,
                              "who_cvd": events_train})
    s0_compute_120= BaseCox.compute_s0(data=cox_train, lp= cox_train["lp"],time= 120)
    pred_val= model.predict(x_val)
    cox_val = pd.DataFrame({"lp": pred_val[:, 0],
                             "time": duration_val,
                             "outcome": events_val})
    cox_val["risk"] = 1 - s0_compute_120 ** np.exp(cox_val["lp"])

    outcome = cox_val[["outcome", "time"]].to_numpy()  # array from list
    aux = [(bool(e1), e2) for e1, e2 in outcome]  # list
    outcome = np.array(aux, dtype=[('status', '?'), ('survival_in_years', '<f8')])

    cindex = concordance_index(cox_val['time'], 1-cox_val['risk'], cox_val['outcome'])
    score = brier_score(outcome, outcome, 1 - cox_val['risk'].values, 120)[1][0]

    cox_val['group'] = cut_groups(cox_val["risk"], 10)
    cali_cox = calibration_plot(cox_val[["time", "outcome", "risk", "group"]], 120)
    beta, cons = EO_ratio(cali_cox)
    return cindex, beta, cons

n_startup_trials = 11 * x_train.shape[1]- 1
sampler = optuna.samplers.MOTPESampler(
    n_startup_trials=n_startup_trials,  seed=1234
)

if __name__ == "__main__":
    study = optuna.create_study(sampler=sampler,directions=["maximize","minimize","minimize"],
                                storage="sqlite:///Suvr_Sep_EO.db", study_name="Surv_Sep_EO", load_if_exists=True) 
    study.optimize(objective, n_trials=100, show_progress_bar = True)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    study = optuna.load_study(study_name="Surv_Sep_EO", storage="sqlite:///Surv_Sep_EO.db")
    len(study.trials)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trials
    # trial = study.trials

    for i in range(len(trial)):
        print("  Value: ", trial[i].values)

    print("  Params: ")
    for key, value in trial[0].params.items():
        print("    {}: {}".format(key, value))
