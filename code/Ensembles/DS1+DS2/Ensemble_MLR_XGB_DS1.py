import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, cross_validate, learning_curve
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
import dataframe_image as dfi
import shap
import warnings
warnings.filterwarnings("ignore")


def plot_actual_vs_pred(target, y_pred, predicted_r2_model, r2_final_model, predicted_mae_model, final_mae_model):
    sns.scatterplot(x=target, y=y_pred, edgecolor="black", color="white")
    plt.xlabel("Actual Dmax(mm)")
    plt.ylabel("Predicted Dmax(mm)")
    plt.ylim(-1, max(target)*1.1)
    plt.xlim(-1, max(target)*1.1)
    plt.title(f"Actual vs Predicted")
    plt.plot(range(int(max(target))), color="black", linestyle="dashed", linewidth=1)
    plt.grid()
    plt.text(x=int(min(target))+int(max(target))*0.1, y=int(max(target))*0.95,
             s=f"Test R2 score: {np.round(predicted_r2_model,2)}", fontsize=10)
    plt.text(x=int(min(target))+int(max(target))*0.1, y=int(max(target))*0.90,
             s=f"Train R2 score: {np.round(r2_final_model,2)}", fontsize=10)
    plt.text(x=int(min(target)) + int(max(target)) * 0.1, y=int(max(target)) * 0.85,
             s=f"Test MAE score: {np.round(predicted_mae_model, 2)}", fontsize=10)
    plt.text(x=int(min(target)) + int(max(target)) * 0.1, y=int(max(target)) * 0.80,
             s=f"Train MAE score: {np.round(final_mae_model, 2)}", fontsize=10)

    plt.show()


def main():
    df = pd.read_csv(r"C:\Users\39366\Desktop\Modelli Finali\Ensemble\DS1+DS2\data.csv")
    df.drop(columns=["Tsol", "Tliq", "T0", "T0 phase",
                     "Hmelt", "Smelt", "Hmix", "Sconf", "DGM Tsol", "DGM phase Tsol",
                     "DGM T0", "DGM phase T0", "r_average", "deltaR", "VEC_average", "chi",
                     "deltachi"], inplace=True)
    features_ensemble = df.drop(columns=["Dmax", "ID", "Cit.", "Ref.", "Alloy", "Fe"], axis=1)  # work on all dataset
    target_ensemble = df["Dmax"].copy()

    cv = KFold(n_splits=10, shuffle=True, random_state=18)
    xgb_model_grid = xgb.XGBRegressor(max_depth=10, n_estimators=50, min_child_weight=1, learning_rate=0.1,
                                      gamma=0.05, subsample=0.5)

    estimators = [("MLR", LinearRegression()), ("XGBoost", xgb_model_grid)]
    for estimator in estimators:
        scores = cross_val_score(estimator[1], features_ensemble, target_ensemble, scoring="r2", cv=cv)
        print(50 * "-")
        print(f" r2 score: {estimator[0]}, {np.round(np.mean(scores), 2)}")
    vr = VotingRegressor(estimators, weights=[0.5, 0.5])
    score_validate_r2 = np.round(pd.DataFrame(cross_validate(vr, features_ensemble, target_ensemble, cv=cv,
                                                             return_train_score=True)).drop(columns=["fit_time",
                                                                                                     "score_time"]), 2)
    score_validate_mae = np.round(pd.DataFrame(cross_validate(vr, features_ensemble, target_ensemble,
                                                              scoring="neg_mean_absolute_error", cv=cv,
                                                              return_train_score=True)).drop(columns=["fit_time",
                                                                                                      "score_time"]), 2)

    predicted_r2_vr = score_validate_r2["test_score"].mean()
    r2_final_vr = score_validate_r2["train_score"].mean()
    predicted_mae_vr = -score_validate_mae["test_score"].mean()
    mae_final_vr = -score_validate_mae["train_score"].mean()
    print(50*"-")
    print(f"t-R2: {round(r2_final_vr,2)}")
    print(f"predicted_r2: {np.round(predicted_r2_vr,2)}")
    print(f"p-r score: {np.round(np.sqrt(predicted_r2_vr),2)}")
    print(f"Δr2 score: {np.round((r2_final_vr-predicted_r2_vr),2)}")
    print(f"(Δr2/t-r2) score: {np.round(((r2_final_vr-predicted_r2_vr)/r2_final_vr)*100,2)} %")
    print(50*"-")
    print(f"t-MAE: {round(mae_final_vr,2)}")
    print(f"predicted_MAE: {np.round(predicted_mae_vr,2)}")
    print(50*"-")
    vr.fit(features_ensemble, target_ensemble)
    y_pred = vr.predict(features_ensemble)
    plot_actual_vs_pred(target_ensemble, y_pred, predicted_r2_vr, r2_final_vr, predicted_mae_vr, mae_final_vr)
    # SHAP_VALUES
    explainer = shap.KernelExplainer(vr.predict, features_ensemble)
    shap_values = explainer.shap_values(features_ensemble)
    shap.summary_plot(shap_values, features_ensemble, max_display=10,
                      feature_names=features_ensemble.columns, plot_type="bar", color="#ff0d57", show=False)
    plt.title("SHAP_VALUES", fontsize=15)
    plt.ylabel("Feature", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()