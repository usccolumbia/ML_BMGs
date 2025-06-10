import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, cross_validate, learning_curve
import xgboost as xgb
from sklearn.ensemble import VotingRegressor


def plot_actual_vs_pred(target, y_pred, predicted_r2_model, r2_final_model, predicted_mae_model, final_mae_model):
    """
        Generate a scatter plot to compare actual and predicted values with additional model metrics.
    """
    sns.set_theme(style="whitegrid") # Set seaborn theme

    # Creazione dello scatter plot con una mappa di colori
    scatter = plt.scatter(
        x=target,
        y=y_pred,
        c=np.abs(target - y_pred),  # Color by error distance
        cmap='coolwarm',  # Color map
        s=70,  # point dimension
        alpha=0.8  # Transparency
    )

    colorbar = plt.colorbar(scatter, label="Error Distance")
    colorbar.set_label(
        'Error Distance',  # Label text
        fontsize=14,  # Font size
        labelpad=10,  # Padding
        color="darkred",  # Label color
        fontweight='bold'  # Font weight
    )
    # Color bar for errors

    # Dynamic axis limits
    x_min, x_max = min(target) * 0.9, max(target) * 1.1
    plt.ylim(x_min, x_max)
    plt.xlim(x_min, x_max)
    step = 2
    x_ticks = np.arange(0.0, max(target) + step, step)
    y_ticks = np.arange(0.0, max(target) + step, step)

    plt.xticks(x_ticks, fontsize=11, fontweight='bold', color="darkblue")
    plt.yticks(y_ticks, fontsize=11, fontweight='bold', color="darkblue")

    # Plot diagonal for perfect predictions
    plt.plot([x_min, x_max], [x_min, x_max], color="blue", linestyle="dotted", linewidth=2, alpha=0.7)

    # Titles and labels
    plt.title("Actual vs Predicted",  fontsize=18, pad=15, color="purple", fontweight='bold')
    plt.xlabel("Actual Dmax (mm)", fontsize=14, labelpad=10, color="darkred", fontweight='bold')
    plt.ylabel("Predicted Dmax (mm)", fontsize=14, labelpad=10, color="darkred", fontweight='bold')

    # Add metrics as annotations
    annotation_x = x_min + (x_max - x_min) * 0.1
    annotation_y_start = x_max * 0.95
    annotation_step = x_max * 0.05

    text_style = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.text(annotation_x, annotation_y_start,
             f"Test R²: {np.round(predicted_r2_model, 2)}",
             fontsize=11, bbox=text_style)
    plt.text(annotation_x, annotation_y_start - annotation_step,
             f"Train R²: {np.round(r2_final_model, 2)}",
             fontsize=11, bbox=text_style)
    plt.text(annotation_x, annotation_y_start - 2 * annotation_step,
             f"Test MAE: {np.round(predicted_mae_model, 2)}",
             fontsize=11, bbox=text_style)
    plt.text(annotation_x, annotation_y_start - 3 * annotation_step,
             f"Train MAE: {np.round(final_mae_model, 2)}",
             fontsize=11, bbox=text_style)

    # Grid and visual adjustments
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.show()


def main():
    df = pd.read_csv(r"C:\Users\39366\Desktop\Modelli Finali\Ensemble\DS1+DS2\data.csv")
    df.drop(columns=["Fe", "Co", "B", "Si", "Nb", "Ni", "Y", "Zr", "Cr", "Mo",
                     "P", "C", "Hf", "Al", "Dy", "Er", "Mn", "Ti", "V", "Cu", "Sn", "W",
                     "Ta", "T0 phase", "DGM phase Tsol", "DGM phase T0"], inplace=True)
    features_ensemble = df.drop(columns=["Dmax", "ID","Cit.", "Ref.", "Alloy"], axis=1)  # work on all dataset
    target_ensemble = df["Dmax"].copy()
    cv = KFold(n_splits=10, shuffle=True, random_state=18)
    xgb_model_grid = xgb.XGBRegressor(max_depth=8, n_estimators=100, min_child_weight=5,
                                      learning_rate=0.1, gamma=0.05, subsample=0.5)

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
    print(f"t-R2: {round(mae_final_vr,2)}")
    print(f"predicted_r2: {np.round(predicted_mae_vr,2)}")
    print(50*"-")
    vr.fit(features_ensemble, target_ensemble)
    y_pred = vr.predict(features_ensemble)
    plot_actual_vs_pred(target_ensemble, y_pred, predicted_r2_vr, r2_final_vr, predicted_mae_vr, mae_final_vr)


if __name__ == '__main__':
    main()
