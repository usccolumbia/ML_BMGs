import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate, learning_curve
import xgboost as xgb
import dataframe_image as dfi
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor



class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for i in self.cols:
            col_list.append(X.iloc[:, i:i+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self


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
    cv = KFold(n_splits=10, shuffle=True, random_state=18)
    df_ensemble_1 = df.copy().drop(columns=["ID",  "Ref.", "Cit.", "Alloy", "Fe", "T0 phase", "DGM phase Tsol",
                                            "DGM phase T0"], axis=1)
    ensemble_features_1 = df_ensemble_1.drop("Dmax", axis=1)
    ensemble_target_1 = df_ensemble_1["Dmax"].copy()
    pipe_xgb_0_1 = Pipeline([
        ("col_extract", ColumnExtractor(cols=range(14, 36))),
        ("xgb", xgb.XGBRegressor(max_depth=10, n_estimators=50, min_child_weight=1, learning_rate=0.1,
                                 gamma=0.05, subsample=0.5))
        ])
    pipe_xgb_1_1 = Pipeline([
        ("col_extract", ColumnExtractor(cols=range(0, 14))),
        ("xgb", xgb.XGBRegressor(max_depth=8, n_estimators=100, min_child_weight=5,
                                 learning_rate=0.1, gamma=0.05, subsample=0.5))
        ])

    xgb_model_grid = xgb.XGBRegressor(max_depth=8, n_estimators=100, min_child_weight=5,
                                      learning_rate=0.1, gamma=0.05, subsample=0.5)

    ensemble_1 = VotingRegressor(estimators=[("df_xgb0_1", pipe_xgb_0_1), ("df_xgb1_1", pipe_xgb_1_1)], weights=[0.5, 0.5])
    ensemble_1.fit(ensemble_features_1, ensemble_target_1)
    df_cross_validate_r2 = np.round(pd.DataFrame(cross_validate(ensemble_1, ensemble_features_1, ensemble_target_1, cv=cv,
                                                                return_train_score=True))
                                    .drop(columns=["fit_time", "score_time"]), 2)
    df_cross_validate_mae = np.round(pd.DataFrame(cross_validate(ensemble_1, ensemble_features_1, ensemble_target_1,
                                                                 scoring="neg_mean_absolute_error", cv=cv,
                                                                 return_train_score=True))
                                     .drop(columns=["fit_time", "score_time"]), 2)

    print(50*"-")
    print(f"{df_cross_validate_r2}")
    print(50*"-")
    print(f"{df_cross_validate_mae}")
    predicted_r2_vr = df_cross_validate_r2["test_score"].mean()
    r2_final_vr = df_cross_validate_r2["train_score"].mean()
    predicted_mae_vr = -df_cross_validate_mae["test_score"].mean()
    mae_final_vr = -df_cross_validate_mae["train_score"].mean()
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
    ensemble_1.fit(ensemble_features_1, ensemble_target_1)
    y_pred = ensemble_1.predict(ensemble_features_1)
    plot_actual_vs_pred(ensemble_target_1, y_pred, predicted_r2_vr, r2_final_vr, predicted_mae_vr, mae_final_vr)


if __name__ == '__main__':
    main()
