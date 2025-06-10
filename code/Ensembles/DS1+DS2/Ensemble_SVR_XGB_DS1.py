import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate, cross_val_score, learning_curve
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import dataframe_image as dfi


class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, x):
        col_list = []
        for i in self.cols:
            col_list.append(x.iloc[:, i:i+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, x, y=None):
        return self


def plot_actual_vs_pred(target, y_pred, predicted_r2_model, r2_final_model, predicted_mae_model, final_mae_model):
    # Tema di Seaborn per uno stile moderno
    sns.set_theme(style="whitegrid")

    # Creazione dello scatter plot con una mappa di colori
    scatter = plt.scatter(
        x=target,
        y=y_pred,
        c=np.abs(target - y_pred),  # Colore in base alla distanza dalla diagonale
        cmap='coolwarm',  # Mappa di colori accattivante
        s=70,  # Dimensione dei punti
        alpha=0.8  # Trasparenza
    )

    colorbar = plt.colorbar(scatter, label="Error Distance")
    colorbar.set_label(
        'Error Distance',  # Label text
        fontsize=14,  # Font size
        labelpad=10,  # Padding
        color="darkred",  # Label color
        fontweight='bold'  # Font weight
    )
    # Barra di colore per gli errori

    # Limiti dinamici degli assi
    x_min, x_max = min(target) * 0.9, max(target) * 1.1
    plt.ylim(x_min, x_max)
    plt.xlim(x_min, x_max)
    step = 2
    x_ticks = np.arange(0.0, max(target) + step, step)
    y_ticks = np.arange(0.0, max(target) + step, step)

    plt.xticks(x_ticks, fontsize=11, fontweight='bold', color="darkblue")
    plt.yticks(y_ticks, fontsize=11, fontweight='bold', color="darkblue")

    # Linea perfetta con stile moderno
    plt.plot([x_min, x_max], [x_min, x_max], color="blue", linestyle="dotted", linewidth=2, alpha=0.7)

    # Titoli e etichette
    plt.title("Actual vs Predicted",  fontsize=18, pad=15, color="purple", fontweight='bold')
    plt.xlabel("Actual Dmax (mm)", fontsize=14, labelpad=10, color="darkred", fontweight='bold')
    plt.ylabel("Predicted Dmax (mm)", fontsize=14, labelpad=10, color="darkred", fontweight='bold')

    # Annotazioni con uno stile personalizzato
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

    # Griglia e aspetto generale
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Mostra il grafico
    plt.show()


def main():
    df = pd.read_csv(r"C:\Users\39366\Desktop\Modelli Finali\Ensemble\DS1+DS2\data.csv")
    np.set_printoptions(edgeitems=200, linewidth=500)
    df.drop(columns=["Tsol", "Tliq", "T0", "T0 phase", "Hmelt", "Smelt", "Hmix", "Sconf", "DGM Tsol", "DGM phase Tsol",
                     "DGM T0", "DGM phase T0", "r_average", "deltaR", "VEC_average", "chi", "deltachi", "ID", "Ref.",
                     "Alloy", "Cit.", "Fe"], inplace=True)
    features_ensemble = df.drop('Dmax', axis=1)  # work on all dataset
    target_ensemble = df['Dmax'].copy()
    cv = KFold(n_splits=10, shuffle=True, random_state=18)
    svr_grid_final = make_pipeline(StandardScaler(),
                                   SVR(C=10, epsilon=0.1, gamma=0.1))
    xgb_model_grid = xgb.XGBRegressor(max_depth=10, n_estimators=50, min_child_weight=1, learning_rate=0.1,
                                      gamma=0.05, subsample=0.5)
    print(50*"-")
    estimators = [('SVR', svr_grid_final), ('XGBoost', xgb_model_grid)]
    for estimator in estimators:
        scores = cross_val_score(estimator[1], features_ensemble, target_ensemble, scoring="r2", cv=cv)
        print(50 * "-")
        print(f"r2_score: {estimator[0]}, {np.round(np.mean(scores), 2)}")

    # cross val score of voting regressor
    vr = VotingRegressor(estimators, weights=[0.5, 0.5])
    score_validate_r2 = np.round(pd.DataFrame(cross_validate(vr, features_ensemble, target_ensemble, cv=cv,
                                                             return_train_score=True))
                                 .drop(columns=["fit_time", "score_time"]), 2)
    score_validate_mae = np.round(pd.DataFrame(cross_validate(vr, features_ensemble, target_ensemble,
                                                              scoring="neg_mean_absolute_error", cv=cv,
                                                              return_train_score=True)).drop(columns=["fit_time",
                                                                                                      "score_time"]), 2)
    print(50*"-")
    print(f"{score_validate_r2}")
    print(50*"-")
    print(f"{score_validate_mae}")
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


if __name__ == '__main__':
    main()
