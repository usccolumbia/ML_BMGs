import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, learning_curve, train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import dataframe_image as dfi
import shap
import warnings
warnings.filterwarnings("ignore")


# Function to plot actual vs. predicted values
def plot_actual_vs_pred(target, y_pred, predicted_r2_model, r2_final_model, predicted_mae_model, final_mae_model):
    """
        Generate a scatter plot to compare actual and predicted values with additional model metrics.
    """
    sns.set_theme(style="whitegrid")  # Set seaborn theme

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

    # Load dataset
    df = pd.read_csv(r"C:\Users\39366\Desktop\Modelli Finali\DS1\DS1.csv")

    # Drop unnecessary columns that are not relevant for the analysis
    df.drop(columns=["ID", "Ref.", "Cit.", "Alloy", "Fe"], inplace=True)
    # Split the data into features and target
    features = df.drop("Dmax", axis=1)
    target = df["Dmax"].copy()

    cv = KFold(n_splits=10, shuffle=True, random_state=18)

    features_train, features_test, targets_train, targets_test = train_test_split(features, target,
                                                                                  test_size=0.2, shuffle=True,
                                                                                  random_state=81)

    optimal_hyperparameters = {
        "C": 10,
        "gamma": 0.1,
        "epsilon": 0.1,
        "kernel": "rbf"
        }

    # Train the optimized SVR model on the full dataset and visualize predictions
    svr_grid_final = make_pipeline(StandardScaler(),
                                   SVR(**optimal_hyperparameters))
    score_grid_r2 = np.round(pd.DataFrame(cross_validate(svr_grid_final, features, target, cv=cv,
                                                         return_train_score=True))
                               .drop(columns=["fit_time", "score_time"]), 2)
    score_grid_mae = np.round(pd.DataFrame(cross_validate(svr_grid_final, features, target,
                                                          scoring="neg_mean_absolute_error", cv=cv,
                                                          return_train_score=True))
                              .drop(columns=["fit_time", "score_time"]), 2)
    # dfi.export(score_grid_r2, "score_grid_svm_rbf_r2(0).png")
    # dfi.export(score_grid_mae, "score_grid_svm_rbf_mae(0).png")
    predicted_r2_grid_1 = score_grid_r2["test_score"].mean()
    r2_final_grid_1 = score_grid_r2["train_score"].mean()
    predicted_mae_grid_1 = -(score_grid_mae["test_score"].mean())
    mae_final_grid_1 = -(score_grid_mae["train_score"].mean())
    print(50*"-")
    print(f"t-R2: {np.round(r2_final_grid_1, 2)}")
    print(f"predicted_r2: {np.round(predicted_r2_grid_1, 2)}")
    print(f"p-r score: {np.round(np.sqrt(predicted_r2_grid_1), 2)}")
    print(f"Δr2 score: {np.round(r2_final_grid_1-predicted_r2_grid_1, 2)}")
    print(f"(Δr2/t-r2) score: {np.round(((r2_final_grid_1-predicted_r2_grid_1)/r2_final_grid_1)*100, 2)} %")
    print(50*"-")
    print(f"t-MAE: {np.round(mae_final_grid_1,2)}")
    print(f"predicted_MAE: {np.round(predicted_mae_grid_1,2)}")
    print(50*"-")
    svr_grid_final.fit(features_train, targets_train)
    y_pred = svr_grid_final.predict(features)
    plot_actual_vs_pred(np.array(target).flatten(), np.array(y_pred).flatten(), predicted_r2_grid_1,
                        r2_final_grid_1, predicted_mae_grid_1, mae_final_grid_1)

    # SHAP analysis for feature impact
    features_train_shap = svr_grid_final[0].fit_transform(features_train)
    explainer = shap.KernelExplainer(svr_grid_final[1].predict, features_train_shap)
    features_test_shap = svr_grid_final[0].fit_transform(features_test)
    shap_values = explainer.shap_values(features_test_shap)
    shap.summary_plot(shap_values, features_test_shap, max_display=10,
                      feature_names=features.columns, plot_type="bar", color="#ff0d57", show=False)
    plt.xticks(fontsize=11, fontweight='bold', color="darkblue")
    plt.yticks(fontsize=11, fontweight='bold', color="darkblue")
    plt.title("SHAP Values: Feature Impact",  fontsize=18, pad=15, color="purple", fontweight='bold')
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=14, labelpad=10,
               color="darkred", fontweight='bold')
    plt.ylabel("Feature", fontsize=14, labelpad=10, color="darkred", fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()