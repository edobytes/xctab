import typer
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pickle import load
from pathlib import Path
from typing_extensions import Annotated
from sklearn.metrics import confusion_matrix, classification_report
from utils import logger, loader, recoder
from config import DATA_DIRECTORY, LOGS_DIRECTORY, MODELS_DIRECTORY, \
    OUTPUT_DIRECTORY


app = typer.Typer()


def prediction_df(predictions, scores):
    """ 
    Format results into a pandas dataframe.

    Args:
        predictions (np.array): predictions
        scores (np.array): scores

    Returns:
        output_df (pd.DataFrame): results
    """
    output_df =  pd.DataFrame({'prediction': predictions,
                            'score': scores
                            })
    return output_df


def report_df(target, predictions):
    """ 
    Format report into a pandas dataframe.

    Args:
        target (np.array): ground truth
        predictions (np.array): predictions

    Returns:
        output_df (pd.DataFrame): results
    """
    labels = [0, 1]
    report_dict = classification_report(target, predictions, output_dict=True)
    output_df = pd.DataFrame(report_dict).round(2).transpose()
    output_df.insert(loc=0, column='class', value=labels + ["accuracy", "macro avg", "weighted avg"])
    return output_df


def cmatrix_plot(target, predictions):
    """_summary_

    Args:
        target (np.array): ground truth
        predictions (np.array): predictions

    Returns:
        fig (plt.fig): plot
    """
    conf_matrix = confusion_matrix(target, predictions)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns_plot = sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
    fig = sns_plot.figure
    return fig


@app.command()
def main(wine_type: Annotated[str, typer.Option("--wine-type", "-wt")],
        model_name: Annotated[str, typer.Option("--model-name", "-mn")],
        test_mode:  Annotated[str, typer.Option("--test-mode", "-tm")],
        export: Annotated[bool, typer.Option("--export", "-ex")] = False    
        ):
    """
    Testing function for inference and prediction.

    Args:
        wine_type (str): _description_
        model_name (str): _description_
        test_mode (str): _description_
        export (bool): _description_
    """
    assert wine_type in ['red', 'white'], "Invalid wine type"
    assert model_name in ['autoencoder', 'ecod', 'knn', 'iforest'], "Invalid model name"
    assert test_mode in ['inference', 'prediction'], "Invalid test mode"

    data_directory = DATA_DIRECTORY
    logs_directory = LOGS_DIRECTORY
    models_directory = MODELS_DIRECTORY
    output_directory = OUTPUT_DIRECTORY

    log = logger(logs_directory / "test.log")
    log.info(f"Loading variables from configuration")

    log.info(f"Loading the {model_name} model trained on {wine_type} wine")
    model_path = models_directory.joinpath(f"{model_name}-{wine_type}.pkl")
    assert model_path, f"Model not found."
    with open(model_path, "rb") as f:
        trained_model = load(f)

    if test_mode == 'prediction':
        user_input = typer.prompt("Type full path to sample(s) .csv file")
        file_path = Path(user_input)
        assert file_path.exists(), "File not found"
        log.info(f"Inference on the wine samples in {file_path}")
        samples_df = pd.read_csv(file_path)
        if 'quality' in samples_df.columns:
            samples_df = samples_df.drop("quality", axis=1)
        X = samples_df.to_numpy()
          
        y_hat = trained_model.predict(X)
        y_hat = recoder(y_hat)
        y_score = trained_model.decision_function(X)

        # feature variables + prediction results
        pred_df = prediction_df(y_hat, y_score)
        copy_df = samples_df.copy()
        results_df = pd.concat([copy_df, pred_df], axis=1)

        if export:
            results_path = output_directory.joinpath(f"{model_name}-{wine_type}-prediction-results.csv")
            results_df.to_csv(results_path, index=False)
            log.info(f"Exporting to : {results_path}")
        else:
            print(pred_df.to_string())      
  
    if test_mode == 'inference':
        if wine_type == 'red':
            test_set = data_directory / "red_test.csv"
        else:
            test_set = data_directory / "white_test.csv"

        log.info(f"Inference on the {wine_type} wine test set")
        log.info(f"Separating features from response variable")
        samples_df = pd.read_csv(test_set)
        X, y, cols = loader(test_set)
        log.info(f"X: {X.shape}")
        log.info(f"y: {y.shape}")
        log.info(f"columns: {cols}")

        y_hat = trained_model.predict(X)
        y_hat = recoder(y_hat)
        y_score = trained_model.decision_function(X)

        # feature variables + prediction results
        pred_df = prediction_df(y_hat, y_score)
        copy_df = samples_df.copy()
        results_df = pd.concat([copy_df, pred_df], axis=1)

        # statistics/evaluation report
        rep_df = report_df(y, y_hat)

        if export:
            results_path = output_directory.joinpath(f"{model_name}-{wine_type}-inference-results.csv")
            results_df.to_csv(results_path, index=False)
            log.info(f"Exporting to : {results_path}") 

            report_path = output_directory.joinpath(f"{model_name}-{wine_type}-inference-report.csv")
            rep_df.to_csv(report_path, index=False)
            log.info(f"Exporting to : {report_path}")   

            cmplot_path = output_directory.joinpath(f"{model_name}-{wine_type}-inference-cmatrix.png")
            cmplot = cmatrix_plot(y, y_hat) 
            cmplot.savefig(cmplot_path)   
            log.info(f"Exporting to : {cmplot_path}")
        else:
            print("Prediction results:")
            print(pred_df.to_string())
            print("*"*30)
            print("Report:")
            print(rep_df.to_string())  

if __name__ == "__main__":
    app()