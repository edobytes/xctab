import typer
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing_extensions import Annotated
from sklearn.metrics import confusion_matrix, classification_report
from utils import logger
from config import LOGS_DIRECTORY, OUTPUT_DIRECTORY


app = typer.Typer()


def cmatrix_plot(target, predictions, title='Confusion Matrix', figsize=(8,6), dpi=300):
    """
    Plot the confusion matrix.

    Args:
        target (np.array): ground truth
        predictions (np.array): predictions
        title (str, optional): title of the plot. Defaults to 'Confusion Matrix'.
        figsize (tuple, optional): _description_. Defaults to (8, 6).
        dpi (int, optional): _description_. Defaults to 300.

    Returns:
        fig (Matplotlib.pyplot.Figure): Figure from matplotlib
        ax (Matplotlib.pyplot.Axe): Axe object from matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    conf_matrix = confusion_matrix(target, predictions)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
    #fig = sns_plot.figure
    plt.title(title)
    return fig, ax


def creport_plot(target, predictions, title='Classification Report', figsize=(8, 6), dpi=300):
    """
    Plot the classification report.

    Args:
        target (np.array): ground truth
        predictions (np.array): predictions
        title (str, optional): title of the plot. Defaults to 'Classification Report'.
        figsize (tuple, optional): _description_. Defaults to (8, 6).
        dpi (int, optional): _description_. Defaults to 300.

    Returns:
       fig (Matplotlib.pyplot.Figure): Figure from matplotlib
       ax (Matplotlib.pyplot.Axe): Axe object from matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)      
    report_dict = classification_report(target, predictions, target_names=["0","1"], output_dict=True)
    keys = [key for key in report_dict.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(report_dict, columns=keys).T
    df.sort_values(by=['support'], inplace=True) 
    
    # plot the heatmap by masking the 'support' column
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True
 
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="Blues", fmt='.3g',
            vmin=0.0,
            vmax=1.0,
            linewidths=2, linecolor='white'
                    )
    # add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True    
    
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="Blues", cbar=False,
            linewidths=2, linecolor='white', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),         
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum())
                    )            
    plt.title(title)
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 360)         
    return fig, ax


@app.command()
def main(wine_type: Annotated[str, typer.Option("--wine-type", "-wt")],
        model_name: Annotated[str, typer.Option("--model-name", "-mn")],
        cmatrix: Annotated[bool, typer.Option("--cmatrix", "-cm")] = False, 
        creport: Annotated[bool, typer.Option("--creport", "-cr")] = False   
        ):
    """
    Plotting function for inference results.

    Args:
        wine_type (str): wine type (red or white)
        model_name (str): one of the implemented models
        cmatrix (bool): plot the confusion matrix
        creport (bool): plot the classification report
    """
    assert wine_type in ['red', 'white'], "Invalid wine type"
    assert model_name in ['autoencoder', 'ecod', 'knn', 'iforest'], "Invalid model name"
 
    output_directory = OUTPUT_DIRECTORY
    file_path = output_directory.joinpath(f"{model_name}-{wine_type}-inference-results.csv")
    assert file_path.exists(), "File {} not found".format(file_path)

    logs_directory = LOGS_DIRECTORY
    log = logger(logs_directory / "plot.log")
    log.info(f"Loading the test results for the {model_name} model trained on {wine_type} wine")

    df = pd.read_csv(file_path)
    target = df['quality']
    predictions = df['prediction']

    if cmatrix:
        cmplot_path = output_directory.joinpath(f"{model_name}-{wine_type}-inference-cmatrix.png")
        cmplot, _ = cmatrix_plot(target, predictions,
                title="Confusion Matrix ({})".format(model_name),
                figsize=(8, 6), 
                dpi=300
            ) 
        cmplot.savefig(cmplot_path)   
        log.info(f"Exporting to : {cmplot_path}")

    if creport:
        crplot_path = output_directory.joinpath(f"{model_name}-{wine_type}-inference-creport.png")
        crplot, _ = creport_plot(target, predictions, 
                title="Classification Report ({})".format(model_name),
                figsize=(8, 6), 
                dpi=300
            )
        crplot.savefig(crplot_path)
        log.info(f"Exporting to : {crplot_path}")  

if __name__ == "__main__":
    app()