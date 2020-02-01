import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import io
import PIL
from torchvision.transforms import ToTensor

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

eps = 1e-5


def plot_rows(
    target_y_rows: pd.DataFrame,
    context_y_rows: pd.DataFrame,
    pred_y: np.array,
    std: np.array,
    undo_log=False,
    legend=True,
):
    """Plots the predicted mean and variance and the context points.
  
  Args: 
    target_y_rows
    context_y_rows: dataframe with datetime index, and labels
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
      """
    if undo_log:
        target_y_rows = np.exp(target_y_rows) - eps
        context_y_rows = np.exp(context_y_rows) - eps

    # Plot everything
    j = 0
    label = "energy(kWh/hh)"

    # Start with true data and use it to get ylimits (that way they are constant)
    plt.plot(target_y_rows.index, target_y_rows.values, "k:", linewidth=2, label="true")
    ylims = plt.ylim()

    # plot predictions
    plt.plot(target_y_rows.index, pred_y[0], "b", linewidth=2, label="predicted")
    plt.fill_between(
        target_y_rows.index,
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.25,
        facecolor="blue",
        interpolate=True,
        label="uncertainty",
    )

    # Finally context, we do this with pandas so it will override x ax and make it nice
    context_y_rows[label].plot(
        style="ko", linewidth=2, label="input data", ax=plt.gca()
    )

    # Make the plot pretty
    plt.grid("off")
    plt.ylim(*ylims)
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh/hh)")
    plt.grid(b=None)
    if legend:
        plt.legend()


def plot_from_loader(
    loader, model, i=0, undo_log=False, title="", plot=True, legend=False
):
    device = next(model.parameters()).device
    data = loader.collate_fn([loader.dataset[i]], sample=False)
    data = [d.to(device) for d in data]
    context_x, context_y, target_x, target_y = data

    x_rows, y_rows = loader.dataset.get_rows(i)
    max_num_context = context_x.shape[1]
    y_context_rows = y_rows[:max_num_context]
    dt = y_context_rows.index[0]

    model.eval()
    with torch.no_grad():
        y_pred, kl, loss_test, y_std = model(context_x, context_y, target_x, target_y)

        if plot:
            plt.figure()
            plt.title(title + f" loss={loss_test: 2.2g} {dt}")
            plot_rows(
                y_rows,
                y_context_rows,
                y_pred.detach().cpu().numpy(),
                y_std.detach().cpu().numpy(),
                undo_log=False,
                legend=legend,
            )
    return loss_test


def plot_from_loader_to_tensor(
    *args, **kwargs
):
    plot_from_loader(*args, **kwargs)
    
    # Send fig to tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)#.unsqueeze(0)
    return image
