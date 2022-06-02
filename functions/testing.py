import numpy as np
import torch


def evalperf(f_true, mean, stddev, mean_reg, sqrbeta):
    # _, var = gp.predict(X, return_std=True)
    err_pred = sqrbeta * stddev
    err = torch.abs(mean - f_true)
    err_reg = torch.mean(torch.abs(mean_reg - f_true))

    # compute prediction error
    pred_err = (err - err_pred).clip(min=0)
    bound_err = pred_err > 0
    percentage = torch.sum(bound_err) / pred_err.shape[0] * 100
    avg_prederr = torch.mean(pred_err)

    return percentage, avg_prederr, err_reg