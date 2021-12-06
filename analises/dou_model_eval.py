### Functions for evaluating the performance of the DOU
### sorter by relevance.

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

def sum_lower_ranks(ranked_true):
    """
    Given a list or array `ranked_true` with values that *should*
    be sorted in ascending order, compute the number of 
    values that appear later in the array than each value that 
    are lower in value (and thus should appear early on), weighted 
    by their difference in value.
    """
    # Compute number of lower labels in front of higher labels, weighted by their label difference:
    label_differences = np.subtract.outer(ranked_true, ranked_true)
    lower_rank_counts = np.triu(np.clip(label_differences, a_min=0, a_max=None))
    total_lower_rank  = np.sum(lower_rank_counts)
    return total_lower_rank


def ranking_metric(y_true, y_pred_rank, **kwargs):
    """
    Measure how badly the true labels `y_true` are ordered by 
    `y_pred_rank` by counting how many labels are in the wrong position, 
    weighted by the difference between each pair of labels.
    """
    # Sort true labels according to the predicted score:
    pred_ranks    = np.array(y_pred_rank).argsort()
    ranked_true   = np.array(y_true)[pred_ranks]
    reversed_true = np.sort(y_true)[::-1]
    
    total_lower_rank   = sum_lower_ranks(ranked_true)
    if True:
        max_ranking_metric = sum_lower_ranks(reversed_true)
    else:
        max_ranking_metric = 1
    return total_lower_rank / max_ranking_metric


def moving_average(a, n=5):
    """
    Given a list or array `a`, compute the moving average with `n` points.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_average_rank(y_true, pred_rank, moving_average_window, color='b', label='Model', random=True):
    """
    Plot the true value of the labels (y axis) as a function of their descending order 
    (x axis) according to their ranking given by `pred_rank`.
    
    Input
    -----
    
    y_true : array
        The true ordinal labels of the examples.
        
    pred_rank : array
        The predicted rank (order) that each label in `y_true` should be.
        
    moving_average_window : int
        The amount of points to aggregate into a moving average before plotting.
        
    color : str
        The color of the line.
        
    label : str
        The label of the line.
    
    random : bool
        Whether or not to include a plot of the expected moving average value 
        of the ordinal labels in the case of random sorting.
    """
    # Sort true labels according to the predicted score:
    pred_sort   = pred_rank.argsort()[::-1]
    ranked_true = y_true[pred_sort]

    # Compute expected average label:
    frequency_y_true  = pd.Series(y_true).value_counts(normalize=True)
    mean_random_label = np.sum(frequency_y_true * frequency_y_true.index)
    # Compute variation:
    moment2_random_label = np.sum(frequency_y_true * frequency_y_true.index**2)
    dev_random_model     = np.sqrt((moment2_random_label - mean_random_label**2) / moving_average_window)

    # Random Plot:
    if random:
        pl.axhspan(mean_random_label - dev_random_model, mean_random_label + dev_random_model, color='lightgray')
        pl.axhline(mean_random_label, color='k', linewidth=1)

    # Plot:
    smoothed   = moving_average(ranked_true, moving_average_window)
    percentage = np.linspace(0,100,len(smoothed))
    pl.plot(percentage, smoothed, color=color, label=label)

    pl.tick_params(labelsize=14)
    pl.xlabel('% of examples', fontsize=14)
    pl.ylabel('Average class', fontsize=14)

    
def relevant_frac_by_order(descending_sorted, relevant_list):
    """
    Given a list of labels in an attempted descending order `descending_sorted`, 
    compute the cumulative fraction of labels that are in `relevant_list`.
    """
    cum_count = np.cumsum(np.isin(descending_sorted, relevant_list).astype(int))
    frac = cum_count / cum_count[-1]
    return frac


def plot_recall_curve(y_true, y_pred_rank):
    """
    Given a list of true label `y_true` and a predicted ranking for them, 
    `y_pred_rank`, plot the cumulative recall as a function of fraction of 
    the examples (in descending ranking).
    """
    # Place labels in predicted order and in perfect order (for comparison):
    sorted_idx    = y_pred_rank.argsort()
    sorted_train  = y_true[sorted_idx[::-1]]
    perfect_train = np.sort(y_true)[::-1] 
    
    y = relevant_frac_by_order(sorted_train, [4,5])
    y_perfect = relevant_frac_by_order(perfect_train, [4,5])

    x = np.arange(1, len(y) + 1) / len(y) * 100

    pl.plot(x, y_perfect * 100, color='r', label='Perfeito')
    pl.plot(x, y * 100, color='b', label='C/ classif. do bot')
    pl.plot(x, x, color='k', linewidth=1, linestyle='--', label='Na unha')
    pl.tick_params(labelsize=14)
    pl.xlabel('% dos atos a serem lidos', fontsize=14)
    pl.ylabel('% encontrada dos atos relevantes', fontsize=14)
    pl.title('Seção 1 do DOU', fontsize=14)
    pl.grid()
    pl.legend(loc='lower right', fontsize=12)

    
def cum_recall_metric(y_true, y_pred_rank, relevant_list=[4,5]):
    """
    This metric computes the cumulative recall curve, i.e. the 
    completeness of the `y_true` items that are in `relevant_list` 
    as we increase the number of examples read (the examples are 
    sorted according to `y_pred_rank`). Then in normalizes from 
    0 (expected random ordering) to 1 (perfect ordering).
    """
    # Place labels in predicted order and in perfect order (for comparison):
    sorted_idx    = y_pred_rank.argsort()
    sorted_train  = y_true[sorted_idx[::-1]]
    perfect_train = np.sort(y_true)[::-1] 
    
    # Compute cumulative recall:
    recall_frac  = relevant_frac_by_order(sorted_train, relevant_list)
    perfect_frac = relevant_frac_by_order(perfect_train, relevant_list)
    
    # Integrate the cumulative recall curves:
    train_area   = np.sum(recall_frac) / len(sorted_train)
    perfect_area = np.sum(perfect_frac) / len(perfect_train)
    
    # Normalize the metric:
    metric = (train_area - 0.5) / (perfect_area - 0.5)
    
    return metric
