from scvi.data import synthetic_iid
import numpy as np
import matplotlib.pyplot as plt


def random_tree(hlevels):
    """
    Generate random tree from the list of level widths and return as a dict of edges

    Parameters
    ----------
    hlevels
        List of number of classes per level

    Returns
    -------
    List of edges (parent:[children]) between levels


    Examples
    --------
    >>> hlevels = [4,19,20]
    >>> tree = random_tree(hlevels)
    """

    edge_dicts = []
    for i_level in range(len(hlevels) - 1):
        level1 = np.arange(hlevels[i_level])
        level2 = np.arange(hlevels[i_level + 1])

        edge_dict = {k: [] for k in level1}
        garant = np.random.choice(level2, size=len(level1), replace=False)  # select garanteed
        for i in range(len(level1)):
            edge_dict[i].append(garant[i])
        rest = np.random.choice(level1, size=len(level2))  # distribute the rest
        for i in range(len(level2)):
            if not i in garant:
                edge_dict[rest[i]].append(i)

        edge_dicts.append(edge_dict)
    return edge_dicts


def invert_dict(d):
    inv_d = {}
    for k, v_list in d.items():
        for v in v_list:
            inv_d[v] = k
    return inv_d


def plot_tree(hlevels, tree):
    """
    Graph plotting

    Parameters
    ----------
    hlevels
        List of number of classes per level
    tree
        List of edges between levels
    """
    ylim = np.max(hlevels[-1])
    xlim = len(hlevels)
    x_offset = [(ylim - hlevels[i]) / 2 for i in range(xlim)]
    x_ploints = []
    y_ploints = []

    x_lines = [[], []]
    y_lines = [[], []]

    for i in range(xlim):
        for j in range(hlevels[i]):
            y_ploints.append(i)
            x_ploints.append(j + x_offset[i])

    for i in range(xlim - 1):
        for k, v in tree[i].items():
            for v_i in v:
                x_lines[0].append(k + x_offset[i])
                x_lines[1].append(v_i + x_offset[i + 1])

                y_lines[0].append(i)
                y_lines[1].append(i + 1)

    plt.scatter(x_ploints, y_ploints)
    plt.plot(x_lines, y_lines, color='black', alpha=0.5)
    plt.gca().invert_yaxis()


def hierarchical_iid(hlevels, *args, **kwargs):
    """
    Wrapper above scvi.data.synthetic_iid to produce hierarchical labels

    Parameters
    ----------
    hlevels
        List of number of classes per level

    Returns
    -------
    AnnData with batch info (``.obs['batch']``), label info (``.obs['labels']``)
    on level i (``.obs['level_i']``). List of edges (parent:[children]) between levels (``.uns['tree']``)
    """
    tree = random_tree(hlevels)

    bottom_level_n_labels = hlevels[-1]
    synthetic_data = synthetic_iid(n_labels=bottom_level_n_labels, *args, **kwargs)

    levels = ['_scvi_labels'] + [f'level_{i}' for i in range(len(hlevels) - 2, -1, -1)]
    for i in range(len(levels) - 1):
        level_up = synthetic_data.obs[levels[i]].apply(lambda x: invert_dict(tree[len(tree) - 1 - i])[x])
        synthetic_data.obs[levels[i + 1]] = level_up
    synthetic_data.uns['tree'] = tree
    return synthetic_data
