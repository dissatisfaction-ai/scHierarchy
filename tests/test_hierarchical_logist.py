import os

import numpy as np
from scvi.data import synthetic_iid

from schierarchy import LogisticModel
from schierarchy.utils.data_transformation import data_to_zero_truncated_cdf
from schierarchy.utils.simulation import hierarchical_iid


def test_hierarchical_logist():
    save_path = "./cell2location_model_test"
    hlevels = [4, 10, 20]
    dataset = hierarchical_iid(hlevels)
    level_keys = [f"level_{i}" for i in range(len(hlevels))]
    # tree = dataset.uns["tree"]
    del dataset.uns["tree"]
    dataset.layers["cdf"] = np.apply_along_axis(
        data_to_zero_truncated_cdf, 0, dataset.X
    )

    for learning_mode in [
        "fixed-sigma",
        "learn-sigma-single",
        "learn-sigma-gene",
        "learn-sigma-celltype",
        "learn-sigma-gene-celltype",
    ]:
        for use_dropout in [
            {"use_dropout": True},
            {"use_dropout": False},
            {"use_gene_dropout": True},
            {"use_gene_dropout": False},
        ]:
            LogisticModel.setup_anndata(dataset, layer="cdf", level_keys=level_keys)
            # train regression model to get signatures of cell types
            sc_model = LogisticModel(
                dataset,
                laplace_learning_mode=learning_mode,
                **use_dropout,
            )
            # test full data training
            sc_model.train(max_epochs=10, batch_size=None)
            # test minibatch training
            sc_model.train(max_epochs=10, batch_size=100)
            # export the estimated cell abundance (summary of the posterior distribution)
            dataset = sc_model.export_posterior(
                dataset, sample_kwargs={"num_samples": 10}
            )
            # test plot_QC'
            # sc_model.plot_QC()
            # test save/load
            sc_model.save(save_path, overwrite=True, save_anndata=True)
            sc_model = LogisticModel.load(save_path)
            os.system(f"rm -rf {save_path}")


def test_hierarchical_logist_prediction():
    hlevels = [4, 10, 20]
    dataset = hierarchical_iid(hlevels)
    level_keys = [f"level_{i}" for i in range(len(hlevels))]
    del dataset.uns["tree"]
    dataset.layers["cdf"] = np.apply_along_axis(
        data_to_zero_truncated_cdf, 0, dataset.X
    )

    LogisticModel.setup_anndata(dataset, layer="cdf", level_keys=level_keys)

    # train regression model to get signatures of cell types
    sc_model = LogisticModel(dataset)
    # test full data training
    sc_model.train(max_epochs=10, batch_size=None)
    # test prediction
    dataset2 = synthetic_iid(n_labels=5)
    dataset2.layers["cdf"] = np.apply_along_axis(
        data_to_zero_truncated_cdf, 0, dataset2.X
    )
    dataset2 = sc_model.export_posterior(
        dataset2, prediction=True, sample_kwargs={"num_samples": 10}
    )
