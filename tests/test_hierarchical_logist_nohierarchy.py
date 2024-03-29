import os

import numpy as np

from schierarchy import LogisticModel
from schierarchy.utils.data_transformation import data_to_zero_truncated_cdf
from schierarchy.utils.simulation import hierarchical_iid


def test_hierarchical_logist_nohierarchy():
    save_path = "./cell2location_model_test"
    hlevels = [20]
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
        LogisticModel.setup_anndata(dataset, layer="cdf", level_keys=level_keys)

        # train regression model to get signatures of cell types
        sc_model = LogisticModel(
            dataset,
            laplace_learning_mode=learning_mode,
        )
        # test full data training
        sc_model.train(max_epochs=10, batch_size=None)
        # test minibatch training
        sc_model.train(max_epochs=10, batch_size=100)
        # export the estimated cell abundance (summary of the posterior distribution)
        dataset = sc_model.export_posterior(dataset, sample_kwargs={"num_samples": 10})
        dataset = sc_model.export_posterior(
            dataset, sample_kwargs={
                "use_median": True,
            },
            use_quantiles=True,
            add_to_varm=['q50'],
        )
        # test plot_QC'
        # sc_model.plot_QC()
        # test save/load
        sc_model.save(save_path, overwrite=True, save_anndata=True)
        sc_model = LogisticModel.load(save_path)
        os.system(f"rm -rf {save_path}")
