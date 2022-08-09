from scvi.data import synthetic_iid

from schierarchy import RegressionModel


def test_regression():
    save_path = "./cell2location_model_test"
    dataset = synthetic_iid(n_labels=5)
    RegressionModel.setup_anndata(dataset, labels_key="labels", batch_key="batch")

    # train regression model to get signatures of cell types
    sc_model = RegressionModel(dataset)
    # test full data training
    sc_model.train(max_epochs=10, batch_size=None)
    # test minibatch training
    sc_model.train(max_epochs=10, batch_size=100)
    # export the estimated cell abundance (summary of the posterior distribution)
    dataset = sc_model.export_posterior(dataset, sample_kwargs={"num_samples": 10})
    # test plot_QC'
    sc_model.plot_QC()
    # test save/load
    sc_model.save(save_path, overwrite=True, save_anndata=True)
    sc_model = RegressionModel.load(save_path)
    # export estimated expression in each cluster
    if "means_per_cluster_mu_fg" in dataset.varm.keys():
        inf_aver = dataset.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in dataset.uns["mod"]["factor_names"]]
        ].copy()
    else:
        inf_aver = dataset.var[
            [f"means_per_cluster_mu_fg_{i}" for i in dataset.uns["mod"]["factor_names"]]
        ].copy()
    inf_aver.columns = dataset.uns["mod"]["factor_names"]
