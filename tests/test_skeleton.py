from scvi.data import synthetic_iid
from mypackage import MyModel, MyPyroModel
import pyro


def test_mymodel():
    n_latent = 5
    adata = synthetic_iid()
    model = MyModel(adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_elbo()
    model.get_latent_representation()
    model.get_marginal_ll(n_mc_samples=5)
    model.get_reconstruction_error()
    model.history

    # tests __repr__
    print(model)


def test_mypyromodel():
    adata = synthetic_iid()
    pyro.clear_param_store()
    model = MyPyroModel(adata)
    model.train(max_epochs=1, train_size=1)
    model.get_latent(adata)
    model.history

    # tests __repr__
    print(model)
