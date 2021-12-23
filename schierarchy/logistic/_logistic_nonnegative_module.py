from typing import Optional

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from scvi import _CONSTANTS


def normalise_by_sum(x, dim=1):
    shape = list(x.shape)
    shape[dim] = 1
    return x / x.sum(axis=dim).reshape(shape)


class HierarchicalLogisticPyroModel(PyroModule):

    prediction = False

    def __init__(
        self,
        n_obs,
        n_vars,
        n_levels,
        n_cells_per_label_per_level,
        tree,
        weights_prior={"alpha": 0.1, "beta": 1, "alpha_hierarchical": 0.3},
        learning_mode="fixed-sigma",
        init_vals: Optional[dict] = None,
        use_softmax: bool = False,
    ):
        """

        Parameters
        ----------
        n_obs
        n_vars
        n_batch
        n_extra_categoricals
        laplace_prior
        learning_mode = 'fixed-sigma', 'learn-sigma-single', 'learn-sigma-gene', 'learn-sigma-celltype',
                                'learn-sigma-gene-celltype', 'learn-sigma-gene-hierarchical'
        """

        ############# Initialise parameters ################
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_levels = n_levels
        self.n_cells_per_label_per_level = n_cells_per_label_per_level
        self.tree = tree
        self.weights_prior = weights_prior
        self.learning_mode = learning_mode
        self.use_softmax = use_softmax

        if self.learning_mode not in [
            "fixed-sigma",
            "learn-sigma-single",
            "learn-sigma-gene",
            "learn-sigma-celltype",
            "learn-sigma-gene-celltype",
            "learn-sigma-gene-hierarchical",
        ]:
            raise NotImplementedError

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))

        for i in range(self.n_levels):
            self.register_buffer(
                f"n_cells_per_label_per_level_{i}",
                torch.tensor(n_cells_per_label_per_level[i]),
            )

        self.register_buffer(
            "weights_prior_alpha",
            torch.tensor(self.weights_prior["alpha"]),
        )
        self.register_buffer(
            "weights_prior_beta",
            torch.tensor(self.weights_prior["beta"]),
        )
        self.register_buffer(
            "weights_prior_alpha_hierarchical",
            torch.tensor(self.weights_prior["alpha_hierarchical"]),
        )

        self.register_buffer("ones", torch.ones((1, 1)))

    @property
    def layers_size(self):
        if self.tree is not None:
            if len(self.tree) > 1:
                return [len(x) for x in self.tree] + [
                    len(
                        [item for sublist in self.tree[-1].values() for item in sublist]
                    )
                ]
            else:
                return [len(self.tree[0])]
        else:
            return None

    @property
    def normalise_by_sum(self):
        if self.use_softmax:
            return torch.nn.functional.softmax
        else:
            return normalise_by_sum

    @property
    def _get_fn_args_from_batch(self):
        if self.prediction:
            return self._get_fn_args_from_batch_prediction
        else:
            return self._get_fn_args_from_batch_training

    @staticmethod
    def _get_fn_args_from_batch_training(tensor_dict):
        x_data = tensor_dict[_CONSTANTS.X_KEY]
        idx = tensor_dict["ind_x"].long().squeeze()
        levels = tensor_dict[_CONSTANTS.CAT_COVS_KEY]
        return (x_data, idx, levels), {}

    @staticmethod
    def _get_fn_args_from_batch_prediction(tensor_dict):
        x_data = tensor_dict[_CONSTANTS.X_KEY]
        idx = tensor_dict["ind_x"].long().squeeze()
        return (x_data, idx, idx), {}

    ############# Define the model ################

    def create_plates(self, x_data, idx, levels):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-1, subsample=idx)

    def list_obs_plate_vars(self):
        """Create a dictionary with the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable"""

        return {
            "name": "obs_plate",
            "input": [],  # expression data + (optional) batch index
            "input_transform": [],  # how to transform input data before passing to NN
            "sites": {},
        }

    def forward(self, x_data, idx, levels):
        obs_plate = self.create_plates(x_data, idx, levels)

        f = []
        for i in range(self.n_levels):
            # create weights for level i
            if self.learning_mode == "fixed-sigma":
                w_i = pyro.sample(
                    f"weight_level_{i}",
                    dist.Gamma(self.weights_prior_alpha, self.weights_prior_beta)
                    .expand([self.n_vars, self.layers_size[i]])
                    .to_event(2),
                )
            elif self.learning_mode == "learn-sigma-single":
                sigma_i = pyro.sample(
                    f"sigma_level_{i}",
                    dist.Exponential(self.weights_prior_beta).expand([1, 1]),
                )
                w_i = pyro.sample(
                    f"weight_level_{i}",
                    dist.Gamma(self.weights_prior_alpha, self.ones / sigma_i)
                    .expand([self.n_vars, self.layers_size[i]])
                    .to_event(2),
                )
            elif self.learning_mode == "learn-sigma-gene":
                sigma_ig = pyro.sample(
                    f"sigma_ig_level_{i}",
                    dist.Exponential(self.weights_prior_beta)
                    .expand([self.n_vars])
                    .to_event(1),
                )
                w_i = pyro.sample(
                    f"weight_level_{i}",
                    dist.Gamma(self.weights_prior_alpha, self.ones / sigma_ig[:, None])
                    .expand([self.n_vars, self.layers_size[i]])
                    .to_event(2),
                )
            elif self.learning_mode == "learn-sigma-gene-hierarchical":
                if i == 0:
                    sigma_ig = pyro.sample(
                        "sigma_ig",
                        dist.Exponential(self.weights_prior_beta)
                        .expand([self.n_vars])
                        .to_event(1),
                    )
                    sigma_ig_level = pyro.sample(
                        "sigma_ig_level",
                        dist.Gamma(
                            self.weights_prior_alpha_hierarchical,
                            self.ones / sigma_ig[:, None],
                        )
                        .expand([self.n_vars, self.n_levels])
                        .to_event(2),
                    )
                w_i = pyro.sample(
                    f"weight_level_{i}",
                    dist.Gamma(
                        self.weights_prior_alpha,
                        self.ones / sigma_ig_level[:, i][:, None],
                    )
                    .expand([self.n_vars, self.layers_size[i]])
                    .to_event(2),
                )
            elif self.learning_mode == "learn-sigma-celltype":
                sigma_ic = pyro.sample(
                    f"sigma_ic_level_{i}",
                    dist.Exponential(self.weights_prior_beta)
                    .expand([self.layers_size[i]])
                    .to_event(1),
                )
                w_i = pyro.sample(
                    f"weight_level_{i}",
                    dist.Gamma(self.weights_prior_alpha, self.ones / sigma_ic[None, :])
                    .expand([self.n_vars, self.layers_size[i]])
                    .to_event(2),
                )
            elif self.learning_mode == "learn-sigma-gene-celltype":
                sigma_ig = pyro.sample(
                    f"sigma_ig_level_{i}",
                    dist.Exponential(self.weights_prior_beta)
                    .expand([self.n_vars])
                    .to_event(1),
                )

                sigma_ic = pyro.sample(
                    f"sigma_ic_level_{i}",
                    dist.Exponential(self.weights_prior_beta)
                    .expand([self.layers_size[i]])
                    .to_event(1),
                )
                w_i = pyro.sample(
                    f"weight_level_{i}",
                    dist.Gamma(
                        self.weights_prior_alpha,
                        self.ones / (sigma_ig[:, None] @ sigma_ic[None, :]),
                    ).to_event(2),
                )
            # parameter for cluster size weight normalisation w / sqrt(n_cells per cluster)
            n_cells_per_label = self.get_buffer(f"n_cells_per_label_per_level_{i}")
            if i == 0:
                # compute f for level 0 (independent)
                f_i = normalise_by_sum(
                    torch.matmul(x_data, w_i / n_cells_per_label ** 0.5), dim=1
                )
            else:
                # initiate f for level > 0
                f_i = torch.ones((x_data.shape[0], self.layers_size[i])).to(
                    x_data.device
                )
                # compute f as f_(i) * f_(i-1) for each cluster group under the parent node
                # multiplication could handle non-tree structures (multiple parents for one child cluster)
                for parent, children in self.tree[i - 1].items():
                    f_i[:, children] *= (
                        normalise_by_sum(
                            torch.matmul(
                                x_data,
                                w_i[:, children]
                                / (n_cells_per_label[children])[None, :] ** 0.5,
                            ),
                            dim=1,
                        )
                        * f[i - 1][:, parent, None]
                    )
            # record level i probabilities as level i+1 depends on them
            f.append(f_i)
            with obs_plate:
                pyro.deterministic(f"label_prob_{i}", f_i.T)
                if not self.prediction:
                    pyro.sample(
                        f"likelihood_{i}", dist.Categorical(f_i), obs=levels[:, i]
                    )
