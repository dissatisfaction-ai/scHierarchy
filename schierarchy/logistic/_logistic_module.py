from typing import Optional

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from scvi import _CONSTANTS


class HierarchicalLogisticPyroModel(PyroModule):

    prediction = False

    def __init__(
        self,
        n_obs,
        n_vars,
        n_levels,
        n_cells_per_label_per_level,
        tree,
        laplace_prior={"mu": 0.0, "sigma": 0.5},
        init_vals: Optional[dict] = None,
    ):
        """

        Parameters
        ----------
        n_obs
        n_vars
        n_batch
        n_extra_categoricals
        laplace_prior
        """

        ############# Initialise parameters ################
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_levels = n_levels
        self.n_cells_per_label_per_level = n_cells_per_label_per_level
        self.tree = tree
        self.laplace_prior = laplace_prior

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
            "laplace_prior_mu",
            torch.tensor(self.laplace_prior["mu"]),
        )

        self.register_buffer(
            "laplace_prior_sigma",
            torch.tensor(self.laplace_prior["sigma"]),
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
            w_i = pyro.sample(
                f"weight_level_{i}",
                dist.Laplace(self.laplace_prior_mu, self.laplace_prior_sigma)
                .expand([self.n_vars, self.layers_size[i]])
                .to_event(2),
            )
            # parameter for cluster size weight normalisation w / sqrt(n_cells per cluster)
            n_cells_per_label = self.get_buffer(f"n_cells_per_label_per_level_{i}")
            if i == 0:
                # computer f for level 0 (it is independent from the previous level as it doesn't exist)
                f_i = torch.nn.functional.softmax(
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
                        torch.nn.functional.softmax(
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
