from typing import Optional

import pandas as pd
from anndata import AnnData
from cell2location.models import Cell2location
from cell2location.models.base._pyro_base_loc_module import Cell2locationBaseModule
from pyro import clear_param_store
from pyro.nn import PyroModule


class HierarchicalCell2location(Cell2location):
    """
    Cell2location model. User-end model class. See Module class for description of the model (incl. math).

    Parameters
    ----------
    adata
        spatial AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    cell_state_df
        pd.DataFrame with reference expression signatures for each gene (rows) in each cell type/population (columns).
    use_gpu
        Use the GPU?
    **model_kwargs
        Keyword args for :class:`~cell2location.models.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel`

    Examples
    --------
    TODO add example
    >>>
    """

    def __init__(
        self,
        adata: AnnData,
        cell_state_df: pd.DataFrame,
        tree: dict,
        n_levels: int,
        model_class: Optional[PyroModule] = None,
        detection_mean_per_sample: bool = False,
        detection_mean_correction: float = 1.0,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        super().__init__(
            adata=adata,
            cell_state_df=cell_state_df,
            model_class=model_class,
            detection_mean_per_sample=detection_mean_per_sample,
            detection_mean_correction=detection_mean_correction,
            **model_kwargs,
        )

        self.tree_ = tree
        self.n_levels_ = n_levels
        # model_kwargs["tree"] = tree
        # model_kwargs["n_levels"] = self.n_levels_

        self.module = Cell2locationBaseModule(
            model=model_class,
            n_obs=self.summary_stats["n_cells"],
            n_vars=self.summary_stats["n_vars"],
            n_factors=self.n_factors_,
            n_batch=self.summary_stats["n_batch"],
            cell_state_mat=self.cell_state_df_.values.astype("float32"),
            tree=self.tree_,
            n_levels=self.n_levels_,
            **model_kwargs,
        )
        self._model_summary_string = f'cell2location model with the following params: \nn_factors: {self.n_factors_} \nn_batch: {self.summary_stats["n_batch"]} '
        self.init_params_ = self._get_init_params(locals())
