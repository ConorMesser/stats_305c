import pickle

import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import os
import arviz as az
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import xarray as xr
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import logsumexp
import pymc as pm
import pytensor.tensor as pt
import numpy as np
import logging
from pymc.sampling.jax import sample_numpyro_nuts

from .utils import compute_metrics, split_random_mask, split_holdout_peptides


logging.basicConfig(level=logging.INFO)
SEED = 42

rng = np.random.default_rng(SEED)


class Hybrid_PMF:
    """Hybrid Probabilistic Matrix Factorization model for MIC prediction."""

    def __init__(self, obs_df, pc_df, esm_df, tax_df, dim=10,
                 horseshoe_u=False, horseshoe_intercept=True, include_esm=True, non_centered=False,
                 linreg=False, hierarchical=False, anchor_pc=False,
                 sigma_obs_sigma=3, pc_sigma=0.4, esm_sigma=0.05, beta_sigma=3,
                 esm_active_num=15, slab_scale = 4.0):
        """
        :param obs_df: DataFrame with ['Peptide ID', 'Target Species', 'mic']
                       (Uses LONG format)
        :param pc_df: pd.DataFrame shape (N_peptides, 12)
        :param esm_df: pd.DataFrame shape (N_peptides, 1280)
        :param tax_df: pd.DataFrame with ['strain', 'species', 'genus',...]
        :param dim: Latent factors (K)
        """
        # save parameters to keep track of model training
        self.param_inputs = dict(dim=dim,
                                 horseshoe_u=horseshoe_u, horseshoe_intercept=horseshoe_intercept, include_esm=include_esm, non_centered=non_centered,
                                 linreg=linreg, hierarchical=hierarchical, anchor_pc=anchor_pc,
                                 sigma_obs_sigma=sigma_obs_sigma, pc_sigma=pc_sigma, esm_sigma=esm_sigma, beta_sigma=beta_sigma,
                                 esm_active_num=esm_active_num, slab_scale = slab_scale,
                                 obs_df_shape=obs_df.shape, pc_df_shape=pc_df.shape, esm_df_shape=esm_df.shape)

        self.name = "Hybrid_PMF"
        self.dim = dim
        self.obs_df = obs_df

        # Clean Data
        strain_dict = {strain: idx for idx, strain in enumerate(obs_df['Target Species'].unique())}
        peptide_dict = {peptide: idx for idx, peptide in enumerate(obs_df['Peptide ID'].unique())}
        self.strain_dict = strain_dict
        self.peptide_dict = peptide_dict

        obs_df['strain_idx'] = obs_df['Target Species'].map(strain_dict)
        obs_df['peptide_idx'] = obs_df['Peptide ID'].map(peptide_dict)

        species_dict = {species: idx for idx, species in enumerate(tax_df['species'].unique())}
        genus_dict = {genus: idx for idx, genus in enumerate(tax_df['genus'].unique())}
        self.species_dict = species_dict
        self.genus_dict = genus_dict

        # Generate species_to_genus and strain_to_species from taxonomy_df
        # 1. Define all ranks in order (lowest to highest)
        ranks = ['strain', 'species', 'genus']

        # 2. Generate string-to-idx dictionaries for ALL ranks
        # We drop NaNs first so missing values aren't accidentally assigned an integer ID
        id_dicts = {'strain': strain_dict,
                    'species': species_dict,
                    'genus': genus_dict}
        # for rank in ranks:
        #     if rank in tax_df.columns:
        #         id_dicts[rank] = {name: idx for idx, name in enumerate(tax_df[rank].dropna().unique())}

        # # Extract them to explicit variables to match your naming convention
        # species_dict = id_dicts.get('species', {})
        # genus_dict = id_dicts.get('genus', {})
        # id_dicts['strain'] = strain_dict

        # 3. Create an index-only dataframe by replacing strings with their IDs
        tax_idx_df = tax_df.copy()
        for rank in ranks:
            if rank in tax_idx_df.columns and rank in id_dicts:
                # .map() is orders of magnitude faster than .replace()
                tax_idx_df[rank] = tax_idx_df[rank].map(id_dicts[rank])

        # 4. Generate the hierarchical mapping dictionaries (Child_ID -> Parent_ID)
        hierarchies = {}
        for i in range(len(ranks) - 1):
            child = ranks[i]
            parent = ranks[i + 1]

            if child in tax_idx_df.columns and parent in tax_idx_df.columns:
                # Drop duplicates and NaNs to ensure a clean 1-to-1 or Many-to-1 mapping
                mapping = tax_idx_df[[child, parent]].dropna().drop_duplicates().astype(int)

                # Convert to dictionary: {child_idx: parent_idx}
                dict_name = f"{child}_to_{parent}_idx"
                hierarchies[dict_name] = mapping.set_index(child)[parent].to_dict()

        pc_df['original_idx'] = pc_df.index
        esm_df['original_idx'] = esm_df.index
        pc_df.index = pc_df.index.map(peptide_dict)
        esm_df.index = esm_df.index.map(peptide_dict)
        self.pc_df = pc_df
        self.esm_df = esm_df

        # Extract Data
        obs_peptide_idx = obs_df['peptide_idx'].values
        obs_strain_idx = obs_df['strain_idx'].values
        obs_mic = obs_df['mic'].values
        # Get the unique integer indices (0 to n_peptides-1) in order
        unique_pep_idx = np.sort(np.unique(obs_peptide_idx))

        # Slice the dataframes using the unique indices
        # And drop the original_idx column
        pc_input = pc_df.loc[unique_pep_idx].iloc[:, :-1].values
        esm_input = esm_df.loc[unique_pep_idx].iloc[:, :-1].values

        n_peptides = len(np.unique(obs_peptide_idx))
        n_strains = len(np.unique(obs_strain_idx))
        n_species = len(species_dict)
        n_genera = len(genus_dict)
        n_obs = len(obs_mic)

        # Create an array of length n_species
        species_to_genus_idx = np.zeros(n_species, dtype=int)
        for child_idx, parent_idx in hierarchies['species_to_genus_idx'].items():
            species_to_genus_idx[child_idx] = parent_idx

        # Create an array of length n_strains
        strain_to_species_idx = np.zeros(n_strains, dtype=int)
        for child_idx, parent_idx in hierarchies['strain_to_species_idx'].items():
            strain_to_species_idx[child_idx] = parent_idx

        self.idata = None
        self.idata_vi = None

        logging.info("Building the PMF model...")

        with pm.Model(
                coords={
                    "peptide": range(n_peptides),
                    "strain": range(n_strains),
                    "species": range(n_species),
                    "genus": range(n_genera),
                    "latent_factor": range(dim),
                    "obs_id": range(n_obs),
                    "phys_chem": range(pc_input.shape[1]),
                    "esm": range(esm_input.shape[1])
                }
        ) as pmf:
            # --- DATA CONTAINERS ---
            pep_idx_ = pm.Data("pep_idx", obs_peptide_idx, dims="obs_id")
            strain_idx_ = pm.Data("strain_idx", obs_strain_idx, dims="obs_id")
            spec_idx_ = pm.Data("spec_idx", strain_to_species_idx, dims="strain")
            genus_idx_ = pm.Data("genus_idx", species_to_genus_idx, dims="species")
            mic_ = pm.Data("mic", obs_mic, dims="obs_id").astype(np.float32)

            X_pc = pm.Data("X_pc", pc_input, dims=("peptide", "phys_chem")).astype(np.float32)
            X_esm = pm.Data("X_esm", esm_input, dims=("peptide", "esm")).astype(np.float32)

            if not linreg:

                # --- HYBRID PEPTIDE MAPPING (u_i) ---
                # 1. Physical Features (Standard Weakly Informative Prior)
                # 1. Sample the unconstrained bulk of the weights
                B_pc = pm.Normal("B_pc", mu=0, sigma=pc_sigma, dims=("phys_chem", "latent_factor"))

                if anchor_pc and dim <= 12:
                    # 2. Sample K positive anchors
                    B_pc_anchors = pm.HalfNormal("B_pc_anchors", sigma=pc_sigma, shape=dim)

                    # 3. Stitch them together using PyTensor
                    # This replaces the diagonal of the top KxK block of B_pc with strictly positive values.
                    B_pc = pt.set_subtensor(
                        B_pc[np.arange(dim), np.arange(dim)],
                        B_pc_anchors
                    )

                if include_esm:
                    if horseshoe_u:
                        # # 2. ESM Features (COLUMN-WISE HORSESHOE PRIOR - Non-Centered)
                        # # Global shrinkage
                        # tau_esm = pm.HalfCauchy("tau_esm", beta=1)
                        # # Local shrinkage (One per ESM feature, applied across all K dims)
                        # lambda_esm = pm.HalfCauchy("lambda_esm", beta=1, dims="esm")

                        # 1. Global Shrinkage (tau) - heavily squashed based on 1280 features
                        # Expecting only ~15 relevant features out of 1280
                        D = X_esm.shape[1]  # 1280
                        tau_0 = esm_active_num / (D - esm_active_num)
                        tau_esm = pm.HalfNormal("tau_esm", sigma=tau_0)

                        # 2. Local Shrinkage (lambda) - Heavy tails to let signals escape
                        lambda_esm = pm.HalfCauchy("lambda_esm", beta=1, dims="esm")

                        # 3. The Slab (c2) - This prevents the escaping signals from going to infinity
                        # Regularizes the tails so NUTS doesn't crash
                        c2 = pm.InverseGamma("c2", alpha=1.5, beta=1.5 * slab_scale ** 2)

                        # 4. Calculate the Regularized local shrinkage
                        lambda_tilde = pt.sqrt((c2 * lambda_esm ** 2) / (c2 + tau_esm ** 2 * lambda_esm ** 2))

                        # 5. Non-centered raw weights
                        B_esm_raw = pm.Normal("B_esm_raw", mu=0, sigma=1, dims=("esm", "latent_factor"))

                        # 6. Final Weights
                        B_esm = pm.Deterministic(
                            "B_esm",
                            B_esm_raw * (tau_esm * lambda_tilde)[:, None],
                            dims=("esm", "latent_factor")
                        )
                    else:
                        B_esm = pm.Normal("B_esm", mu=0, sigma=esm_sigma, dims=("esm", "latent_factor"))

                    # 3. Calculate Latent Peptides (U) via Matrix Dot Product
                    # U is shape (N_peptides, K)
                    U = pm.Deterministic("U", pt.dot(X_pc, B_pc) + pt.dot(X_esm, B_esm),
                                         dims=("peptide", "latent_factor"))
                else:
                    U = pm.Deterministic("U", pt.dot(X_pc, B_pc), dims=("peptide", "latent_factor"))

                # --- TAXONOMIC HIERARCHY (v_j) ---
                v_strains_sigma = pm.HalfNormal("v_strains_sigma",
                                                sigma=beta_sigma) #, dims=("latent_factor")) # shape D or 1?
                if hierarchical:
                    v_genus_sigma = pm.HalfNormal("v_genus_sigma",
                                                    sigma=beta_sigma)
                    v_species_sigma = pm.HalfNormal("v_species_sigma",
                                                    sigma=beta_sigma)
                    V_genus = pm.Normal("V_genus", mu=0, sigma=v_genus_sigma,
                                        dims=("genus", "latent_factor"))  # Make ZeroSumNormal?
                    if non_centered:
                        V_species_raw = pm.Normal("V_species_raw", mu=0,
                                                  sigma=1, dims=("species", "latent_factor"))
                        V_species = pm.Deterministic("V_species", V_genus[genus_idx_] + (V_species_raw * v_species_sigma),
                                                     dims=("species", "latent_factor"))
                    else:
                        V_species = pm.Normal("V_species", mu=V_genus[genus_idx_],
                                              sigma=v_species_sigma, dims=("species", "latent_factor"))
                    V_mu = V_species[spec_idx_]
                else:
                    V_mu = 0

                #### hard coded sigma to avoid correlation with learned variance in U ####
                if non_centered:
                    V_strains_raw = pm.Normal("V_strains_raw", mu=0,
                                              sigma=1, dims=("strain", "latent_factor"))
                    V_strains = pm.Deterministic("V_strains", V_mu + V_strains_raw * v_strains_sigma,
                                                 dims=("strain", "latent_factor"))
                else:
                    V_strains = pm.Normal("V_strains", mu=V_mu,
                                          sigma=v_strains_sigma, dims=("strain", "latent_factor"))

            # --- INTERCEPTS ---
            ## global_mu removed to avoid centering issue (where sampler can add to mu and subtract elsewhere)
            ## required that I subtract the mean from the dataset before inputting

            # Strain Intercept (Hierarchical Intrinsic Resistance)
            beta_strain_sigma = pm.HalfNormal("beta_strain_sigma", sigma=beta_sigma)

            if hierarchical:
                beta_genus_sigma = pm.HalfNormal("beta_genus_sigma", sigma=beta_sigma)
                beta_species_sigma = pm.HalfNormal("beta_species_sigma", sigma=beta_sigma)

                beta_genus = pm.ZeroSumNormal("beta_genus", sigma=beta_genus_sigma, dims="genus")
                if non_centered:
                    beta_species_raw = pm.Normal("beta_species_raw", mu=0,
                                                 sigma=1, dims="species")
                    beta_species = pm.Deterministic("beta_species", beta_genus[genus_idx_] + (beta_species_raw * beta_species_sigma),
                                                    dims="species")
                else:
                    beta_species = pm.Normal("beta_species", mu=beta_genus[genus_idx_],
                                             sigma=1, dims="species")
                beta_mu = beta_species[spec_idx_]
            else:
                beta_mu = 0

            if non_centered:
                # 1. Sample raw from a standard normal
                beta_strain_raw = pm.Normal("beta_strain_raw", mu=0, sigma=1, dims="strain")
                # 2. Scale it deterministically
                beta_strain = pm.Deterministic("beta_strain", beta_mu + beta_strain_raw * beta_strain_sigma,
                                               dims="strain")
            elif hierarchical:
                beta_strain = pm.Normal("beta_strain", mu=beta_mu,
                                        dims="strain", sigma=beta_strain_sigma)
            else:
                beta_strain = pm.ZeroSumNormal("beta_strain",
                                               dims="strain", sigma=beta_strain_sigma)

            # Peptide Intercept (Cold-start friendly mapping)
            w0_pc = pm.Normal("w0_pc", mu=0, sigma=pc_sigma, dims="phys_chem")

            if include_esm:
                if horseshoe_intercept:
                    # # 2. ESM Features (COLUMN-WISE HORSESHOE PRIOR - Non-Centered)
                    # # Global shrinkage
                    # tau_esm = pm.HalfCauchy("tau_esm", beta=1)
                    # # Local shrinkage (One per ESM feature, applied across all K dims)
                    # lambda_esm = pm.HalfCauchy("lambda_esm", beta=1, dims="esm")

                    # 1. Global Shrinkage (tau) - heavily squashed based on 1280 features
                    # Expecting only ~15 relevant features out of 1280
                    D = X_esm.shape[1]  # 1280
                    tau_0 = esm_active_num / (D - esm_active_num)
                    tau_esm_int = pm.HalfNormal("tau_esm_int", sigma=tau_0)

                    # 2. Local Shrinkage (lambda) - Heavy tails to let signals escape
                    lambda_esm_int = pm.HalfCauchy("lambda_esm_int", beta=1, dims="esm")

                    # 3. The Slab (c2) - This prevents the escaping signals from going to infinity
                    # Regularizes the tails so NUTS doesn't crash
                    c2_int = pm.InverseGamma("c2_int", alpha=1.5, beta=1.5 * slab_scale ** 2)

                    # 4. Calculate the Regularized local shrinkage
                    lambda_tilde_int = pt.sqrt((c2_int * lambda_esm_int ** 2) / (c2_int + tau_esm_int ** 2 * lambda_esm_int ** 2))

                    # 5. Non-centered raw weights
                    w0_esm_raw = pm.Normal("w0_esm_raw", mu=0, sigma=1, dims="esm")

                    # 6. Final Weights
                    w0_esm = pm.Deterministic(
                        "w0_esm",
                        w0_esm_raw * (tau_esm_int * lambda_tilde_int),
                        dims="esm"
                    )
                else:
                    w0_esm = pm.Normal("w0_esm", mu=0, sigma=esm_sigma, dims="esm")
                alpha_peptide = pm.Deterministic("alpha_peptide", pt.dot(X_pc, w0_pc) + pt.dot(X_esm, w0_esm),
                                                 dims="peptide")
            else:
                alpha_peptide = pm.Deterministic("alpha_peptide", pt.dot(X_pc, w0_pc),
                                                 dims="peptide")

            # --- LIKELIHOOD ---
            if not linreg:
                # 1. Gather the relevant parameters for the observed data points
                U_obs = U[pep_idx_]  # Shape (N_obs, K)
                V_obs = V_strains[strain_idx_]  # Shape (N_obs, K)

                # 2. Calculate interaction term using batched dot product along the K dimension
                interaction = pt.batched_dot(U_obs, V_obs)
            else:
                # Use pep_idx_.shape[0] to get the size of the tensor
                interaction = pt.zeros(pep_idx_.shape[0])

            # 3. Build the predicted mean
            mu_obs = pm.Deterministic("mu_obs", alpha_peptide[pep_idx_] + beta_strain[strain_idx_] + interaction,  # + global_mu,
                                      dims="obs_id")

            # 4. Observation Noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=sigma_obs_sigma)

            # 5. Final Distribution
            pm.Normal("MIC_obs", mu=mu_obs, sigma=sigma_obs, observed=mic_, dims="obs_id")

        self.model = pmf
        logging.info("Done building the PMF model.")

    # Draw MCMC samples.
    def draw_samples(self, **kwargs):
        with self.model:
            if 'chains' in kwargs and kwargs['chains'] > 1 and \
                    'nuts_sampler' in kwargs and kwargs['nuts_sampler'] == 'numpyro':
                self.idata = sample_numpyro_nuts(
                    **kwargs,
                    chain_method="vectorized",  # Guaranteed to use vmap
                    keep_untransformed=False
                )
            else:
                self.idata = pm.sample(**kwargs)
        return self.idata

    def var_inference(self, **kwargs):
        with self.model:
            self.idata_vi = pm.fit(**kwargs)
        return self.idata_vi

    def predict_new_peptides(self, new_long_df, return_full_posterior=False, var_names=None):
        # 1. Identify the unique peptides in this specific prediction batch
        predict_peps = new_long_df["Peptide ID"].unique()

        # 2. Extract their features from your master dataframes
        # Assumes original_idx is a column containing the Peptide IDs
        pc_df_predict = self.pc_df[self.pc_df['original_idx'].isin(predict_peps)]
        esm_df_predict = self.esm_df[self.esm_df['original_idx'].isin(predict_peps)]

        # 3. Create a LOCAL dictionary mapping to 0, 1, 2... N
        # This prevents the PyMC IndexError trap
        local_pep_dict = {pep: i for i, pep in enumerate(predict_peps)}

        # 4. Map the observation dataframe to these local indices
        obs_peptide_idx = new_long_df["Peptide ID"].map(local_pep_dict).values

        # 5. Extract feature matrices in the exact order of the local dictionary
        # Make sure to set the index to Peptide ID temporarily so .loc alignment works perfectly
        pc_input = pc_df_predict.set_index('original_idx').loc[predict_peps]
        esm_input = esm_df_predict.set_index('original_idx').loc[predict_peps]

        # Ensure all target species exist in the training dictionary
        assert len(set(new_long_df["Target Species"]) - set(self.strain_dict.keys())) == 0, \
            "Cannot predict for a species not in the training set!"

        obs_strain_idx = new_long_df["Target Species"].map(self.strain_dict).values

        with self.model:
            pm.set_data(
                {
                    "pep_idx": obs_peptide_idx,
                    "strain_idx": obs_strain_idx,
                    "mic": np.ones(len(new_long_df)),  # Dummy values for shape
                    "X_pc": pc_input.values,
                    "X_esm": esm_input.values,
                },
                coords={
                    "obs_id": range(len(new_long_df)),
                    "peptide": range(len(predict_peps)),
                },
            )

            idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi
            ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=False, var_names=var_names)

        # if no var_names given, just return the final observations: MIC_obs
        var_names = ["MIC_obs"] if var_names is None else var_names
        raw = ppc.posterior_predictive[var_names]
        return raw if return_full_posterior else raw.mean(("chain", "draw")).values

    def evaluate(self, test_df, label=""):
        # check for MCMC first
        y_pred = self.predict_new_peptides(test_df)
        y_true = test_df["mic"].values
        return compute_metrics(y_true, y_pred, label=label)

    def get_HDI_coverage(self, test_df, hdi_prob=0.94):
        # 1. Get the FULL matrix of posterior predictions
        # Shape will be (chains, draws, n_test_observations)
        posterior_preds_xr = self.predict_new_peptides(test_df, return_full_posterior=True)

        # 2. Calculate the HDI for every single test observation
        # az.hdi returns an array of shape (n_test_observations, 2) containing [lower_bound, upper_bound]
        hdi_bounds = az.hdi(posterior_preds_xr, hdi_prob=hdi_prob)["MIC_obs"].values

        # 3. Check how many true values fall inside their respective HDIs
        true_values = test_df["mic"].values
        in_interval = (true_values >= hdi_bounds[:, 0]) & (true_values <= hdi_bounds[:, 1])

        coverage_pct = in_interval.mean() * 100

        return coverage_pct

    def get_elpd_test(self, test_df, var_names=None):

        y_true = test_df["mic"].values
        posterior_preds_xr = self.predict_new_peptides(test_df, return_full_posterior=True,
                                                  var_names=var_names)


        mu_test_samples = posterior_preds_xr["mu_obs"].stack(sample=("chain", "draw")).values.T
        sigma_samples = posterior_preds_xr["sigma_obs"].stack(sample=("chain", "draw")).values

        # 2. Calculate log likelihood of the true test data for every posterior sample
        # shape: (n_samples, n_test_obs)
        log_lik_matrix = norm.logpdf(y_true, loc=mu_test_samples, scale=sigma_samples[:, None])

        # 3. Average the probabilities (in log space) across all posterior samples for each test point
        n_samples = log_lik_matrix.shape[0]
        test_point_elpd = logsumexp(log_lik_matrix, axis=0) - np.log(n_samples)

        # 4. Sum to get the total Test ELPD (Higher/less negative is better)
        total_test_elpd = test_point_elpd.sum()
        return total_test_elpd, test_point_elpd

    def full_evaluation(self, test_df):
        """
        on the training data (save for each value)
        -run LOO-CV
        -run RMSE

        on test data (save for each)
        -calculate ELPD
        -RMSE

        Calculate HDI - 94%
        """
        # training data
        if 'log_likelihood' not in self.idata.keys() or len(self.idata.log_likelihood) == 0:
            with self.model:
                pm.compute_log_likelihood(self.idata)

        train_loo_cv_vals = az.loo(self.idata)

        y_pred_train = self.predict_new_peptides(self.obs_df)
        y_true_train = test_df["mic"].values
        error_train = y_true_train - y_pred_train
        rmse_train = np.sqrt(np.mean(error_train ** 2))

        train_data = self.obs_df.copy()
        train_data['elpd'] = train_loo_cv_vals
        train_data['error'] = error_train

        # test data
        y_pred = self.predict_new_peptides(test_df)
        y_true = test_df["mic"].values
        error = y_true - y_pred
        rmse = np.sqrt(np.mean(error ** 2))

        total_test_elpd, test_point_elpd = self.get_elpd_test(test_df)

        test_df['elpd'] = test_point_elpd
        test_df['error'] = error

        return train_data, test_df, dict(rmse_train=rmse_train, rmse_test=rmse,
                                         elpd_train=train_loo_cv_vals.sum(), elpd_test=total_test_elpd)





    def _norms(self):
        """Return norms of latent variables at each step in the
        sample trace. These can be used to monitor convergence
        of the sampler.
        """
        # check for MCMC first
        idata = self.idata if self.idata else self.idata_vi
        norms = dict()
        norms["U"] = xr.apply_ufunc(
            np.linalg.norm,
            idata.posterior["U"],
            input_core_dims=[["peptide", "latent_factor"]],
            kwargs={"ord": "fro", "axis": (-2, -1)},
        )
        norms["V"] = xr.apply_ufunc(
            np.linalg.norm,
            idata.posterior["V_strains"],
            input_core_dims=[["strain", "latent_factor"]],
            kwargs={"ord": "fro", "axis": (-2, -1)},
        )
        return xr.Dataset(norms)

    def traceplot(self):
        """Plot Frobenius norms of U and V as a function of sample #."""
        fig, axs = plt.subplots(2, 2, figsize=(12, 7))
        az.plot_trace(self._norms(), axes=axs)
        axs[0][1].set_title(label=r"$\|U\|_{Fro}^2$ at Each Sample", fontsize=10)
        axs[1][1].set_title(label=r"$\|V\|_{Fro}^2$ at Each Sample", fontsize=10)
        axs[1][1].set_xlabel("Sample Number", fontsize=10)

    def plot_intercept_traces(self, species_list=None, peptide_list=None, **kwargs):
        """
        Plot the traces for the global mean and specific intercepts.

        :param species_list: List of string 'Target Species' to plot.
        :param peptide_list: List of string 'Peptide ID' to plot.
        """
        idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi

        # Always plot the global mean
        var_names = []  # ["global_mu"]
        coords = {}

        if species_list is not None:
            var_names.append("beta_strain")
            # Convert string names to the integer indices used in the model
            coords["strains"] = [self.strain_dict[s] for s in species_list]

        if peptide_list is not None:
            var_names.append("alpha_peptide")
            coords["peptide"] = [self.peptide_dict[p] for p in peptide_list]

        var_names.append("sigma_obs")
        var_names.append("beta_sigma_species")

        # Plot using ArviZ
        az.plot_trace(idata, var_names=var_names, coords=coords, figsize=(12, 3 * len(var_names)), **kwargs)
        plt.tight_layout()

    def plot_mic_posterior(self, peptide_id, strains_id, true_mic=None, **kwargs):
        """
        Reconstruct and plot the posterior distribution of the expected log(MIC)
        for a specific peptide and strain.

        :param peptide_id: String ID of the peptide.
        :param strains_id: String ID of the target strain.
        :param true_mic: Optional float (in log space) to plot as a reference line.
        """
        idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi
        post = idata.posterior

        # Look up the internal integer indices
        try:
            p_idx = self.peptide_dict[peptide_id]
            s_idx = self.strain_dict[strains_id]
        except KeyError as e:
            raise ValueError(f"ID {e} not found in the training dictionary.")

        # 1. Extract the specific slices for this pair
        alpha = post["alpha_peptide"].sel(peptide=p_idx)
        beta = post["beta_strain"].sel(strain=s_idx)
        u_vec = post["U"].sel(peptide=p_idx)
        v_vec = post["V_strains"].sel(strain=s_idx)

        # 2. Calculate the dot product across the latent factor dimension
        # xarray automatically handles broadcasting across chains and draws
        interaction = (u_vec * v_vec).sum(dim="latent_factor")

        # 3. Reconstruct the expected log(MIC)
        mu_posterior = alpha + beta + interaction  # + post["global_mu"]

        # 4. Plot using ArviZ
        ax = az.plot_posterior(
            mu_posterior,
            ref_val=true_mic,
            point_estimate="mean",
            hdi_prob=0.95,
            **kwargs
        )
        ax.set_title(f"Expected log(MIC)\n{peptide_id} vs {strains_id}")

        # Return the raw xarray DataArray in case you want to do math with it later
        return mu_posterior, ax

    def plot_feature_weights(self, **kwargs):
        """Plot a forest plot of the physical feature weights (w0_pc)."""
        idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi

        # Get the original column names from your dataframe
        feature_names = self.pc_df.columns[:12].tolist()

        # Create the forest plot
        axes = az.plot_forest(
            idata,
            var_names=["w0_pc"],
            combined=True,
            hdi_prob=0.95,
            figsize=(8, 6),
            **kwargs
        )

        # Relabel the y-axis with the actual feature names
        axes[0].set_yticklabels(feature_names[::-1])  # Reversed because ArviZ plots bottom-to-top
        axes[0].set_title("Posterior Weights of Physical Features (95% HDI)")
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)

        return axes

    def save(self, path):
        self.idata.to_netcdf(path + '.nc')
        with open(path + '.txt', 'w') as f:
            json.dump(self.param_inputs, f, indent=4)
