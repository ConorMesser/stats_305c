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

    def __init__(self, obs_df, pc_df, tax_df, dim=10,
                 horseshoe_u=False, horseshoe_intercept=True, non_centered=False,
                 esm_intercept=None, esm_interaction=None,
                 linreg=False, hierarchical=False, anchor_pc=False, random_effects=False,
                 sigma_obs_sigma=3, beta_sigma=2,
                 esm_active_num=15, slab_scale = 4.0):
        """
        :param obs_df: DataFrame with ['Peptide ID', 'Target Species', 'mic']
                       (Uses LONG format)
        :param pc_df: pd.DataFrame shape (N_peptides, 12)
        :param tax_df: pd.DataFrame with ['strain', 'species', 'genus',...]
        :param dim: Latent factors (K)
        """
        # save parameters to keep track of model training
        use_esm_intercept = esm_intercept is not None
        esm_intercept_shape = (0,0) if not use_esm_intercept else esm_intercept.shape
        use_esm_interaction = esm_interaction is not None
        esm_interaction_shape = (0,0) if not use_esm_interaction else esm_interaction.shape

        self.param_inputs = dict(dim=dim,
                                 horseshoe_u=horseshoe_u, horseshoe_intercept=horseshoe_intercept,
                                 non_centered=non_centered, linreg=linreg, hierarchical=hierarchical,
                                 anchor_pc=anchor_pc, random_effects=random_effects,
                                 sigma_obs_sigma=sigma_obs_sigma, beta_sigma=beta_sigma,
                                 esm_active_num=esm_active_num, slab_scale = slab_scale,
                                 obs_df_shape=obs_df.shape, pc_df_shape=pc_df.shape,
                                 esm_intercept_shape=esm_intercept_shape, esm_interaction_shape=esm_interaction_shape)

        assert not (random_effects and linreg), ('Random_effects and linreg cannot both be set to true, '
                                                 'as linear regression means no interaction.')
        assert not (use_esm_interaction and linreg), ('esm_interaction cannot be provided with linreg, '
                                                      'as linear regression means no interaction.')
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
        pc_df.index = pc_df.index.map(peptide_dict)
        self.pc_df = pc_df

        # Extract Data
        obs_peptide_idx = obs_df['peptide_idx'].values
        obs_strain_idx = obs_df['strain_idx'].values
        obs_mic = obs_df['mic'].values
        # Get the unique integer indices (0 to n_peptides-1) in order
        unique_pep_idx = np.sort(np.unique(obs_peptide_idx))

        # Slice the dataframes using the unique indices
        # And drop the original_idx column
        pc_input = pc_df.loc[unique_pep_idx].iloc[:, :-1].values

        if use_esm_intercept:
            esm_intercept['original_idx'] = esm_intercept.index
            esm_intercept.index = esm_intercept.index.map(peptide_dict)
            self.esm_intercept_df = esm_intercept
            esm_intercept_input = esm_intercept.loc[unique_pep_idx].iloc[:, :-1].values
        else:
            self.esm_intercept_df = None
            esm_intercept_input = None

        if use_esm_interaction:
            esm_interaction['original_idx'] = esm_interaction.index
            esm_interaction.index = esm_interaction.index.map(peptide_dict)
            self.esm_interaction_df = esm_interaction
            esm_interaction_input = esm_interaction.loc[unique_pep_idx].iloc[:, :-1].values
        else:
            self.esm_interaction_df = None
            esm_interaction_input = None

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
                    "esm_intercept": range(esm_intercept_shape[1]),
                    "esm_interaction": range(esm_interaction_shape[1]),
                }
        ) as pmf:
            # --- DATA CONTAINERS ---
            pep_idx_ = pm.Data("pep_idx", obs_peptide_idx, dims="obs_id")
            strain_idx_ = pm.Data("strain_idx", obs_strain_idx, dims="obs_id")
            spec_idx_ = pm.Data("spec_idx", strain_to_species_idx, dims="strain")
            genus_idx_ = pm.Data("genus_idx", species_to_genus_idx, dims="species")
            mic_ = pm.Data("mic", obs_mic, dims="obs_id").astype(np.float32)

            X_pc = pm.Data("X_pc", pc_input, dims=("peptide", "phys_chem")).astype(np.float32)
            if use_esm_intercept:
                X_esm_intercept = pm.Data("X_esm_intercept", esm_intercept_input, dims=("peptide", "esm_intercept")).astype(np.float32)
            if use_esm_interaction:
                X_esm_interaction = pm.Data("X_esm_interaction", esm_interaction_input, dims=("peptide", "esm_interaction")).astype(np.float32)

            if not linreg:
                if random_effects:
                    # Sample a separate U for each peptide with no learning from phys_chem features or ESM
                    sigma_u = pm.HalfNormal("sigma_u", 1.0)

                    if non_centered:
                        U_raw = pm.Normal("U_raw", mu=0,
                                                  sigma=1, dims=("peptide", "latent_factor"))
                        U = pm.Deterministic("U", U_raw * sigma_u,
                                                     dims=("peptide", "latent_factor"))
                    else:
                        U = pm.Normal("U", mu=0, sigma=sigma_u, dims=("peptide", "latent_factor"))
                else:

                    # --- HYBRID PEPTIDE MAPPING (u_i) ---
                    # Learn variance of Physical Features (Standard Weakly Informative Prior)
                    sigma_b_pc = pm.HalfNormal("sigma_b_pc", sigma=1.0)

                    # Anchor the physical_chemical features to prevent sign flipping instability
                    if anchor_pc and dim <= pc_input.shape[1]:
                        B_pc_raw = pm.Normal("B_pc_raw", mu=0, sigma=sigma_b_pc, dims=("phys_chem", "latent_factor"))

                        # Sample K positive anchors
                        B_pc_anchors = pm.HalfNormal("B_pc_anchors", sigma=sigma_b_pc, shape=dim)

                        # Stitch them together using PyTensor
                        # This replaces the diagonal of the top KxK block of B_pc with strictly positive values.
                        B_pc = pt.set_subtensor(
                            B_pc_raw[np.arange(dim), np.arange(dim)],
                            B_pc_anchors
                        )
                    else:
                        B_pc = pm.Normal("B_pc", mu=0, sigma=sigma_b_pc, dims=("phys_chem", "latent_factor"))

                    if use_esm_interaction:
                        if horseshoe_u:
                            B_esm = self._horseshoe(X_esm_interaction, esm_active_num, slab_scale, intercept=False)
                        else:
                            sigma_b_esm = pm.HalfNormal("sigma_b_esm", sigma=1.0)
                            B_esm = pm.Normal("B_esm", mu=0, sigma=sigma_b_esm, dims=("esm_interaction", "latent_factor"))

                        # 3. Calculate Latent Peptides (U) via Matrix Dot Product
                        # U is shape (N_peptides, K)
                        U = pm.Deterministic("U", pt.dot(X_pc, B_pc) + pt.dot(X_esm_interaction, B_esm),
                                             dims=("peptide", "latent_factor"))
                    else:
                        U = pm.Deterministic("U", pt.dot(X_pc, B_pc), dims=("peptide", "latent_factor"))

                # --- TAXONOMIC HIERARCHY (v_j) ---
                # V_strains is sampled the sample regardless of if random_effects model or not,
                # as we don't have strain features to utilize as a hybrid model
                if hierarchical:
                    # Dirichlet ensures variance is identifiable between U and V - constrains the total V variance
                    var_fracs = pm.Dirichlet("var_fracs", a=np.ones(3))

                    # 2. Convert the variance fractions to standard deviations
                    v_genus_sigma = pm.Deterministic("sig_genus", pt.sqrt(var_fracs[0]))
                    v_species_sigma = pm.Deterministic("sig_species", pt.sqrt(var_fracs[1]))
                    v_strains_sigma = pm.Deterministic("sig_strain", pt.sqrt(var_fracs[2]))

                    if non_centered:
                        V_genus_raw = pm.Normal("V_genus_raw", mu=0, sigma=1, dims=("genus", "latent_factor"))
                        V_genus = pm.Deterministic("V_genus", V_genus_raw * v_genus_sigma,
                                                   dims=("genus", "latent_factor"))
                        V_species_raw = pm.Normal("V_species_raw", mu=0,
                                                  sigma=1, dims=("species", "latent_factor"))
                        V_species = pm.Deterministic("V_species", V_genus[genus_idx_] + (V_species_raw * v_species_sigma),
                                                     dims=("species", "latent_factor"))
                    else:
                        V_genus = pm.Normal("V_genus", mu=0, sigma=v_genus_sigma,
                                            dims=("genus", "latent_factor"))
                        V_species = pm.Normal("V_species", mu=V_genus[genus_idx_],
                                              sigma=v_species_sigma, dims=("species", "latent_factor"))
                    V_mu = V_species[spec_idx_]
                else:
                    v_strains_sigma = pm.HalfNormal("v_strains_sigma", sigma=1.0)
                    V_mu = 0

                if non_centered:
                    V_strains_raw = pm.Normal("V_strains_raw", mu=0,
                                              sigma=1, dims=("strain", "latent_factor"))
                    V_strains = pm.Deterministic("V_strains", V_mu + V_strains_raw * v_strains_sigma,
                                                 dims=("strain", "latent_factor"))
                else:
                    V_strains = pm.Normal("V_strains", mu=V_mu,
                                          sigma=v_strains_sigma, dims=("strain", "latent_factor"))

                # --- Calculate Interaction Likelihood ---
                # 1. Gather the relevant parameters for the observed data points
                U_obs = U[pep_idx_]  # Shape (N_obs, K)
                V_obs = V_strains[strain_idx_]  # Shape (N_obs, K)

                # 2. Calculate interaction term using batched dot product along the K dimension
                interaction = pt.batched_dot(U_obs, V_obs)
            else:
                if anchor_pc:
                    print("Warning: anchor_pc=True has no effect in linear model (no interaction).")
                # Use pep_idx_.shape[0] to get the size of the tensor
                interaction = pt.zeros(pep_idx_.shape[0])

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
                                             sigma=beta_species_sigma, dims="species")
                beta_mu = beta_species[spec_idx_]
            else:
                beta_mu = 0

            if non_centered:
                # Sample strain intercept from a standard normal
                beta_strain_raw = pm.Normal("beta_strain_raw", mu=0, sigma=1, dims="strain")
                # Scale it deterministically
                beta_strain = pm.Deterministic("beta_strain", beta_mu + beta_strain_raw * beta_strain_sigma,
                                               dims="strain")
            elif hierarchical:
                beta_strain = pm.Normal("beta_strain", mu=beta_mu, sigma=beta_strain_sigma, dims="strain")
            else:
                beta_strain = pm.ZeroSumNormal("beta_strain", sigma=beta_strain_sigma, dims="strain")

            # Peptide Intercept (Cold-start friendly mapping)
            w0_pc_sigma = pm.HalfNormal("w0_pc_sigma", sigma=1.0)
            w0_pc = pm.Normal("w0_pc", mu=0, sigma=w0_pc_sigma, dims="phys_chem")

            if use_esm_intercept:
                if horseshoe_intercept:
                    w0_esm = self._horseshoe(X_esm_intercept, esm_active_num, slab_scale, intercept=True)
                else:
                    w0_esm_sigma = pm.HalfNormal("w0_esm_sigma", sigma=1.0)
                    w0_esm = pm.Normal("w0_esm", mu=0, sigma=w0_esm_sigma, dims="esm_intercept")
                alpha_peptide = pm.Deterministic("alpha_peptide", pt.dot(X_pc, w0_pc) + pt.dot(X_esm_intercept, w0_esm),
                                                 dims="peptide")
            else:
                alpha_peptide = pm.Deterministic("alpha_peptide", pt.dot(X_pc, w0_pc),
                                                 dims="peptide")

            # Add together the intercepts and interaction terms (if using)
            # Predicted mean for each observation based on peptide and strain indices
            mu_obs = pm.Deterministic("mu_obs", alpha_peptide[pep_idx_] + beta_strain[strain_idx_] + interaction,  # + global_mu,
                                      dims="obs_id")

            # Learn the observation noise (primarily from repeated measurements for same pairing
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=sigma_obs_sigma)

            # Final MIC Distribution
            pm.Normal("MIC_obs", mu=mu_obs, sigma=sigma_obs, observed=mic_, dims="obs_id")

        self.model = pmf
        logging.info("Done building the PMF model.")

    def _horseshoe(self, X_esm, esm_active_num, slab_scale, intercept=True):
        if intercept:
            name_suffix = "_int"
            dim = "esm_intercept"
            full_dims = dim
            output_name = "w0_esm"
        else:
            name_suffix = ""
            dim = "esm_interaction"
            full_dims = (dim, "latent_factor")
            output_name = "B_esm"

        # 1. Global Shrinkage (tau) - heavily squashed based on 1280 features
        # Expecting only ~15 relevant features out of 1280
        D = X_esm.shape[1]  # 1280
        tau_0 = esm_active_num / (D - esm_active_num)
        tau_esm = pm.HalfNormal(f"tau_esm{name_suffix}", sigma=tau_0)  # tau_esm

        # 2. Local Shrinkage (lambda) - Heavy tails to let signals escape
        lambda_esm = pm.HalfCauchy(f"lambda_esm{name_suffix}", beta=1, dims=dim)

        # 3. The Slab (c2) - This prevents the escaping signals from going to infinity
        # Regularizes the tails so NUTS doesn't crash
        c2 = pm.InverseGamma("c2", alpha=1.5, beta=1.5 * slab_scale ** 2)

        # 4. Calculate the Regularized local shrinkage
        lambda_tilde = pt.sqrt((c2 * lambda_esm ** 2) / (c2 + tau_esm ** 2 * lambda_esm ** 2))

        # 5. Non-centered raw weights
        esm_raw = pm.Normal("esm_raw", mu=0, sigma=1, dims=full_dims)

        # 6. Final Weights
        calculation = esm_raw * (tau_esm * lambda_tilde)
        if not intercept:
            calculation = calculation[:, None]
        esm = pm.Deterministic(output_name, calculation, dims=full_dims)

        return esm

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
        # TODO what to do if random_effects (for new peptides??)
        # Identify the unique peptides in this specific prediction batch
        predict_peps = new_long_df["Peptide ID"].unique()

        # Create a LOCAL dictionary mapping to 0, 1, 2... N
        local_pep_dict = {pep: i for i, pep in enumerate(predict_peps)}

        # Map the observation dataframe to these local indices
        obs_peptide_idx = new_long_df["Peptide ID"].map(local_pep_dict).values

        # Extract their features from master dataframes and set to order of the local dictionary
        # Assumes original_idx is a column containing the Peptide IDs
        pc_df_predict = self.pc_df[self.pc_df['original_idx'].isin(predict_peps)]
        pc_input = pc_df_predict.set_index('original_idx').loc[predict_peps].values

        esm_data_dict = {}
        if self.esm_intercept_df is not None:
            esm_intercept_predict = self.esm_intercept_df[self.esm_intercept_df['original_idx'].isin(predict_peps)]
            esm_data_dict["X_esm_intercept"] = esm_intercept_predict.set_index('original_idx').loc[predict_peps].values

        if self.esm_interaction_df is not None:
            esm_interaction_predict = self.esm_interaction_df[self.esm_interaction_df['original_idx'].isin(predict_peps)]
            esm_data_dict["X_esm_interaction"] = esm_interaction_predict.set_index('original_idx').loc[predict_peps].values

        # Ensure all target species exist in the training dictionary
        assert len(set(new_long_df["Target Species"]) - set(self.strain_dict.keys())) == 0, \
            "Cannot predict for a species not in the training set!"

        obs_strain_idx = new_long_df["Target Species"].map(self.strain_dict).values

        with self.model:
            idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi

            # if self.param_inputs["random_effects"]:
            #     idata.posterior = self.gen_peptides_U()

            pm.set_data(
                {
                    "pep_idx": obs_peptide_idx,
                    "strain_idx": obs_strain_idx,
                    "mic": np.ones(len(new_long_df)),  # Dummy values for shape
                    "X_pc": pc_input,
                    **esm_data_dict
                },
                coords={
                    "obs_id": range(len(new_long_df)),
                    "peptide": range(len(predict_peps)),
                },
            )

            ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=False, var_names=var_names)

        # if no var_names given, just return the final observations: MIC_obs
        var_names = ["MIC_obs"] if var_names is None else var_names
        raw = ppc.posterior_predictive[var_names]
        return raw if return_full_posterior else raw.mean(("chain", "draw"))

        # u_mean = self.idata.posterior["U"].mean(dim="peptide")
        # post = self.idata.posterior
        #
        # obs_strain_idx = new_long_df["Target Species"].map(self.strain_dict).values
        #
        # X_pc_new = xr.DataArray(pc_input.values, dims=("peptide_new", "phys_chem"))
        #
        # alpha_new = xr.dot(X_pc_new, post["w0_pc"], dims="phys_chem")
        #
        # # get features
        # obs_strain_idx = new_long_df["Target Species"].map(self.strain_dict).values
        #
        # X_esm_new = xr.DataArray(esm_input.values, dims=("peptide_new", "esm"))
        # alpha_new = alpha_new + xr.dot(X_esm_new, post["w0_esm"], dims="esm")
        #
        # # new alpha from calculation and beta from sample
        # alpha_obs = alpha_new.isel(peptide_new=obs_peptide_idx).rename({"peptide_new": "obs_id"})
        # beta_obs = post["beta_strain"].isel(strain=obs_strain_idx).rename({"strain": "obs_id"})
        #
        # # mean u, post v
        # u_mean = post["U"].mean(dim="peptide")
        # v_obs = post["V"].isel(strain=obs_strain_idx).rename({"strain": "obs_id"})
        #
        # interaction = (u_mean * v_obs).sum(dim="latent_factor")
        #
        # mu = alpha_obs + beta_obs + interaction
        #
        # # take mean of chain samples
        # return mu.mean(dim=("chain", "draw")).values

    def evaluate(self, test_df, label=""):
        # check for MCMC first
        y_pred = self.predict_new_peptides(test_df, return_full_posterior=True).mean(("chain", "draw")).MIC_obs.values
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

        train_loo_cv_vals = az.loo(self.idata).loo_i.values

        y_pred_train = self.predict_new_peptides(self.obs_df, return_full_posterior=True).mean(("chain", "draw")).MIC_obs.values
        y_true_train = self.obs_df["mic"].values
        error_train = y_true_train - y_pred_train
        rmse_train = np.sqrt(np.mean(error_train ** 2))

        train_data = self.obs_df.copy()
        train_data['elpd'] = train_loo_cv_vals
        train_data['error'] = error_train

        # test data
        y_pred = self.predict_new_peptides(test_df, return_full_posterior=True).mean(
            ("chain", "draw")).MIC_obs.values
        y_true = test_df["mic"].values
        error = y_true - y_pred
        rmse = np.sqrt(np.mean(error ** 2))

        total_test_elpd, test_point_elpd = self.get_elpd_test(test_df, var_names=["mu_obs", "sigma_obs"])

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
