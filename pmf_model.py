import pymc as pm
import pytensor.tensor as pt
import numpy as np
import logging
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import os
import arviz as az
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO)
SEED = 42

rng = np.random.default_rng(SEED)

def compute_metrics(y_true, y_pred, label=""):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pearson_r,  pearson_p  = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    print(f"\n{'─'*48}")
    if label:
        print(f"  {label}")
    print(f"  N        : {mask.sum()}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  Pearson  : r={pearson_r:.4f}  p={pearson_p:.3g}")
    print(f"  Spearman : r={spearman_r:.4f}  p={spearman_p:.3g}")
    print(f"{'─'*48}\n")
    return dict(n=int(mask.sum()), rmse=rmse,
                pearson_r=pearson_r,  pearson_p=pearson_p,
                spearman_r=spearman_r, spearman_p=spearman_p)

# ─────────────────────────────────────────────────────────────
# STRATEGY 1 — RANDOM MASK SPLIT
# Hides individual observations regardless of peptide.
# Tests interpolation: model has seen all peptides during training.
# ─────────────────────────────────────────────────────────────

def split_random_mask(long_df, holdout_frac=0.20, random_state=42):
    rand_num_gen  = np.random.default_rng(random_state)
    n        = len(long_df)
    test_idx = rand_num_gen.choice(n, size=int(n * holdout_frac), replace=False)
    mask     = np.zeros(n, dtype=bool)
    mask[test_idx] = True

    train_df = long_df[~mask].copy()
    test_df  = long_df[ mask].copy()

    print(f"[random mask]  train obs={len(train_df)}  test obs={len(test_df)}")
    return train_df, test_df

# ─────────────────────────────────────────────────────────────
# STRATEGY 2 — HELD-OUT PEPTIDE SPLIT
# Holds out entire peptides from training.
# Tests generalization: model must predict from features alone.
# ─────────────────────────────────────────────────────────────

def split_holdout_peptides(long_df, holdout_frac=0.20, random_state=42):
    #set holdouts
    peptides  = long_df["Peptide ID"].unique()
    rand_num_gen = np.random.default_rng(random_state)
    n_holdout = max(1, int(len(peptides) * holdout_frac))
    held_out  = set(rand_num_gen.choice(peptides, size=n_holdout, replace=False))

    train_df = long_df[~long_df["Peptide ID"].isin(held_out)].copy()
    test_df  = long_df[ long_df["Peptide ID"].isin(held_out)].copy()

    print(f"[peptide holdout]  train peptides={train_df['Peptide ID'].nunique()}"
          f"  test peptides={test_df['Peptide ID'].nunique()}  ({len(test_df)} obs)")
    return train_df, test_df

class Hybrid_PMF:
    """Hybrid Probabilistic Matrix Factorization model for MIC prediction."""

    def __init__(self, obs_df, pc_df, esm_df, species_to_genus_idx, dim=10,
                 horseshoe=True, include_esm=True, non_centered=False,
                 linreg=False):
        """
        :param obs_df: DataFrame with ['Peptide ID', 'Target Species', 'mic']
                       (Uses LONG format)
        :param pc_df: pd.DataFrame shape (N_peptides, 12)
        :param esm_df: pd.DataFrame shape (N_peptides, 1280)
        :param species_to_genus_idx: 1D array mapping species_id -> genus_id
        :param dim: Latent factors (K)
        """
        self.name = "Hybrid_PMF"
        self.dim = dim

        # Clean Data
        species_dict = {species: idx for idx, species in enumerate(obs_df['Target Species'].unique())}
        peptide_dict = {peptide: idx for idx, peptide in enumerate(obs_df['Peptide ID'].unique())}
        self.species_dict = species_dict
        self.peptide_dict = peptide_dict

        obs_df['species_idx'] = obs_df['Target Species'].map(species_dict)
        obs_df['peptide_idx'] = obs_df['Peptide ID'].map(peptide_dict)

        pc_df['original_idx'] = pc_df.index
        esm_df['original_idx'] = esm_df.index
        pc_df.index = pc_df.index.map(peptide_dict)
        esm_df.index = esm_df.index.map(peptide_dict)
        self.pc_df = pc_df
        self.esm_df = esm_df

        # Extract Data
        obs_peptide_idx = obs_df['peptide_idx'].values
        obs_species_idx = obs_df['species_idx'].values
        obs_mic = obs_df['mic'].values
        # Get the unique integer indices (0 to n_peptides-1) in order
        unique_pep_idx = np.sort(np.unique(obs_peptide_idx))

        # Slice the dataframes using the unique indices
        # And drop the original_idx column
        pc_input = pc_df.loc[unique_pep_idx].iloc[:, :-1].values
        esm_input = esm_df.loc[unique_pep_idx].iloc[:, :-1].values

        n_peptides = len(np.unique(obs_peptide_idx))
        n_species = len(np.unique(obs_species_idx))
        n_genera = len(np.unique(species_to_genus_idx))
        n_obs = len(obs_mic)

        self.idata = None
        self.idata_vi = None

        logging.info("Building the PMF model...")

        with pm.Model(
                coords={
                    "peptide": range(n_peptides),
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
            spec_idx_ = pm.Data("spec_idx", obs_species_idx, dims="obs_id")
            mic_ = pm.Data("mic", obs_mic, dims="obs_id")

            X_pc = pm.Data("X_pc", pc_input, dims=("peptide", "phys_chem"))
            X_esm = pm.Data("X_esm", esm_input, dims=("peptide", "esm"))

            if not linreg:

                # --- HYBRID PEPTIDE MAPPING (u_i) ---
                # 1. Physical Features (Standard Weakly Informative Prior)
                B_pc = pm.Normal("B_pc", mu=0, sigma=1, dims=("phys_chem", "latent_factor"))

                if include_esm:
                    if horseshoe:
                        # 2. ESM Features (COLUMN-WISE HORSESHOE PRIOR - Non-Centered)
                        # Global shrinkage
                        tau_esm = pm.HalfCauchy("tau_esm", beta=1)
                        # Local shrinkage (One per ESM feature, applied across all K dims)
                        lambda_esm = pm.HalfCauchy("lambda_esm", beta=1, dims="esm")
                        # Raw weights
                        B_esm_raw = pm.Normal("B_esm_raw", mu=0, sigma=1, dims=("esm", "latent_factor"))
                        # Decoupled matrix multiplication
                        B_esm = pm.Deterministic(
                            "B_esm",
                            B_esm_raw * (tau_esm * lambda_esm)[:, None],
                            dims=("esm", "latent_factor")
                        )
                    else:
                        B_esm = pm.Normal("B_esm", mu=0, sigma=1, dims=("esm", "latent_factor"))

                    # 3. Calculate Latent Peptides (U) via Matrix Dot Product
                    # U is shape (N_peptides, K)
                    U = pm.Deterministic("U", pt.dot(X_pc, B_pc) + pt.dot(X_esm, B_esm), dims=("peptide", "latent_factor"))
                else:
                    U = pm.Deterministic("U", pt.dot(X_pc, B_pc), dims=("peptide", "latent_factor"))

                # --- TAXONOMIC HIERARCHY (v_j) ---  TODO - add back?
                # Genus-level baseline factors
                # V_genus = pm.Normal("V_genus", mu=0, sigma=1, dims=("genus", "latent_factor"))
                # Species-level factors (Centered on their Genus)
                # sigma_species = pm.HalfNormal("sigma_species",
                #                               sigma=1) #, dims=("latent_factor")) # shape D or 1?

                # if non_centered:
                #     # --- NON-CENTERED LATENT VECTORS ---
                #     V_species_raw = pm.Normal("V_species_raw", mu=0, sigma=1, dims=("species", "latent_factor"))
                #     # Broadcasting the scale across the latent dimensions
                #     V_species = pm.Deterministic("V_species", V_species_raw * sigma_species, dims=("species", "latent_factor"))
                # else:
                #     V_species = pm.Normal(
                #         "V_species",
                #         mu=0,  # V_genus[species_to_genus_idx],
                #         sigma=sigma_species,
                #         dims=("species", "latent_factor")
                #     )

                #### hard coded sigma to avoid correlation with learned variance in U ####
                V_species = pm.Normal("V_species", mu=0, sigma=1, dims=("species", "latent_factor"))


            # --- INTERCEPTS ---
            ## global_mu removed to avoid centering issue (where sampler can add to mu and subtract elsewhere)
            ## required that I subtract the mean from the dataset before inputting
            # data_mean = np.mean(obs_mic)
            # global_mu = pm.Normal("global_mu", mu=data_mean, sigma=5, initval=data_mean)

            # Species Intercept (Hierarchical Intrinsic Resistance)  TODO - make hierarchical?
            # beta_genus = pm.Normal("beta_genus", mu=0, sigma=1, dims="genus")
            beta_sigma_species = pm.HalfNormal("beta_sigma_species", sigma=1)
            if non_centered:
                # 1. Sample raw from a standard normal
                beta_species_raw = pm.Normal("beta_species_raw", mu=0, sigma=1, dims="species")
                # 2. Scale it deterministically
                beta_species = pm.Deterministic("beta_species", beta_species_raw * beta_sigma_species, dims="species")
            else:
                beta_species = pm.ZeroSumNormal("beta_species",
                                        #mu=0,  #beta_genus[species_to_genus_idx],
                                        dims="species", sigma=beta_sigma_species)

            # Peptide Intercept (Cold-start friendly mapping)
            w0_pc = pm.Normal("w0_pc", mu=0, sigma=1, dims="phys_chem")

            if include_esm:
                w0_esm = pm.Normal("w0_esm", mu=0, sigma=0.1, dims="esm")
                alpha_peptide = pm.Deterministic("alpha_peptide", pt.dot(X_pc, w0_pc) + pt.dot(X_esm, w0_esm),
                                                dims="peptide")
            else:
                alpha_peptide = pm.Deterministic("alpha_peptide", pt.dot(X_pc, w0_pc),
                                                dims="peptide")

            # --- LIKELIHOOD ---
            if not linreg:
                # 1. Gather the relevant parameters for the observed data points
                U_obs = U[pep_idx_]  # Shape (N_obs, K)
                V_obs = V_species[spec_idx_]  # Shape (N_obs, K)

                # 2. Calculate interaction term using batched dot product along the K dimension
                interaction = (U_obs * V_obs).sum(axis=-1)
            else:
                interaction = pt.zeros(len(pep_idx_))

            # 3. Build the predicted mean
            mu_obs = alpha_peptide[pep_idx_] + beta_species[spec_idx_] + interaction #+ global_mu

            # 4. Observation Noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)

            # 5. Final Distribution (assuming output is not in log space yet)
            pm.Normal("MIC_obs", mu=mu_obs, sigma=sigma_obs, observed=mic_, dims="obs_id")

        self.model = pmf
        logging.info("Done building the PMF model.")

    # Draw MCMC samples.
    def draw_samples(self, **kwargs):
        with self.model:
            self.idata = pm.sample(**kwargs)
        return self.idata

    def var_inference(self, **kwargs):
        with self.model:
            self.idata_vi = pm.fit(**kwargs)
        return self.idata_vi

    def predict_new_peptides(self, new_long_df, return_full_posterior=False):
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
        assert len(set(new_long_df["Target Species"]) - set(self.species_dict.keys())) == 0, \
            "Cannot predict for a species not in the training set!"

        with self.model:
            pm.set_data(
                {
                    "pep_idx":  obs_peptide_idx,
                    "spec_idx": new_long_df["Target Species"].map(self.species_dict).values,
                    "mic":      np.ones(len(new_long_df)), # Dummy values for shape
                    "X_pc":     pc_input.values,
                    "X_esm":    esm_input.values,
                },
                coords={
                    "obs_id":  range(len(new_long_df)),
                    "peptide": range(len(predict_peps)),
                },
            )

            idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi
            ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=False)

        raw = ppc.posterior_predictive["MIC_obs"]
        return raw if return_full_posterior else raw.mean(("chain", "draw")).values

    def evaluate(self, test_df, label=""):
        # check for MCMC first
        y_pred = self.predict_new_peptides(test_df)
        y_true = test_df["mic"].values
        return compute_metrics(y_true, y_pred, label=label)

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
            idata.posterior["V_species"],
            input_core_dims=[["species", "latent_factor"]],
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

    def plot_intercept_traces(self, species_list=None, peptide_list=None):
        """
        Plot the traces for the global mean and specific intercepts.

        :param species_list: List of string 'Target Species' to plot.
        :param peptide_list: List of string 'Peptide ID' to plot.
        """
        idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi

        # Always plot the global mean
        var_names = [] #["global_mu"]
        coords = {}

        if species_list is not None:
            var_names.append("beta_species")
            # Convert string names to the integer indices used in the model
            coords["species"] = [self.species_dict[s] for s in species_list]

        if peptide_list is not None:
            var_names.append("alpha_peptide")
            coords["peptide"] = [self.peptide_dict[p] for p in peptide_list]

        var_names.append("sigma_obs")
        var_names.append("beta_sigma_species")

        # Plot using ArviZ
        az.plot_trace(idata, var_names=var_names, coords=coords, figsize=(12, 3 * len(var_names)))
        plt.tight_layout()
        plt.show()

    def plot_mic_posterior(self, peptide_id, species_id, true_mic=None):
        """
        Reconstruct and plot the posterior distribution of the expected log(MIC)
        for a specific peptide and species.

        :param peptide_id: String ID of the peptide.
        :param species_id: String ID of the target species.
        :param true_mic: Optional float (in log space) to plot as a reference line.
        """
        idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi
        post = idata.posterior

        # Look up the internal integer indices
        try:
            p_idx = self.peptide_dict[peptide_id]
            s_idx = self.species_dict[species_id]
        except KeyError as e:
            raise ValueError(f"ID {e} not found in the training dictionary.")

        # 1. Extract the specific slices for this pair
        alpha = post["alpha_peptide"].sel(peptide=p_idx)
        beta = post["beta_species"].sel(species=s_idx)
        u_vec = post["U"].sel(peptide=p_idx)
        v_vec = post["V_species"].sel(species=s_idx)

        # 2. Calculate the dot product across the latent factor dimension
        # xarray automatically handles broadcasting across chains and draws
        interaction = (u_vec * v_vec).sum(dim="latent_factor")

        # 3. Reconstruct the expected log(MIC)
        mu_posterior = alpha + beta + interaction # + post["global_mu"]

        # 4. Plot using ArviZ
        ax = az.plot_posterior(
            mu_posterior,
            ref_val=true_mic,
            point_estimate="mean",
            hdi_prob=0.95
        )
        ax.set_title(f"Expected log(MIC)\n{peptide_id} vs {species_id}")
        plt.show()

        # Return the raw xarray DataArray in case you want to do math with it later
        return mu_posterior

    def plot_feature_weights(self):
        """Plot a forest plot of the physical feature weights (w0_pc)."""
        idata = self.idata if hasattr(self, 'idata') and self.idata else self.idata_vi

        # Get the original column names from your dataframe
        feature_names = self.pc_df.columns[1:13].tolist()

        # Create the forest plot
        axes = az.plot_forest(
            idata,
            var_names=["w0_pc"],
            combined=True,
            hdi_prob=0.95,
            figsize=(8, 6)
        )

        # Relabel the y-axis with the actual feature names
        axes[0].set_yticklabels(feature_names[::-1]) # Reversed because ArviZ plots bottom-to-top
        axes[0].set_title("Posterior Weights of Physical Features (General Potency)")
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)
        plt.show()