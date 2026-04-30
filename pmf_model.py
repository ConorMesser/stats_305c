import pymc as pm
import pytensor.tensor as pt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
SEED = 42

rng = np.random.default_rng(SEED)


class Hybrid_PMF:
    """Hybrid Probabilistic Matrix Factorization model for MIC prediction."""

    def __init__(self, obs_df, pc_df, esm_df, species_to_genus_idx, dim=10, horseshoe=True, include_esm=True):
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
        peptide_dict = {v: k for k, v in obs_df['Peptide ID'].drop_duplicates().reset_index()['Peptide ID'].to_dict().items()}
        self.species_dict = species_dict
        self.peptide_dict = peptide_dict

        obs_df['species_idx'] = obs_df['Target Species'].map(species_dict)
        obs_df['peptide_idx'] = obs_df['Peptide ID'].map(peptide_dict)
        pc_df.index = pc_df.index.map(peptide_dict)
        esm_df.index = esm_df.index.map(peptide_dict)

        # Extract Data
        obs_peptide_idx = obs_df['peptide_idx'].values
        obs_species_idx = obs_df['species_idx'].values
        obs_mic = obs_df['mic'].values
        pc_input = pc_df.loc[obs_peptide_idx].values
        esm_input = esm_df.loc[obs_peptide_idx].values

        n_peptides = len(np.unique(obs_peptide_idx))
        n_species = len(np.unique(obs_species_idx))
        n_genera = len(np.unique(species_to_genus_idx))
        n_obs = len(obs_mic)

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

            # --- TAXONOMIC HIERARCHY (v_j) ---  TODO - add more?
            # Genus-level baseline factors
            # V_genus = pm.Normal("V_genus", mu=0, sigma=1, dims=("genus", "latent_factor"))
            # Species-level factors (Centered on their Genus)
            V_species = pm.Normal(
                "V_species",
                mu=0,  # V_genus[species_to_genus_idx],
                sigma=1,
                dims=("species", "latent_factor")
            )

            # --- INTERCEPTS ---
            global_mu = pm.Normal("global_mu", mu=0, sigma=5)

            # Species Intercept (Hierarchical Intrinsic Resistance)  TODO - make hierarchical?
            # beta_genus = pm.Normal("beta_genus", mu=0, sigma=1, dims="genus")
            beta_species = pm.Normal("beta_species",
                                     mu=0,  #beta_genus[species_to_genus_idx],
                                     sigma=1, dims="species")

            # Peptide Intercept (Cold-start friendly mapping)
            w0_pc = pm.Normal("w0_pc", mu=0, sigma=1, dims="phys_chem")

            if include_esm:
                # Can add horseshoe for w0_esm here if desired, keeping simple for now
                w0_esm = pm.Normal("w0_esm", mu=0, sigma=0.1, dims="esm")
                alpha_peptide = pm.Deterministic("alpha_peptide", pt.dot(X_pc, w0_pc) + pt.dot(X_esm, w0_esm),
                                                dims="peptide")
            else:
                alpha_peptide = pm.Deterministic("alpha_peptide", pt.dot(X_pc, w0_pc),
                                                dims="peptide")

            # --- LIKELIHOOD ---
            # 1. Gather the relevant parameters for the observed data points
            U_obs = U[pep_idx_]  # Shape (N_obs, K)
            V_obs = V_species[spec_idx_]  # Shape (N_obs, K)

            # 2. Calculate interaction term using batched dot product along the K dimension
            interaction = (U_obs * V_obs).sum(axis=-1)

            # 3. Build the predicted mean
            mu_obs = global_mu + alpha_peptide[pep_idx_] + beta_species[spec_idx_] + interaction

            # 4. Observation Noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)

            # 5. Final Distribution (assuming output is not in log space yet)
            pm.Lognormal("MIC_obs", mu=mu_obs, sigma=sigma_obs, observed=mic_, dims="obs_id")

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

# def generate_test_train(obs_df, random):
#     # We want to be able to do totally random masking
#     # But also delete all the observations for a single peptide for one species - can it impute the obs?