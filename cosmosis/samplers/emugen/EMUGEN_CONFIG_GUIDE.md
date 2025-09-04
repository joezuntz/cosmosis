# EmugenSampler Configuration Guide

The EmugenSampler is an emulator-accelerated MCMC sampler that uses neural networks to speed up cosmological parameter estimation. This guide explains all configuration options available in the `[emugen]` section of your ini file.

## Table of Contents

1. [Overview](#overview)
2. [Basic Setup](#basic-setup)
3. [Training Configuration](#training-configuration)
4. [Emulation Settings](#emulation-settings)
5. [MCMC Settings](#mcmc-settings)
6. [Output and Saving](#output-and-saving)
7. [Advanced Options](#advanced-options)
8. [Complete Example](#complete-example)

## Overview

The EmugenSampler works by:
1. Running an initial set of exact (slow) calculations to create a training dataset
2. Training a neural network emulator on this data
3. Using the fast emulator for MCMC sampling
4. Iteratively improving the emulator with additional training data

## Basic Setup

### `keys` (string, required if not emulating full data vector)
**What it does:** Specifies which specific data vector components to emulate.
**Format:** Space-separated list of section.parameter names
**Default:** Empty (emulates full data vector)
**Example:** `keys = shear_cl.ell galaxy_cl.theory`

### `fixed_keys` (string, optional)
**What it does:** Parameters that remain constant during emulation (computed once and reused).
**Format:** Space-separated list of section.parameter names
**Default:** Empty
**Example:** `fixed_keys = distances.z distances.d_m`

### `error_keys` (string, optional)
**What it does:** Specifies error vectors for weighted training (advanced feature).
**Format:** Space-separated list of section.parameter names
**Default:** Empty
**Example:** `error_keys = shear_cl.error galaxy_cl.error`

## Training Configuration

### `iterations` (integer)
**What it does:** Number of training cycles. Each cycle improves the emulator.
**Default:** 4
**Recommended:** 3-6 for most problems
**Example:** `iterations = 5`

### `initial_size` (integer)
**What it does:** Number of exact calculations for initial training dataset.
**Default:** 9600
**Recommended:** 500-2000 for testing, 5000-20000 for production
**Example:** `initial_size = 1000`

### `resample_size` (integer)
**What it does:** Additional exact calculations added in each iteration.
**Default:** 4800
**Recommended:** 20-50% of initial_size
**Example:** `resample_size = 500`

### `batch_size` (integer)
**What it does:** Neural network training batch size (increases each iteration).
**Default:** 32
**Recommended:** 16-128 depending on dataset size
**Example:** `batch_size = 64`

### `training_iterations` (integer)
**What it does:** Number of neural network training epochs per iteration.
**Default:** 5
**Recommended:** 5-20 for most problems
**Example:** `training_iterations = 10`

### `chi2_cut_off` (float, required)
**What it does:** Removes training points with chi² above this threshold.
**Purpose:** Excludes poor fits that could confuse the emulator
**Recommended:** 1e6 for initial runs, lower for refined training
**Example:** `chi2_cut_off = 1000.0`

## Emulation Settings

### `last_emulated_module` (string, optional)
**What it does:** Specifies which pipeline module is the last one to be emulated.
**Purpose:** Allows partial pipeline emulation
**Default:** Empty (emulates up to likelihood)
**Example:** `last_emulated_module = pk_to_cl`

### `trained_before` (boolean)
**What it does:** Skip training and load pre-trained emulator.
**Default:** false
**Example:** `trained_before = true`

### `load_emu_filename` (string, required if trained_before=true)
**What it does:** Path to pre-trained emulator file.
**Default:** Empty
**Example:** `load_emu_filename = output/trained_emulator.pkl`

## MCMC Settings

### `emcee_walkers` (integer, required)
**What it does:** Number of MCMC walkers for emcee sampler.
**Constraint:** Must be less than initial_size
**Recommended:** 2-4 times the number of parameters
**Example:** `emcee_walkers = 20`

### `emcee_samples` (integer, required)
**What it does:** Number of MCMC samples per walker per iteration.
**Recommended:** 1000-10000 depending on convergence needs
**Example:** `emcee_samples = 5000`

### `emcee_burn` (float)
**What it does:** Fraction of samples to discard as burn-in.
**Default:** 0.3
**Range:** 0.0-1.0 (if <1) or integer (if ≥1 for absolute number)
**Example:** `emcee_burn = 0.2`

### `emcee_thin` (integer)
**What it does:** Keep every Nth sample (reduces autocorrelation).
**Default:** 1
**Recommended:** 1-5
**Example:** `emcee_thin = 2`

### `tempering` (float)
**What it does:** Likelihood tempering factor for early iterations.
**Purpose:** Helps exploration by flattening likelihood
**Default:** 0.05
**Range:** 0.01-0.5
**Example:** `tempering = 0.1`

### `tempering_file` (string, optional)
**What it does:** File containing custom tempering schedule.
**Format:** Text file with one tempering value per line
**Default:** Empty (uses constant tempering)
**Example:** `tempering_file = output/custom_tempering.txt`

### `seed` (integer)
**What it does:** Random seed for reproducible results.
**Default:** 0 (random seed)
**Example:** `seed = 12345`

## Output and Saving

### `save_outputs` (string)
**What it does:** Controls what gets saved during training.
**Options:**
- `""` (empty): Save nothing (not recommended)
- `"model"`: Save only final trained emulator
- `"all"`: Save everything (models, training data, diagnostics)
**Default:** Empty
**Recommended:** `"all"` for development, `"model"` for production
**Example:** `save_outputs = all`

### `save_outputs_dir` (string, required if save_outputs is set)
**What it does:** Directory where outputs are saved.
**Example:** `save_outputs_dir = output/emulator_training`

## Advanced Options

These options are typically not changed by users:

### `data_trafo` (string, internal)
**What it does:** Data transformation method for neural network training.
**Default:** "log_norm"
**Note:** Currently fixed, not user-configurable

### `n_pca` (integer, internal)
**What it does:** Number of PCA components for data compression.
**Default:** 32
**Note:** Currently fixed, not user-configurable

### `loss_function` (string, internal)
**What it does:** Neural network loss function.
**Default:** "default"
**Note:** Currently fixed, not user-configurable

## Complete Example

Here's a complete example configuration for the `[emugen]` section:

```ini
[emugen]
# Basic emulation settings
keys = 
fixed_keys = 
error_keys = 

# Training configuration
iterations = 5
initial_size = 1000
resample_size = 500
chi2_cut_off = 1e6
batch_size = 32
training_iterations = 10

# MCMC settings
emcee_walkers = 20
emcee_samples = 5000
emcee_burn = 0.3
emcee_thin = 1

# Tempering (helps with exploration)
tempering = 0.05
tempering_file = 

# Pipeline settings
last_emulated_module = 

# Loading pre-trained emulator
trained_before = false
load_emu_filename = 

# Output settings
save_outputs = all
save_outputs_dir = output/emulator_results

# Random seed for reproducibility
seed = 12345
```

## Tips for Effective Use

1. **Start Small:** Begin with small `initial_size` and `iterations` for testing
2. **Monitor Training:** Use `save_outputs = all` to check emulator accuracy
3. **Check Convergence:** Ensure MCMC chains converge within each iteration
4. **Adjust Tempering:** Lower tempering if chains get stuck in local minima
5. **Scale Up Gradually:** Increase training size once you're confident in the setup

## Common Issues and Solutions

- **Memory errors:** Reduce `initial_size` or `batch_size`
- **Poor emulator accuracy:** Increase `training_iterations` or `initial_size`
- **Slow convergence:** Adjust `tempering` or increase `emcee_samples`
- **Walkers getting stuck:** Ensure `emcee_walkers < initial_size` and increase `tempering`
