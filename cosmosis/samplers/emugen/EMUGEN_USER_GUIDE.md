# EmugenSampler User Guide

## What is EmugenSampler?

EmugenSampler is an **emulator-accelerated MCMC sampler** that dramatically speeds up cosmological parameter estimation by replacing slow theoretical calculations with fast neural network predictions. Instead of running your full cosmological pipeline thousands of times, it:

1. **Trains** a neural network on a smaller set of exact calculations
2. **Uses** the fast emulator for MCMC sampling  
3. **Improves** the emulator iteratively with new training data

This can speed up parameter estimation by **10-1000x** depending on your pipeline complexity.

## When Should You Use EmugenSampler?

✅ **Good for:**
- Slow cosmological pipelines (>1 second per evaluation)
- Parameter estimation with many samples needed
- Parameter studies or forecast analyses
- Pipelines with smooth parameter dependencies

❌ **Not recommended for:**
- Very fast pipelines (<0.1 seconds per evaluation)
- Highly non-linear or discontinuous theory predictions
- Single quick runs (overhead not worth it)
- Pipelines with strong parameter degeneracies that are hard to learn

## Quick Start Guide

### 1. Prepare Your Pipeline

Make sure your standard CosmoSIS pipeline works correctly:

```bash
# Test your pipeline first
cosmosis your_pipeline.ini
```

### 2. Set Up EmugenSampler

Change your sampler to `emugen` in your ini file:

```ini
[runtime]
sampler = emugen

[emugen]
# Basic required settings
iterations = 3
initial_size = 500
resample_size = 250
chi2_cut_off = 1e6
emcee_walkers = 16
emcee_samples = 2000
save_outputs = all
save_outputs_dir = output/my_emulator_run
```

### 3. Run Your First Test

Start with small numbers for testing:

```bash
cosmosis your_pipeline.ini
```

The sampler will:
1. Generate 500 initial training samples (this takes time!)
2. Train a neural network emulator
3. Run MCMC using the emulator (this is fast!)
4. Repeat 2 more times with additional training data

### 4. Check Results

Look in your `save_outputs_dir` for:
- **Training diagnostics**: Loss curves, emulator accuracy plots
- **MCMC chains**: Standard CosmoSIS output files
- **Trained models**: For reuse in future runs

## Detailed Workflow

### Phase 1: Initial Training (Slow)
```
[Iteration 1]
├── Generate initial_size exact calculations (SLOW)
├── Train neural network emulator
├── Run MCMC with emulator (FAST)
└── Save results
```

### Phase 2: Iterative Improvement (Faster)
```
[Iterations 2-N]
├── Select resample_size points from MCMC chain
├── Run exact calculations on selected points (SLOWER)
├── Retrain emulator with expanded dataset
├── Run MCMC with improved emulator (FAST)
└── Save results
```

### Phase 3: Final Sampling (Fastest)
```
[Final Iteration]
├── Run final MCMC without tempering
├── Generate publication-ready chains
└── Save final emulator for reuse
```

## Configuration Strategies

### For Testing and Development
```ini
[emugen]
iterations = 2
initial_size = 100
resample_size = 50
emcee_samples = 1000
save_outputs = all
tempering = 0.1
```

### For Production Runs
```ini
[emugen]
iterations = 5
initial_size = 5000
resample_size = 1000
emcee_samples = 10000
save_outputs = model
tempering = 0.05
```

### For Reusing Trained Emulators
```ini
[emugen]
trained_before = true
load_emu_filename = output/previous_run/emumodel_5.pkl
emcee_samples = 20000
save_outputs = model
```

## Understanding the Output

### During Training
You'll see output like:
```
Generating initial sample
sample = parameters:  [[0.31 0.85] [0.29 0.82] ...]
500 initial samples had chi^2 < cut-off (1000000.0)
Training emulator from 6 parameters -> 1200 data vector points on 500 points.
```

### During MCMC
```
Running emcee with tempering (0.05) - iteration 1
100%|████████| 2000/2000 [00:45<00:00, 44.2it/s]
```

### Key Files Created
- `emumodel_N.pkl`: Trained emulator from iteration N
- `total_training_set.npz`: All training data used
- `total_testing_set.npz`: Validation data for diagnostics
- Standard CosmoSIS chain files

## Monitoring Emulator Quality

### 1. Check Training Loss
Look for files like `training_loss_iteration_N.png` in your output directory. The loss should decrease and plateau.

### 2. Validate Predictions
Compare emulator predictions against exact calculations on test data. Look for:
- Low relative errors (<1% for most applications)
- No systematic biases
- Good coverage of parameter space

### 3. Check MCMC Convergence
Use standard MCMC diagnostics:
```bash
# Check chain convergence
cosmosis-postprocess output/my_emulator_run/
```

## Common Issues and Solutions

### Issue: "Number of walkers must be smaller than the initial training set!"
**Solution:** Reduce `emcee_walkers` or increase `initial_size`
```ini
emcee_walkers = 10  # Was too large
initial_size = 200  # Or increase this
```

### Issue: Poor emulator accuracy
**Symptoms:** Large scatter in emulator vs exact comparisons
**Solutions:**
- Increase `initial_size` and `training_iterations`
- Check if your theory predictions are smooth
- Reduce `chi2_cut_off` to exclude outliers

### Issue: MCMC chains not converging
**Symptoms:** Walkers stuck, poor mixing
**Solutions:**
- Increase `tempering` (makes likelihood flatter)
- Increase `emcee_burn` and `emcee_samples`
- Check parameter priors are reasonable

### Issue: Memory errors
**Solutions:**
- Reduce `initial_size` and `batch_size`
- Use fewer `emcee_walkers`
- Run on machine with more RAM

### Issue: Very slow initial training
**This is normal!** The first iteration is slow because it runs your full pipeline many times. Subsequent iterations are much faster.

## Advanced Usage

### Partial Pipeline Emulation
Emulate only part of your pipeline:
```ini
[emugen]
last_emulated_module = pk_to_cl
keys = galaxy_cl.theory
```

### Custom Tempering Schedule
Create a file `tempering.txt`:
```
0.1
0.3
0.7
1.0
1.0
```

Then use:
```ini
tempering_file = tempering.txt
```

### Parameter Studies
Train once, then reuse for different data:
```bash
# Train emulator
cosmosis pipeline_train.ini

# Reuse for different analyses
cosmosis pipeline_data1.ini  # with trained_before = true
cosmosis pipeline_data2.ini  # with trained_before = true
```

## Performance Tips

### 1. Start Small, Scale Up
- Begin with small `initial_size` for testing
- Gradually increase for production runs
- Monitor emulator accuracy at each step

### 2. Use Tempering Wisely
- Higher tempering (closer to 1) for simple parameter spaces
- Lower tempering (0.01-0.1) for complex, multimodal spaces
- Gradually increase tempering across iterations

### 3. Optimize Training Size
- Rule of thumb: `initial_size` ≈ 100-500 × number of parameters
- More training data = better emulator but longer setup time
- Balance accuracy needs vs computational budget

### 4. Reuse Emulators
- Save trained emulators for future runs
- Useful for forecasting studies with same theory but different data
- Can significantly reduce overall computational cost

## Integration with CosmoSIS

EmugenSampler works with all standard CosmoSIS features:

### Output Formats
```ini
[output]
filename = chains/my_analysis.txt
format = text  # or cosmosis, fits
```

### Parallel Processing
```ini
[runtime]
sampler = emugen

[emugen]
# Will automatically use available cores for training
```

### Post-Processing
```bash
# Standard post-processing works
cosmosis-postprocess output/my_emulator_run/
getdist-plot output/my_emulator_run/
```

## Troubleshooting Checklist

Before asking for help, check:

- [ ] Does your standard pipeline work without emugen?
- [ ] Are your parameter priors reasonable?
- [ ] Is `emcee_walkers < initial_size`?
- [ ] Have you checked the training diagnostic plots?
- [ ] Are you using appropriate values for your problem size?

## Getting Help

1. **Check the diagnostic plots** in your `save_outputs_dir`
2. **Read the detailed configuration guide**: `EMUGEN_CONFIG_GUIDE.md`
3. **Try the example configuration**: `example_emugen_config.ini`
4. **Post issues** with your configuration and error messages

Remember: EmugenSampler is a powerful tool, but it requires some tuning for optimal performance. Start simple, monitor the diagnostics, and gradually optimize for your specific problem!
