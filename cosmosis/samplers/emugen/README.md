# EmugenSampler Documentation

EmugenSampler is an emulator-accelerated MCMC sampler that uses neural networks to speed up cosmological parameter estimation by 10-1000x.

## Quick Links

📚 **[User Guide](EMUGEN_USER_GUIDE.md)** - Start here! Complete guide on how to use EmugenSampler

⚙️ **[Configuration Guide](EMUGEN_CONFIG_GUIDE.md)** - Detailed explanation of all configuration options

📝 **[Example Configuration](example_emugen_config.ini)** - Template configuration file with detailed comments

## Quick Start

1. **Copy the example configuration:**
   ```bash
   cp cosmosis/samplers/emugen/example_emugen_config.ini my_emugen.ini
   ```

2. **Edit your pipeline ini file:**
   ```ini
   [runtime]
   sampler = emugen
   
   # Include the emugen configuration
   %include my_emugen.ini
   ```

3. **Run your analysis:**
   ```bash
   cosmosis your_pipeline.ini
   ```

## What's in this directory

- `README.md` - This file
- `EMUGEN_USER_GUIDE.md` - Complete user guide with examples and troubleshooting
- `EMUGEN_CONFIG_GUIDE.md` - Detailed reference for all configuration options  
- `example_emugen_config.ini` - Template configuration with extensive comments
- `emugen_sampler.py` - Main sampler implementation
- `cosmopower.py` - Neural network emulator backend

## Need Help?

1. **Read the User Guide** for step-by-step instructions
2. **Check the Configuration Guide** for parameter details
3. **Use the example configuration** as a starting point
4. **Look at diagnostic plots** in your output directory
5. **Post issues** with your configuration and error messages

## Key Features

- ⚡ **10-1000x speedup** for slow cosmological pipelines
- 🧠 **Neural network emulation** with iterative training
- 🔄 **Adaptive sampling** that improves the emulator over time
- 💾 **Reusable emulators** for parameter studies
- 📊 **Built-in diagnostics** to monitor emulator quality
- 🔧 **Full CosmoSIS integration** with all standard features
