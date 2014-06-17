#!/usr/bin/env Rscript

###
### Read a directory tree created by a run of cosmosis, and produce a
### standard set of plots, guided by what directories and files are
### found.
###

suppressPackageStartupMessages(library("optparse"))
suppressPackageStartupMessages(library("lattice"))
suppressPackageStartupMessages(library("latticeExtra"))
suppressPackageStartupMessages(library("tools"))
suppressPackageStartupMessages(library("reshape2"))

# These are a few functions to help make generating 'pretty' log-log
# plots a bit easier.
# xyplot.
scales.log.log <- function()
{
  list( x=list(log=10, equispaced=FALSE)
      , y=list(log=10, equispaced=FALSE)
      )
}

xscale.log <- function() xscale.components.log10ticks

yscale.log <- function() yscale.components.log10ticks

# Scan the given file in the given directory, ignoring lines that
#  start with '#'.
cosmo.scan <- function(dirname, filename)
{
  scan(file.path(dirname, filename), comment.char="#", quiet=TRUE)
}

# Create a dataframe from all the named files. It is assumed that each
# file contains a single column of data, and that all the columns are of
# the same length. Lines beginning with "#" are ignored. The names of
# the columns are determined from the names of the files, by dropping
# the file extension.
#
# If we change the output of CosmoSIS to make a single file with
# multiple columns, then read.table() would work directly and probably a
# bit more efficiently.
make.dataframe.from.files <- function(fnames)
{
  # Scan each file, creating a vector of the right name
  # bind the columns into a dataframe.
  columns <- lapply(fnames, function(n) scan(n, comment.char="#", quiet=TRUE))
  result <- data.frame(columns)
  names(result) <- lapply(fnames, function(n) file_path_sans_ext(basename(n)))
  result
}

# Create a matter power dataframe. This is handled specially, because
# the p_k matrix is stored in a linearized format, and needs to be
# interpreted correctly.
make.matter.power.dataframe.from.files <- function(dirname, type)
{
  # If the directory does not exist, return an empty dataframe.
  if (! file.exists(dirname)) return(data.frame())

  # Scan each file, creating a vector of the right name
  # bind the columns into a dataframe.

  z <- cosmo.scan(dirname, "z.txt")
  k_h <- cosmo.scan(dirname, "k_h.txt")
  p_k <- cosmo.scan(dirname, "p_k.txt")
  nkh <- length(k_h)

  # The p_k array carries the data for a matrix, with 'z' varying slowly
  # and 'k_h' varying rapidly. Thus to build the dataframe, we can rely
  # on the recycling rule to get k_h correct, but have to construct z
  # ourselves.
  dframe <- data.frame( p_k = p_k
                      , k_h = k_h
                      , z = rep(z, each=nkh)
                      )
  dframe$type = type
  dframe
}

# Create a single "distance plot". A distance plot is always a plot of
# something as a function of reshift "z'. The plot is written to a file,
# using the specified device. All additional arguments are passed to the
# device function.
make.distance.plot <- function(colname, dframe, prefix, outdir, verbose, devname, ...)
{
  device.fcn = get(devname)
  plotfile <- paste(outdir, "/", prefix, "_", colname, ".", devname, sep="")
  if (verbose) cat("Plotting", colname, "versus z into", plotfile, "\n")
  device.fcn(plotfile, ...)
  print(xyplot( as.formula(paste(colname, "~z"))
              , dframe
              , xlab="Redshift z"
              , ylab=toupper(colname)
              , grid=TRUE
              , type="l",
              , lwd = 2
              ))
  invisible(dev.off())
}

# Create a single "CMB spectrum plot". A CMB spectrum plot is always a
# plot fo something as a function of C_ell 'ell'. The plot is written to
# a file, using the specified device. All additional arguments are
# passed to the device function.
make.cmb.spectrum.plot <- function(colname, dframe, prefix, outdir, verbose, devname, ...)
{
  device.fcn <- get(devname)
  plotfile <- paste(outdir, "/", prefix, "_", colname, ".", devname, sep="")
  if (verbose) cat("Plotting", colname, "versus ell into", plotfile, "\n")
  device.fcn(plotfile, ...)
  print(xyplot( as.formula(paste(colname, "~ell"))
              , dframe
              , xlab = "ell"
              , ylab = paste("C_ell ", toupper(colname), "/uK^2", sep="")
              , grid=TRUE
              , type="l",
              , lwd = 2
              ))
  invisible(dev.off())
}

# Create the CMB spectrum plot with all spectra shown together. Because
# some of the spectra may have negative values, we take the absolute
# value (so that our log-log plot doesn't complain).
make.cmb.grand.spectrum.plot <- function(dframe, prefix, outdir, verbose, devname, ...)
{
  device.fcn <- get(devname)
  plotfile <- paste(outdir, "/", prefix, "_grand.", devname, sep="")
  if (verbose)
    cat("Plotting all spectra versus ell into", plotfile, "\n")
  device.fcn(plotfile, ...)
  print(xyplot( abs(value)~ell
              , dframe
              , xlab="ell"
              , ylab="C_ell spectra / uK^2"
              , grid=TRUE
              , type="l",
              , lwd = 2
              , group = variable
              , auto.key = list(space="right", points=FALSE, lines=TRUE)
              , scales = scales.log.log()
              , xscale.components = xscale.log()
              , yscale.components = yscale.log()
              ))
  invisible(dev.off())
}

# Make all the distance plots for files under the given top-level
# directory *topdir*. Distance plots are made for data files in the
# subdirectory "distances".
make.distance.plots <- function(topdir, verbose, prefix, outdir, devname)
{
  # data files are in "distances".
  datadir <- file.path(topdir, "distances")
  if (verbose) cat("Making distance plots from data in", datadir, "\n")

  # We don't want to read the "values.txt" file.
  excluded <- file.path(datadir, "values.txt")
  files <- Filter(function(x) {x!=excluded}, dir(datadir, full.names=TRUE))

  # Create a dataframe from all the named files.
  dframe <- make.dataframe.from.files(files)
  if (verbose)
     cat("Made the data frame for distances\n")

  # If we have a "h" column, we want to scale it. We do it here because
  # this is the function that knows about the nature of the directory
  # that it is working on.
  if (any(names(dframe) %in% "h"))
    dframe$h <- dframe$h * 2.99792458e+05

  # Plot each interesting column against "z".
  y.cols <- Filter(function(x) x!="z", names(dframe))
  for (y.col in y.cols)
    make.distance.plot(y.col, dframe, prefix, outdir, verbose, devname)
}

# Make all the CMB spectrum plots for files under the given top-level
# directory *topdir*. CMB spectrum plots are made for all data files in
# the subdirectory "cmb_cl".
make.cmb.spectrum.plots <- function(topdir, verbose, prefix, outdir, devname)
{
  # data files are in "cmb_cl"
  datadir <- file.path(topdir, "cmb_cl")
  if (verbose) cat("Making CMB spectrum plots from data in", datadir, "\n")

  files <- dir(datadir, full.names=TRUE)

  # Create a dataframe from all the named files.
  dframe <- make.dataframe.from.files(files)
  if (verbose)
    cat("Made the data frame for CMS spectrum plots\n")

  # Plot each interesting column against "C_ell"
  y.cols <- Filter(function(x) x!="ell", names(dframe))
  for (y.col in y.cols)
    make.cmb.spectrum.plot(y.col, dframe, prefix, outdir, verbose, devname)

  # Make a combined plot of all spectra versus "ell".
  molten<-melt(dframe, id="ell")
  make.cmb.grand.spectrum.plot(molten, prefix, outdir, verbose, devname)
}

make.matter.power.plots <- function(topdir, verbose, prefix, outdir, devname)
{
  # data files are in "matter_power_lin" and "matter_power_nl".
  lin_data_dir <- file.path(topdir, "matter_power_lin")
  nl_data_dir <- file.path(topdir, "matter_power_nl")

  if (verbose)
    cat("Making matter power plots from data in", lin_data_dir, "and", nl_data_dir)

  # We don't have to search what files are present, because the matter
  # power output format is special. Our handling has to rely on that.
  df.lin <- make.matter.power.dataframe.from.files(lin_data_dir, "linear")
  df.nl <- make.matter.power.dataframe.from.files(nl_data_dir, "nonlinear")

  # Combine the dataframes, creating appropriate factors.
  dframe <- rbind(df.lin, df.nl)
  dframe$type <- as.factor(dframe$type)

  plotfile <- paste(outdir, "/", prefix, "_matter_power.", devname, sep="")
  if (verbose)
    cat("Plotting matter power plot into", plotfile, "\n")
  log.log.scales=list( x=list(log=10, equispaced=FALSE)
                     , y=list(log=10, equispaced=FALSE)
                     )
  device.fcn = get(devname)
  device.fcn(plotfile)
  print( xyplot( p_k~k_h
               , subset(dframe, z==0)
               , ylab="p_k"
               , xlab="k"
               , group=type
               , type="l"
               , grid=TRUE
               , auto.key=list(space="top", lines=TRUE, points=FALSE)
               , scales = scales.log.log()
               , xscale.components = xscale.log()
               , yscale.components = yscale.log()
               ))
  invisible(dev.off())
}

################################################################################
###
### Start of the main program.

option_list <-
  list(  make_option( c("-v", "--verbose")
                     , action="store_true"
                     , default = FALSE
                     , help = "Print extra output [%default]"
                     )
       , make_option( c("-p", "--prefix")
                     , default="plot"
                     , type="character"
                     , help="Prefix for all output files"
                     )
       , make_option( c("-o", "--output")
                     , default="."
                     , type="character"
                     , help="Directory for all output files"
                     )
       , make_option( c("-d", "--device")
                    , default="png"
                    , type="character"
                    , help="Graphics device for plots: png or pdf"
                    )
       )

parser <- OptionParser(option_list=option_list, usage="%prog [options] directory")
args   <- parse_args(parser, positional_arguments = TRUE)
opt    <- args$options

if (length(args$args) != 1)
  {
    cat("Incorrect number of required arguments\n\n")
    print_help(parser)
    stop()    
  }

input.directory <- as.character(args$args)
if (opt$output != ".")
  if (! file.exists(opt$output))
    dir.create(opt$output, recursive=TRUE)

opt$device <- tolower(opt$device)

if (opt$device %in% c("png", "pdf") == FALSE)
{
  cat("Device", opt$device, "is not supported, using png\n")
  opt$device <- "png"
}

if (opt$verbose) 
   cat("Processing files in", input.directory, "\n")
                   
make.distance.plots(input.directory, opt$verbose, opt$prefix, opt$output, opt$device)
make.cmb.spectrum.plots(input.directory, opt$verbose, opt$prefix, opt$output, opt$device)
make.matter.power.plots(input.directory, opt$verbose, opt$prefix, opt$output, opt$device)
