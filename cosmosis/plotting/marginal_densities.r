#!/usr/bin/env Rscript

### Read a CosmoSIS standard output file, and generate a series of 1-
### and 2-dimensional posterior probability density plots.

suppressPackageStartupMessages(library("optparse"))
suppressPackageStartupMessages(library("lattice"))
suppressPackageStartupMessages(library("MASS"))    # for kde2d

### Create a dataframe from CosmoSIS output. We expect the first line of
### the output to contain the names of the parameters, separated by
### spaces, and with section names separated from parameter names by a
### double-hyphen.

make.data.frame <- function(fname, burn)
{
  d <- read.table(fname)
  # Remove the first 'burn' elements
  d <- d[-c(1:burn),]
  first <- readLines(fname, n=1)
  first <- sub("#", "", first)          # Remove comment
  parts <- strsplit(first, "\t")[[1]]   # split on tabs
  cols <- sub("[a-zA-Z_]+--", "", parts) # remove leading section names
  names(d) <- cols
  likes <- exp(d$like)
  norm <- sum(likes)
  d$l <- likes/norm
  return(d)
}

# Make a 1-d posterior density plot for each variable in the dataframe.
# We use bw="nrd", which gives a bandwidth calculation according to:
#    Scott, D. W. (1992) Multivariate Density Estimation: Theory, Practice, and
#    Visualization. Wiley.
make.1d.density.plots <- function(df, prefix, output, device, verbose)
{
  cols <- Filter(function(n) { ! n %in% c("l","LIKE") }, names(dframe))
  for(col in cols)
  {
    if (verbose) cat("Making 1-d density plot for", col, "\n")
    dev.fcn <- get(device)
    filename <- file.path(output, paste(prefix, "_", col, ".", device, sep=""))
    form = as.formula(paste("~", col, sep=""))
    dev.fcn(filename)
    print(densityplot( form
                     , df
                     , type="l"
                     , lwd=2
                     , panel = function(...)
                     {
                       panel.grid(-1,-1)
                       panel.densityplot(...)
                     }
                     , ylab = "posterior probability density"
                     , plot.points = FALSE
                     , bw = "nrd"
                     ))
    invisible(dev.off())
  }
}

find.contours <- function(kde, levels = c(0.68, 0.95))
{
  probs.sorted <- sort(kde$z, decreasing=TRUE)
  probs.cs <- cumsum(probs.sorted)
  # Get the indices of the first values greater than the given confidence levels.
  indices <- sapply( levels
                   , function(x) which(probs.cs>x)[1]
                   )
  probs.sorted[indices]
}

# vmat2df will convert the kind of list returned by kde2d (containing two)
# vectors and a matrix, named x, y and z) into a dataframe with columns
# x, y, z.
vmat2df <- function(u)
{
  g = expand.grid(u$x, u$y)
  data.frame( x = g$Var1, y = g$Var2, z = as.numeric(u$z))
}

# Make a 2-d density plot of  xcol vs. ycol, using data from df.
make.2d.density.plot <- function( df, xcol, ycol, prefix, output, device
                                , use.color
																, nbins)
{
  kde <- kde2d(df[,xcol], df[,ycol], n=nbins	)

  # kde$x and kde$y carry the bin co-ordinates.
  # kde$z carries the estimated density at (x,y). We convert kde$z into
  # the probability for the corresponding bin. We could multiply the density
  # by the area of the bin, but encounter less trouble from rounding if we
  # normalize the sum to 1.
  norm <- sum(kde$z)
  kde$z <- kde$z/norm

  # Find the values of z which correspond to the given confidence levels.
  conf.levels = c(0.68, 0.95)
  zvals = find.contours(kde, conf.levels)
  
  # Convert from vectors+matrix to dataframe, to use lattice plotting.
  d <- vmat2df(kde)
  
  dev.fcn <- get(device)
  filename <- file.path( output
                       , paste(prefix, "_", xcol, "_", ycol, ".", device, sep=""))
  levels <- c(0, zvals, 1)
  labels <- as.character(c(0, conf.levels, 1))
  p <- contourplot( z~x*y, d, at=levels, labels=labels
                  , panel=function(...){panel.grid(-1,-1); panel.contourplot(...)}
                  , xlab = xcol
                  , ylab = ycol
                  , region = use.color
                  , lwd=2
                  , col.regions = function(n,a) rev(terrain.colors(n,a))
                  , colorkey = FALSE
                  )
  dev.fcn(filename)
  print(p)
  invisible(dev.off())
}

# For each pair, generate the kde2d result matrix.
# Determine the values of z at which the 68% and 95% contour lines lie.
# Transform the matrix to a dataframe.
# Make the contour plot.

make.2d.density.plots <- function(df, prefix, output, device, verbose, use.color, nbins)
{
  # Go through all pairs of interesting variables (all but 'LIKE' and 'l',
  # the last two columns).
  n.interesting <- ncol(df)-2
  cols <- names(df)[1:n.interesting]
  pairs <- combn(cols, 2, simplify=FALSE)
  for(pair in pairs)
  {
    xcol <- pair[[1]]
    ycol <- pair[[2]]
    if (verbose) cat("Making 2-d density plot of", xcol, "vs.", ycol, "\n")
    make.2d.density.plot(df, xcol, ycol, prefix, output, device, use.color, nbins)
  }

}

################################################################################
###
### Start of the main program.

option_list <-
  list( make_option( c("-v", "--verbose")
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
      , make_option( c("-f", "--fill")
                   , action="store_true"
                   , default=FALSE
                   , help="Color regions in 2-d density plots"
                   )
      , make_option( c("-b", "--burn")
                   , default = 0
                   , type="integer"
                   , help="Number of burn-in samples to ignore [%default]"
                   )
      , make_option( c("-n", "--nbins")
                   , default = 101
                   , type = "integer"
                   , help="Number of bins in KDE for each of x and y [%default]"
                   )
      )

parser <- OptionParser(option_list=option_list, usage="%prog [options] infile")
args   <- parse_args(parser, positional_arguments = TRUE)
opt    <- args$options

if (length(args$args) != 1)
{
  cat("Incorrect number of required arguments\n\n")
  print_help(parser)
  stop()    
}

input.file <- as.character(args$args)
if (file.exists(input.file) == FALSE )
{
  cat("Unable to open input file", input.file, "\n")
  stop()
}

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
   cat("Processing file", input.file, "\n")

dframe = make.data.frame(input.file, opt$burn)
make.1d.density.plots(dframe, opt$prefix, opt$output, opt$device, opt$verbose)
make.2d.density.plots(dframe, opt$prefix, opt$output, opt$device, opt$verbose, opt$fill, opt$nbins)
