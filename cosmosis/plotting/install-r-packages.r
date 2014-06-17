#!/usr/bin/env Rscript

install.if.needed <- function(pkgname)
{
  cat("Trying to load package", pkgname, "\n")
  rc <- require(pkgname, character.only=TRUE)
  if ( !rc )
    invisible(install.packages( pkgname
                              , repos="http://cran.us.r-project.org"))
}

needed.packages <- c( "optparse"
                    , "latticeExtra"
                    , "reshape2"
                    )
for (pkg in needed.packages)
  install.if.needed(pkg)

