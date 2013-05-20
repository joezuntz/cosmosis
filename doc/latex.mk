########################################################################
# latex.mk
#
# Make production of LaTeX documents trivial, not just easy.
#
####################################
# Usage.
#
# In order to use this, simply define a variable PRODUCTS to sepcify
# your final documents, like:
#
#   PRODUCTS := mydoc1.pdf mydoc2.pdf
#
# Followed by the include of this fragment:
#
#   include <relative-path>/latex.mk
#
# Optionally, following the include you may:
#
#   override clean_files += more files to clean
#
#   override clean_dirs += more dirs to clean
#
# See below for the default contents of these variables.
#
####################################
# Defined targets.
#
# An unqualified "make" is equivalent to, "make all," which will produce
# all defined $(PRODUCTS).
#
# Other targets:
#
#   clean: Remove $(clean_dirs) and $(clean_files)
#
#   clobber: Additionally remove $(products)
#
#   macenv: Add the LaTeX environment variables (TEXINPUTS, etc.) to the
#           Mac OS X GUI environment to enable applications such as
#           aquamacs started from the GUI to have the correct settings
#           for (e.g.)  internal LaTeX invocation.
#
#   view: Start the correctly defined viewer for each defined
#         $(PRODUCT).  e.g. to set the pdf view to skim, you need a
#         file ~/.latexmkrc containing:
#
#           $pdf_previewer = "open -a skim %O %S"
#
#   <prod>: Make the single product <prod> regardless of whether it is
#           in $(PRODUCTS).
#
#   view-<prod>: Run the defined viewer for <prod> after first making
#                sure that it is up-to-date.
#
####################################
# Advanced use:
#
# If you need to add targets (generation of some dependencies for
# example, like figures), you should define that target, eg:
#
# myFigure.tex: myFigure.dat
#
# The latexmk machinery shoudl take care of ensuring that it is invoked
# at the correct time.
#		
########################################################################

########################################################################
# These variables may be overridden or added to for the clean / clobber
# targets.
clean_files = \
  *.aux \
  *.bbl \
  *.blg \
  *.fdb_latexmk \
  *.fls \
  *.log \
  *.nav \
  *.out \
  *.rel \
  *.snm \
  *.synctex.gz \
  *.tex.bak \
  *.toc \
  *.vrb \
  *~

clean_dirs = $(DEPS_DIR)

########################################################################
# No user-serviceable parts below.

####################################
# Variable definitions.

DEPS_DIR = .deps

LATEXMK = latexmk \
            -recorder -use-make -deps \
            -e '@cus_dep_list = ();' \
            -f

$(foreach file,$(PRODUCTS),$(eval -include $(DEPS_DIR)/$(file).d))

rm_files := $(RM) *~ *.out *.toc *.snm *.nav *.vrb *.log *.aux *.synctex.gz *.fdb_latexmk *.fls *.rel *.bbl *.blg; $(RM) -r $(DEPS_DIR)
ensure_deps = mkdir -p $(DEPS_DIR)

safe_fail = { $(RM) "$@"; false; }

def_maybe_pushenv = \
function maybe_pushenv() \
{ \
  local var; \
  for var in "$$@"; do \
    if [[ -n "$$var" ]] && [[ -n "$$(eval echo \"\$$$$var\")" ]] ; then \
      launchctl setenv "$$var" "$$(eval echo \"\$$$$var\")" && \
        echo "Pushed $$var to application env."; \
    else \
      true; \
    fi; \
  done; \
}

####################################
# Dependencies.

$(foreach file,$(PRODUCTS),$(eval -include $(DEPS_DIR)/$(file).d))

####################################
# Targets

all: $(PRODUCTS)

.PHONY: clean clobber macenv

$(filter %.dvi, $(PRODUCTS)): %.dvi : %.tex
	$(ensure_deps)
	$(LATEXMK) --dvi --deps-out=$(DEPS_DIR)/$@.d $(<) || $(safe_fail)

$(filter %.pdf, $(PRODUCTS)): %.pdf : %.tex
	$(ensure_deps)
	$(LATEXMK) --pdf --deps-out=$(DEPS_DIR)/$@.d $(<) || $(safe_fail)

$(filter %.ps, $(PRODUCTS)): %.ps : %.tex
	$(ensure_deps)
	$(LATEXMK) --ps --deps-out=$(DEPS_DIR)/$@.d $(<) || $(safe_fail)

clean:
	$(RM) -r $(clean_dirs)
	$(RM) $(clean_files)

clobber: clean
	$(RM) $(PRODUCTS)

macenv:
	@$(def_maybe_pushenv); maybe_pushenv TEXINPUTS BSTINPUTS BIBINPUTS MFINPUTS MPINPUTS

# View targets
.PHONY: $(foreach d, $(PRODUCTS), view-$d)

$(filter %.pdf,$(foreach d,$(PRODUCTS),view-$d)) : view-% : %
	latexmk -pdf -pv $(*:.pdf=)

$(filter %.dvi,$(foreach d,$(PRODUCTS),view-$d)) : view-% : %
	latexmk -dvi -pv $(*:.dvi=)

$(filter %.ps,$(foreach d,$(PRODUCTS),view-$d))  : view-% : %
	latexmk -ps -pv $(*:.ps=)

view: $(foreach v,$(filter %.ps %.dvi %.pdf,$(PRODUCTS)),view-$(v))

### Local Variables:
### mode: makefile-gmake
### End:
