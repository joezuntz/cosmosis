# PRODUCTS is the list of PDFs we are to build.
PRODUCTS := cosmosis-intro.pdf cosmosis-design.pdf

# include our latex "build system".
include latex.mk

# You must have GraphViz's 'dot' command on your path to run this.
astro_packages.pdf: astro_packages.dot
	tred $< | dot -Gratio=compressed -T ps2 | ps2pdf - $@

# Tell make that the document PDF file depends on the generated
# figure(s).
cosmosis-intro.pdf: astro_packages.pdf

# Tell make that the generated figure(s) should be removed by the
# 'clean' target
clean_files += astro_packages.pdf


