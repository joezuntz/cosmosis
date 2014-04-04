OBJ = aplowly.c \
clik.c \
clik_gibbs.c \
clik_helper.c \
clik_lowlike.c \
erfinv.c \
lklbs.c \
cldf/cldf.c \
minipmc/distribution.c \
minipmc/errorlist.c  \
minipmc/io.c     \
minipmc/mvdens.c
#clik_egfs.c \
# clik_actspt.c \
#clik_egfs_addon.c \
#clik_example_c.c \
#clik_fortran.c \
# clik_wmap.c \
#lowly.c \
#lowly_common.c \
#lowly_pol.c \


CFLAGS = -DNO_HEALPIX -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers/ -framework veclib -I./minipmc 
LDFLAGS = -lgsl -lcfitsio
all:
	gcc $(OBJ) $(CFLAGS) -o lib1.a $(LDFLAGS)

