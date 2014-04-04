import pydesglue
import numpy as np

#Load the test data
filename="/Users/jaz/pack/wmap_likelihood_v4p1/data/test_cls_v4.dat"
test_data = np.loadtxt(filename).T
#Ell is expected to be type int, so convert that.  Everything else is just doubles.
ell=test_data[0].astype(np.int32)
tt=test_data[1]
ee=test_data[2]
bb=test_data[3]
te=test_data[4]

#Save to a data package
data  = pydesglue.DesDataPackage()
data.set_data(pydesglue.section_names.cmb_cl, "ELL", ell)
data.set_data(pydesglue.section_names.cmb_cl, "TT", tt)
data.set_data(pydesglue.section_names.cmb_cl, "TE", te)
data.set_data(pydesglue.section_names.cmb_cl, "EE", ee)
data.set_data(pydesglue.section_names.cmb_cl, "BB", bb)

#Run the interface
n = data.to_new_fits_handle()
f = pydesglue.load_interface("wmap_interface.so","execute")
result = f(n)


if result:
	print "Failed to run WMAP7"
	sys.exit(1)


data = pydesglue.DesDataPackage.from_fits_handle(n)
like = data.get_param(pydesglue.section_names.likelihoods,"WMAP7_LIKE")
print "Computed WMAP7 like: ", like
expected_like_tot = -7477.656769/2
print "Expected: ", expected_like_tot
print "Difference = ",expected_like_tot - like
