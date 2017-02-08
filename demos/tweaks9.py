from cosmosis.postprocessing.plots import Tweaks
import pylab


# Note that these are not really the truth values for
# the actual Universe.  If I knew those I wouldn't
# need CosmoSIS.  This is an example of something that
# you might more sensibly do in a simulation.

truths = {
    "cosmological_parameters--omega_m":0.2935148566413599,
    "cosmological_parameters--h0":0.73800089258391999,
    "cosmological_parameters--w":-1.0,
    "cosmological_parameters--omega_b":0.04,
    "cosmological_parameters--omega_k":0.0,
    "supernova_params--deltam": 0.040281808929275797,
    "supernova_params--alpha": 0.13675033800637484,
    "supernova_params--beta": 3.1029245604093223,
}






class AddTruth(Tweaks):
    #This can be a single filename, like "matter_power",
    #a list of filenames, like ["matter_power", "hubble"]
    filename="all plots"

    def run(self):
        #These commands are run on all the target plots.
        #Let's add a title.  We could put as many commands
        #as we like here.

        # self.info contains, in this case, the parts of the 
        # filename, e.g. cosmological_parameters--omega_m_cosmological_parameters--h0.png
        # has info = ["2D", "cosmological_parameters--omega_m", "cosmological_parameters--h0"]
        # whereas "supernova_params--alpha.png" has
        # info = "supernova_params--alpha"

        if len(self.info)==3 and self.info[0]=="2D":
            # This is a 2D plot - draw a star at the truth
            x = truths.get(self.info[1])
            y = truths.get(self.info[2])
            if x is None:
                print "No truth value for {}".format(self.info[1])
            if y is None:
                print "No truth value for {}".format(self.info[2])
            if x is None or y is None:
                return
            pylab.plot([x], [y], 'y*', markersize=20)
        elif len(self.info)==1:
            # This is a 1D plot - just draw a line at the truth
            x = truths.get(self.info[0])
            if x is None:
                print "No truth value for {}".format(self.info[1])
                return
            pylab.plot([x,x], [0,1], 'k-')



