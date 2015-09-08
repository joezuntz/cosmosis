from cosmosis.postprocessing.plots import Tweaks
import pylab



class ThinLabels(Tweaks):
    #The file to which this applies
    filename=["2D_cosmological_parameters--omega_b_cosmological_parameters--hubble",
        "2D_cosmological_parameters--yhe_cosmological_parameters--omega_b",
        "2D_cosmological_parameters--yhe_cosmological_parameters--hubble"]

    def run(self):
        ax=pylab.gca()
        labels = ax.get_xticks().tolist()
        for i in xrange(0,len(labels),2):
            labels[i] = ''
        ax.set_xticklabels(labels)
