from cosmosis.postprocessing.plots import Tweaks
import pylab



class AddLegend(Tweaks):
    #This can be a single filename, like "matter_power",
    #a list of filenames, like ["matter_power", "hubble"]
    filename="all plots"

    def run(self):
        #These commands are run on all the target plots.
        #Let's add a title.  We could put as many commands
        #as we like here.
        ax=pylab.gca()
        lines=ax.get_lines()
        pylab.legend([lines[0], lines[5]],["Smith", "Takahashi"], loc="upper left")

        pylab.ylabel("${\cal L}$")
