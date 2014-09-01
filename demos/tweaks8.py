from cosmosis.postprocessing.plots import Tweaks
import pylab

class AddTitle(Tweaks):
    #This can be a single filename, like "matter_power",
    #a list of filenames, like ["matter_power", "hubble"]
    #or a special value (see MoreSpace below), 
    #which means all plots are affected.
    filename="matter_power"

    def run(self):
        #These commands are run on all the target plots.
        #Let's add a title.
        pylab.title("Matter Power Spectra")

#Any number of tweaks can be applied.
#We could have put this in the same tweak 
#as above, or like this as a separate one
class MoreSpace(Tweaks):
    filename="all plots"
    def run(self):
        #And also tweak the spacing a little,
        #since sometimes matplotlib latex labels fall of the
        #edge of the plot
        pylab.subplots_adjust(bottom=0.15, left=0.15)

