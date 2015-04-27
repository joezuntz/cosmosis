"""
This file illustrates how we can tweak plots after they
have been created, for example to fix problems or add
more features.

If you want to make a completely new plot then have a look 
at demo 9 instead, and the extra9.py file.

For python experts: any class inheriting from Tweaks 
that is visible here will be instantiated and run on
any matching after the plots are finished. The correct
figure will be made active before this is run.

"""
from cosmosis.postprocessing.plots import Tweaks
import pylab



class AddTitle(Tweaks):
    #This can be a single filename, like "matter_power",
    #a list of filenames, like ["matter_power", "hubble"]
    filename="matter_power"

    def run(self):
        #These commands are run on all the target plots.
        #Let's add a title.  We could put as many commands
        #as we like here.
        pylab.title("CAMB vs Eisenstein & Hu")

#Any number of tweaks can be applied.
class MoreSpace(Tweaks):
    #We can also use this special 'filename',
    #which means all plots are tweaked.
    filename="all plots"
    def run(self):
        #Matplotlib latex labels sometimes fall of the
        #edge of the plot.  Let's tweak to bring them back
        pylab.subplots_adjust(bottom=0.15, left=0.15)

