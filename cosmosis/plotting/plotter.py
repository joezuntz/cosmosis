from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import object
import matplotlib
matplotlib.use('Agg')
import pylab
import configparser
import collections
from .utils import NoSuchParameter, section_code

class Plotter(object):
    colors=['blue','red','green','cyan','gray']
    def __init__(self, chain_data,latex_file=None, filetype="png", root_dir='.',prefix='',blind=False, fatal=False, **options):
        self._chain_data = chain_data
        all_names = set()
        for chain_datum in list(self._chain_data.values()):
            for name in list(chain_datum.keys()):
                all_names.add(name)
        self.all_names = sorted(list(all_names))
        self.load_latex(latex_file)
        self.nfile = len(chain_data)
        self.filetype=filetype
        self.options=options
        self.root_dir = root_dir
        self.prefix = prefix
        self.blind = blind
        self.fatal=fatal
        if self.prefix and not self.prefix.endswith('_'):
            self.prefix = self.prefix + "_"


    def command(self, command, *args, **kwargs):
        cmd = command.format(*args, **kwargs) + '\n'
        self._output_file.write(cmd)

    def blind_axes(self):
        pylab.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',      # ticks along the bottom edge are off
            right='off',         # ticks along the top edge are off
            labelbottom='off', # labels along the bottom edge are off
            labeltop='off', # labels along the bottom edge are off
            labelleft='off', # labels along the bottom edge are off
            labelright='off') # labels along the bottom edge are off


    def load_latex(self, latex_file):
        self._display_names = {}
        latex_names = {}
        if latex_file is not None:
            latex_names = configparser.ConfigParser(strict=False)
            latex_names.read(latex_file)
        for i,col_name in enumerate(self.all_names):
            display_name=col_name
            if '--' in col_name:
                section,name = col_name.lower().split('--')
                try:
                    display_name = latex_names.get(section,name)
                except configparser.NoSectionError:
                    section = section_code(section)
                except configparser.NoOptionError:
                    pass                    
                try:
                    display_name = latex_names.get(section,name)
                except:
                    pass
            else:
                if col_name in ["LIKE","likelihood"]:
                    display_name=r"{\cal L}"
            self._display_names[col_name]=display_name

    def cols_for_name(self, name):
        cols = collections.OrderedDict()
        for filename, chain_datum in list(self._chain_data.items()):
            if name in list(chain_datum.keys()):
                cols[filename] = chain_datum[name]
        if not cols:
            raise NoSuchParameter(name)
        return cols

    def parameter_range(self, name):
        cols = self.cols_for_name(name)
        xmin = 1e30
        xmax = -1e30
        for col in list(cols.values()):
            if col.min() < xmin: xmin=col.min()
            if col.max() > xmax: xmax=col.max()
        if xmin==1e30 or xmax==-1e30:
            raise ValueError("Could not find col max/min - NaNs in chain?")
        return xmin,xmax

    def plot_1d_params(self, names):
        if not names:
            names = self.all_names
        for name in names:
            if name.lower()=='like' or name.lower()=='like': continue
            print("Plotting 1D curve for ", name)
            try:
                self._plot_1d(name)
                pylab.savefig("%s/%s%s.%s"%(self.root_dir, self.prefix, name, self.filetype))
            except Exception as error:
                if self.fatal:
                    raise
                else:
                    print("Unable to plot curve - may be only one value?")
                    print(error)
            finally:
                pylab.close()

    def plot_2d_params(self, names):
        if not names:
            names = self.all_names
        for name1 in names:
            for name2 in names:
                if name1!=name2 and name1<name2:
                    if name1.lower()=='like' or name2.lower()=='like': continue
                    print("Plotting 2D curve for ", name1, "versus", name2)
                    try:
                        self._plot_2d(name1,name2)
                        pylab.savefig("%s/%s%s_%s.%s"%(self.root_dir, self.prefix, name1,name2,self.filetype))
                    except Exception as error:
                        print("(No plot)")
                        print(error)
                    finally:
                        pylab.close()
