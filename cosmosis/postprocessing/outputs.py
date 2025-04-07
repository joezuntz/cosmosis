from . import lazy_pylab as pylab

class MiniTable:
    """
    A simple class for storing and writing out a table of data.

    This is to avoid imposing a dependency on astropy tables.

    """
    def __init__(self, cols):
        self.cols = cols
        self.rows = []

    def to_astropy(self):
        from astropy.table import Table
        return Table(rows=self.rows, names=self.cols)

    def append(self, row):
        if len(row)!=len(self.cols):
            raise ValueError("Row has wrong number of columns")
        self.rows.append(row)

    def write(self, filename):
        with open(filename, "w") as f:
            f.write("#")
            f.write(" ".join(self.cols))
            f.write("\n")
            for row in self.rows:
                f.write("  ".join(str(x) for x in row))
                f.write("\n")
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, row_or_col):
        if isinstance(row_or_col, int):
            return self.rows[row_or_col]
        else:
            i = self.cols.index(row_or_col)
            return np.array([row[i] for row in self.rows])


class PostprocessProduct(object):
    def __init__(self, name, filename, value, info=None):
        self.name = name
        self.filename = filename
        self.value = value
        self.info = info

    def save(self):
        pass

class PostprocessPlot(PostprocessProduct):
    def save(self):
        pylab.figure(self.value.number)
        pylab.savefig(self.filename)
        pylab.close()

    def tweak(self, tweak):
        print("Tweaking", self.name)
        pylab.figure(self.value.number)
        tweak.info = self.info
        tweak.run()

class PostprocessTable(PostprocessProduct):
    def save(self):
        self.value.write(self.filename)

class PostprocessText(PostprocessProduct):
    def save(self):
        self.value.seek(0)
        text = self.value.read()
        with open(self.filename, "w") as f:
            f.write(text)
        self.value.close()