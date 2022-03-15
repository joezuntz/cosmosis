import numpy as np
from astropy.table import Table
from cosmosis import Inifile
import tempfile
import glob
import sys
import os

ignored_cols =["prior", "post", "like", "weight", "old_weight", "old_like"]

def make_names_file(chain, filename):
    print(f"Making names file {filename}")
    latex_file = os.path.join(os.environ["COSMOSIS_SRC_DIR"], "postprocessing", "latex.ini")
    ini = Inifile(latex_file)
    with open(filename, "w") as f:
        for name in chain.colnames:
            if name in ignored_cols:
                continue
            if '--' in name:
                sec, key = name.split("--", 1)
                lname = ini.get(sec.lower(), key.lower(), fallback="")
                f.write(f"{name}    {lname}\n")
            else:
                f.write(f"{name}\n")

            
def make_ranges_file(chain, filename):
    print(f"Making ranges file {filename}")
    with tempfile.NamedTemporaryFile("w") as f:
        started = False
        for line in chain.meta["comments"]:
            line = line.strip('#').strip()
            if line == "END_OF_VALUES_INI":
                break
            elif started:
                f.write(line + "\n")
            elif line == "START_OF_VALUES_INI":
                name = True
                started = True
        f.flush()
        
        ini = Inifile(f.name)

    with open(filename, "w") as f:
        for section in ini.sections():
            for key, value in ini.items(section):
                bits = value.split()
                if len(bits) == 3:
                    f.write(f"{section}-{key}  {bits[0]}  {bits[2]}\n")

def make_chain_file(chain, filename):
    print(f"Making chain file {filename}")
    post = np.array(chain["post"])
    if "weight" in chain.colnames:
        weights = chain["weight"]
    else:
        weights = np.ones(post.size)

    for col in ignored_cols:
        if col in chain.colnames:
            chain.remove_column(col)

    row = chain[0]
    w = weights[0]
    with open(filename, "w") as f:
        for i in range(1, len(chain)):
            next_row = chain[i]
            if row == next_row:
                w += weights[i]
            else:
                row_txt = "  ".join(str(row[col]) for col in chain.colnames)
                f.write(f"{w}  {post[i]}  {row_txt}\n")
                row = next_row
                w = weights[i]
        # write the final row
        f.write(f"{w}  {post[i]}  {row_txt}\n")


def convert_chain(root, out_root):
    chain_files = sorted(glob.glob(f"{root}_*.txt"))
    if not chain_files:
        chain_files = [f"{root}.txt"]

    for f in chain_files:
        print(f"Reading {f}")
    chain = Table.read(chain_files[0], format="ascii")
    make_names_file(chain, f"{out_root}.paramnames")
    make_ranges_file(chain, f"{out_root}.ranges")
    for i, chains_file in enumerate(chain_files):
        chain = Table.read(chains_file, format="ascii")
        make_chain_file(chain, f"{out_root}_{i+1}.txt")


if __name__ == '__main__':
    root = sys.argv[1]
    out_root = sys.argv[2]
    convert_chain(root, out_root)
