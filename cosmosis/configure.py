import os
import sys
import argparse

parser = argparse.ArgumentParser(description="print commands that set up the cosmosis env")
cosmosis_src_dir = os.path.split(__file__)[0]
parser.add_argument("--source", default=cosmosis_src_dir)
parser.add_argument("--no-omp", action='store_false', dest='omp', help='Switch off OpenMP')
parser.add_argument("--omp", action='store_true', dest='omp', help='Switch on OpenMP (default)')
parser.add_argument("--debug", action='store_true', dest='debug', help='Switch on debug mode')
parser.add_argument("--no-debug", action='store_false', dest='debug', help='Switch on debug mode')
parser.add_argument("--no-conda", action='store_false', dest='conda', help='Switch off conda flags')
parser.add_argument("--conda", action='store_true', dest='conda', help='Switch on conda flags (default)')


def generate_commands(cosmosis_src_dir, debug=False, omp=True, conda=True):
    if conda and "CONDA_PREFIX" not in os.environ:
        raise RuntimeError("Conda environment not detected")

    commands = [
        f"export COSMOSIS_SRC_DIR=\"{cosmosis_src_dir}\"",
        "export LIBRARY_PATH=$LIBRARY_PATH:$COSMOSIS_SRC_DIR/datablock",
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COSMOSIS_SRC_DIR/datablock",
        "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$COSMOSIS_SRC_DIR/datablock",
        "export COSMOSIS_ALT_COMPILERS=1",
    ]

    if conda:
        commands += [
            'export GSL_LIB=$CONDA_PREFIX/lib',
            'export GSL_INC=$CONDA_PREFIX/include',
            'export FFTW_LIBRARY=$CONDA_PREFIX/lib',
            'export FFTW_INCLUDE_DIR=$CONDA_PREFIX/include',
            'export LAPACK_LINK="-L$CONDA_PREFIX/lib -llapack"',
            'export CFITSIO_LIB=$CONDA_PREFIX/lib',
            'export CFITSIO_INC=$CONDA_PREFIX/include',
        ]


    if omp:
        commands.append("export COSMOSIS_OMP=1")
        
    if debug:
        commands.append("COSMOSIS_DEBUG=1")

    return commands

if __name__ == '__main__':
    args = parser.parse_args()
    cmds = generate_commands(args.source, debug=args.debug, omp=args.omp, conda=args.conda)
    for command in cmds:
        print(command)

