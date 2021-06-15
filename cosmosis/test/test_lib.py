import subprocess
import os
import tempfile
import shutil
import pytest

this_dir = os.path.split(__file__)[0]
libtest_dir = os.path.join(this_dir, "libtest")
src_dir = os.path.abspath(os.path.join(this_dir, os.pardir))

def test_lib():
    with tempfile.TemporaryDirectory() as tempdir:
        shutil.copytree(libtest_dir, f"{tempdir}/libtest")
        env = os.environ.copy()
        env['COSMOSIS_SRC_DIR'] = src_dir

        # First try making the fortran tester.  If it can't
        # build it at all then we skip the test as our env is
        # presumably not set up to build things properly
        cmd1 = f"cd {tempdir}/libtest && make fortran_t"
        print(cmd1)
        result = subprocess.run(cmd1, shell=True, check=False, env=env)

        if result.returncode:
            pytest.skip("Not in an environment that can compile library tests")

        # If we can compile things, though, then we will want to
        # try compiling everything else and running all the tests
        cmd2 = f"cd {tempdir}/libtest && make test"
        print(cmd2)
        subprocess.run(cmd2, shell=True, check=True, env=env)


if __name__ == '__main__':
    test_lib()
