import subprocess
import os
import tempfile
import shutil


this_dir = os.path.split(__file__)[0]
libtest_dir = os.path.join(this_dir, "libtest")
src_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
print("Source dir = ", src_dir)

def test_lib():
    with tempfile.TemporaryDirectory() as tempdir:
        shutil.copytree(libtest_dir, f"{tempdir}/libtest")
        cmd = f"cd {tempdir}/libtest && make test"
        print(cmd)
        env = os.environ.copy()
        env['COSMOSIS_SRC_DIR'] = src_dir
        subprocess.run(cmd, shell=True, check=True, env=env)


if __name__ == '__main__':
    test_lib()
