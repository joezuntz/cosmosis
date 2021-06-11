import subprocess
import os

this_dir = os.path.split(__file__)[0]
libtest_dir = os.path.join(this_dir, "libtest")


def test_lib():
    cmd = f"cd {libtest_dir} && make clean && make test"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


if __name__ == '__main__':
    test_lib()
