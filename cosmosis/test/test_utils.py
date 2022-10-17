import cosmosis.utils
import tempfile
import os
import contextlib
import subprocess
import pytest
import sys

dulwich_orig = cosmosis.utils.dulwich


@contextlib.contextmanager
def setup_git_repo():
    with tempfile.TemporaryDirectory() as dirname:
        repo_dir = f"{dirname}/repo"
        repo_subdir = f"{repo_dir}/subdir"
        os.mkdir(repo_dir)
        os.mkdir(repo_subdir)

        cmd = ["git", "init", "."]
        p = subprocess.run(cmd, cwd=repo_dir)
        cmd = ["git", "config", "--local", "user.email", "test@user.com"]
        p = subprocess.run(cmd, cwd=repo_dir)
        cmd = ["git", "config", "--local", "user.name", "Test User"]
        p = subprocess.run(cmd, cwd=repo_dir)


        with open(f"{repo_subdir}/f.txt", "w") as f:
            f.write("hello\n")

        cmd = ["git", "add", "subdir/f.txt"]
        p = subprocess.run(cmd, cwd=repo_dir)

        cmd = ["git", "commit", "-m", "added_file"]
        p = subprocess.run(cmd, cwd=repo_dir)

        cmd = ["git", "log"]
        p = subprocess.run(cmd, cwd=repo_dir, capture_output=True, universal_newlines=True)
        print('log stdout:', p.stdout)
        print('log stderr:', p.stderr)
        sha = p.stdout.split("\n")[0].split()[1]
        
        yield sha, repo_dir, repo_subdir

@pytest.mark.skipif(sys.version_info < (3, 7), reason="test requires python3.7 or higher")
@pytest.mark.skipif(dulwich_orig is None, reason="dulwich not installed")
def test_dulwich_git_path1():
    with setup_git_repo() as info:
        sha, repo_dir, repo_subdir = info
        sha2 = cosmosis.utils.get_git_revision_dulwich(repo_dir)
        sha3 = cosmosis.utils.get_git_revision_dulwich(repo_subdir)
        assert sha == sha2
        assert sha == sha3

@pytest.mark.skipif(sys.version_info < (3, 7), reason="test requires python3.7 or higher")
@pytest.mark.skipif(dulwich_orig is None, reason="dulwich not installed")
def test_dulwich_git_path2():
    with setup_git_repo() as info:
        sha, repo_dir, repo_subdir = info
        sha2 = cosmosis.utils.get_git_revision(repo_dir)
        sha3 = cosmosis.utils.get_git_revision(repo_subdir)
        assert sha == sha2
        assert sha == sha3

@pytest.mark.skipif(sys.version_info < (3, 7), reason="test requires python3.7 or higher")
def test_git_fallback():
    cosmosis.utils.dulwich = None
    try:
        with setup_git_repo() as info:
            sha, repo_dir, repo_subdir = info
            sha2 = cosmosis.utils.get_git_revision(repo_dir)
            sha3 = cosmosis.utils.get_git_revision(repo_subdir)
            assert sha == sha2
            assert sha == sha3
    finally:
        cosmosis.utils.dulwich = dulwich_orig

@pytest.mark.skipif(sys.version_info < (3, 7), reason="test requires python3.7 or higher")
def test_git_nosub():
    os.environ["COSMOSIS_NO_SUBPROCESS"] = "1"
    cosmosis.utils.dulwich = None
    try:
        with setup_git_repo() as info:
            sha, repo_dir, repo_subdir = info
            sha2 = cosmosis.utils.get_git_revision(repo_dir)
            sha3 = cosmosis.utils.get_git_revision(repo_subdir)
            assert sha2 == ""
            assert sha3 == ""
    finally:
        cosmosis.utils.dulwich = dulwich_orig
        del os.environ["COSMOSIS_NO_SUBPROCESS"]
