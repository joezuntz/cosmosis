from cosmosis.runtime import FunctionModule
from cosmosis.datablock import DataBlock
from cosmosis.gaussian_likelihood import GaussianLikelihood, SingleValueGaussianLikelihood
import numpy as np
import os

def test_gaussian():
    class MyLikelihood(GaussianLikelihood):
        x_section = "aaa"
        x_name = "a"
        y_section = "bbb"
        y_name = "b"
        like_name = "lll"

        def build_data(self):
            x_obs = np.array([1.0, 2.0, 3.0])
            y_obs = x_obs * 2
            return x_obs, y_obs

        def build_covariance(self):
            covmat = np.diag([0.1, 0.1, 0.1])
            return covmat

    mod = MyLikelihood.as_module("my")

    # no extra config info
    mod.setup({"my":{"include_norm":True}})

    block = DataBlock()
    block["aaa", "a"] = np.arange(5.)
    block["bbb", "b"] = np.arange(5.) * 2
    status = mod.execute(block)

    assert status == 0
    assert np.isclose(block["data_vector", "lll_chi2"], 0)
    assert np.isclose(block["data_vector", "lll_log_det"], 3*np.log(0.1))
    assert block["data_vector", "lll_n"] == 3

    assert np.isclose(block["likelihoods", "lll_like"], -3*np.log(0.1)/2)

def test_single_gaussian():

    class MySingleLikelihood(SingleValueGaussianLikelihood):
        section = "sec"
        name = "name"
        like_name = "xxx"
        mean = 3.0
        sigma = 0.1


    mod = MySingleLikelihood.as_module("my2")

    # no extra config info
    mod.setup({"my2":{"include_norm":True, "likelihood_only": False}})

    block = DataBlock()

    block["sec", "name"] = 4.0
    status = mod.execute(block)

    # check cholesky correctly calculated
    assert np.isclose(mod.data.chol, 0.1)

    assert status == 0
    assert np.isclose(block["data_vector", "xxx_chi2"], 100)
    assert np.isclose(block["data_vector", "xxx_log_det"], 2*np.log(0.1))
    assert block["data_vector", "xxx_n"] == 1

    assert np.isclose(block["data_vector", "xxx_theory"], 4.0)
    assert np.isclose(block["data_vector", "xxx_data"], 3.0)
    assert np.isclose(block["data_vector", "xxx_inverse_covariance"], 100.0)
    # sim should be within 10 sigma!
    assert 3.0 < block["data_vector", "xxx_simulation"] < 5.0

    assert np.isclose(block["likelihoods", "xxx_like"], -50.0 - np.log(0.1))


if __name__ == '__main__':
    test_gaussian()


