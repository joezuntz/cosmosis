from cosmosis.runtime import FunctionModule
from cosmosis.datablock import DataBlock
from cosmosis.gaussian_likelihood import GaussianLikelihood
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



if __name__ == '__main__':
    test_gaussian()


