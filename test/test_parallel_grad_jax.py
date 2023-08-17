import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import config
import jax
config.update("jax_enable_x64", True)
from scipy.stats import unitary_group
import unittest
import time
import rqcopt as oc




class TestBrickwallJax(unittest.TestCase):

    def test_parallel_gates_gradient(self):
            """
            Test gradient computation for parallel gates.
            """
            rng = np.random.default_rng()
            # system size
            L = 8
            # random unitary
            V = unitary_group.rvs(4, random_state=rng)
            # surrounding matrix
            U = 0.5 * oc.crandn(2 * (2**L,), rng)
            perm = rng.permutation(L)
            # Start timing for the computation
            start_time = time.time()
            dV = oc.parallel_gates_grad(V, L, U, perm)
            dV_jax = oc.parallel_gates_grad_jax(V, L, U, perm)
            print(dV_jax)
            print(dV)
            err = np.linalg.norm(dV.conj() - dV_jax)
            print("Error: ", err)

    def test_parallel_gates_directed_gradient(self):
        """
        Test directed gradient computation for parallel gates.
        """
        rng = np.random.default_rng()
        # system size
        L = 8
        # random unitary
        V = unitary_group.rvs(4, random_state=rng)
        # direction
        Z = 0.5 * oc.crandn((4, 4), rng)
        perm = rng.permutation(L)
        dW = oc.parallel_gates_directed_grad(V, L, Z, perm)
        # numerical gradient via finite difference approximation
        f = lambda t: oc.parallel_gates(V + t*Z, L, perm)
        h = 1e-6
        dW_num = (f(h) - f(-h)) / (2*h)
        self.assertTrue(np.allclose(dW_num, dW))


if __name__ == "__main__":
    unittest.main()
