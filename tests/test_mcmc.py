"""Focused mathematical and resume tests for the two-stage experiment."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from ddpm import data
from ddpm.mcmc import (gibbs_parameters, gibbs_mu, initialize_paths, predict_mean,
                       conditional_log_density, proposal_log_kernel, mh_log_ratio,
                       accept_proposals, make_sweep, adapt_epsilons)
from ddpm.sampling import p_sample_step
from ddpm.two_stage import split_indices, experiment_schedule
from scripts.run_two_stage_mcmc import save_chain, load_chain


def nonlinear_predictor(variables, y, t):
    return variables['params']['scale'] * jnp.tanh(y) + t.reshape((-1,1,1,1)) * .001


class MCMCTests(unittest.TestCase):
    def setUp(self):
        self.schedule = experiment_schedule()
        self.params = {'scale': jnp.array(.15)}
        self.y = jnp.array([.2,-.5]).reshape(2,1,1,1)

    def test_disjoint_deterministic_partition(self):
        a,b,p = split_indices(20261002,64,64)
        self.assertEqual((len(a),len(b),len(p)), (30000,30000,64))
        self.assertEqual(len(np.intersect1d(a,b)),0)
        np.testing.assert_array_equal(np.sort(np.r_[a,b]), np.arange(60000))
        self.assertTrue(np.isin(p,b).all())
        np.testing.assert_array_equal(p,split_indices(20261002,64,64)[2])

    def test_image_only_loading_never_requests_labels(self):
        with patch.object(data,'_download',return_value=np.zeros((1,28,28,1),np.uint8)) as download:
            data.load_mnist_images()
            download.assert_called_once_with('train-images-idx3-ubyte.gz',data.MNIST_IMAGE_URL,data._read_images)

    def test_schedule(self):
        expected = np.linspace(.001,.2,100)
        np.testing.assert_allclose(self.schedule.betas,expected,rtol=2e-6)
        np.testing.assert_allclose(self.schedule.alphas,1-expected,rtol=2e-6)
        np.testing.assert_allclose(self.schedule.alpha_bars,np.cumprod(1-expected),rtol=2e-6)
        self.assertLess(float(self.schedule.alpha_bars[-1]),1e-4)

    def test_reverse_mean_and_beta_variance_including_final(self):
        for index in [0,50,99]:
            t = jnp.full((2,),index)
            eps = nonlinear_predictor({'params':self.params},self.y,t)
            mean = (self.y-self.schedule.betas[index]*eps/jnp.sqrt(1-self.schedule.alpha_bars[index]))/jnp.sqrt(self.schedule.alphas[index])
            np.testing.assert_allclose(predict_mean(nonlinear_predictor,self.params,self.schedule,self.y,index),mean)
            key = jax.random.key(index)
            actual = p_sample_step(nonlinear_predictor,self.params,self.schedule,self.y,index,key,variance='beta')
            np.testing.assert_allclose(actual,mean+jnp.sqrt(self.schedule.betas[index])*jax.random.normal(key,self.y.shape))

    def test_gibbs_analytic_and_identity_terminal_covariance(self):
        for tau in [1.0,2.5]:
            mean,var = gibbs_parameters(self.y,tau)
            expected_var = 1/(1/tau+2)
            np.testing.assert_allclose(mean,expected_var*self.y.sum(axis=0))
            self.assertEqual(var,expected_var)
            key=jax.random.key(3)
            np.testing.assert_allclose(gibbs_mu(self.y,key,tau),mean+jnp.sqrt(var)*jax.random.normal(key,mean.shape))

    def test_terminal_and_interior_logs_and_input_gradients(self):
        lower=self.y*.8
        for index in [0,45,99]:
            if index==99:
                upper_mean=jnp.full_like(self.y,.1)
                upper_var=1.0
            else:
                upper_mean=predict_mean(nonlinear_predictor,self.params,self.schedule,self.y*1.2,index+1)
                upper_var=self.schedule.betas[index+1]
            def log(y):
                return conditional_log_density(y,lower,upper_mean,upper_var,index,nonlinear_predictor,self.params,self.schedule)
            mean=predict_mean(nonlinear_predictor,self.params,self.schedule,self.y,index)
            expected=-.5*((lower-mean)**2/self.schedule.betas[index]+(self.y-upper_mean)**2/upper_var)
            np.testing.assert_allclose(log(self.y),expected.reshape(2),rtol=1e-6)
            grad=jax.grad(lambda y:log(y).sum())(self.y)
            delta=jnp.zeros_like(self.y).at[0,0,0,0].set(1e-3)
            difference=(log(self.y+delta)[0]-log(self.y-delta)[0])/.002
            np.testing.assert_allclose(grad[0,0,0,0],difference,rtol=1e-3,atol=1e-3)
            self.assertTrue(np.isfinite(grad).all())

    def test_mala_both_proposal_densities_and_acceptance(self):
        current=self.y; proposed=self.y+.17
        gradient=-current; reverse_gradient=-proposed
        epsilon=.3
        forward=-.5*((proposed-current-epsilon**2/2*gradient)**2/epsilon**2).reshape(2)
        reverse=-.5*((current-proposed-epsilon**2/2*reverse_gradient)**2/epsilon**2).reshape(2)
        np.testing.assert_allclose(proposal_log_kernel(proposed,current,gradient,epsilon),forward,rtol=1e-6)
        np.testing.assert_allclose(proposal_log_kernel(current,proposed,reverse_gradient,epsilon),reverse,rtol=1e-6)
        ratio=mh_log_ratio(jnp.array([-2.,-3.]),jnp.array([-4.,-1.]),current,proposed,gradient,reverse_gradient,epsilon)
        np.testing.assert_allclose(ratio,jnp.array([-2.,2.])+reverse-forward,rtol=1e-6)
        values,accept=accept_proposals(current,proposed,jnp.array([0.,-100.]),jnp.array([.5,.5]))
        np.testing.assert_array_equal(accept,[True,False])
        np.testing.assert_array_equal(values[0],proposed[0])
        np.testing.assert_array_equal(values[1],current[1])

    def test_forward_initialization_is_markov(self):
        key=jax.random.key(19)
        actual,end=initialize_paths(self.y,self.schedule,key)
        previous=self.y
        for i in range(100):
            key,noise_key=jax.random.split(key)
            previous=jnp.sqrt(self.schedule.alphas[i])*previous+jnp.sqrt(self.schedule.betas[i])*jax.random.normal(noise_key,previous.shape)
            np.testing.assert_allclose(actual[i],previous,atol=2e-6)
        np.testing.assert_array_equal(jax.random.key_data(end),jax.random.key_data(key))

    def test_sweep_freezes_theta_and_y0_and_adaptation(self):
        observed=self.y
        original=np.asarray(observed).copy()
        paths,key=initialize_paths(observed,self.schedule,jax.random.key(4))
        eps=.05*jnp.sqrt(self.schedule.betas)
        sweep=make_sweep(nonlinear_predictor,self.schedule)
        mu,updated,key,diagnostics=sweep(self.params,observed,paths,key,eps,1.)
        self.assertEqual(updated.shape,(100,2,1,1,1))
        np.testing.assert_array_equal(observed,original)
        self.assertTrue(np.asarray(diagnostics[-1]).all())
        # Differentiation through the sampler cannot train theta; input scores still exist.
        gradient=jax.grad(lambda scale:sweep({'scale':scale},observed,paths,key,eps,1.)[1].sum())(self.params['scale'])
        self.assertEqual(float(gradient),0.)
        np.testing.assert_array_equal(adapt_epsilons(eps,jnp.zeros(100),2,201,200),eps)
        self.assertTrue(np.all(adapt_epsilons(eps,jnp.zeros(100),2,1,200)<eps))

    def test_checkpoint_resume_reproduces_chain(self):
        observed=jnp.zeros((2,28,28,1))
        paths,key=initialize_paths(observed,self.schedule,jax.random.key(11))
        eps=.02*jnp.sqrt(self.schedule.betas)
        sweep=make_sweep(nonlinear_predictor,self.schedule)
        mu,paths,key,_=sweep(self.params,observed,paths,key,eps,1.)
        state=dict(sweep=np.array(1),mu=np.asarray(mu),paths=np.asarray(paths),
                   rng=np.asarray(jax.random.key_data(key)),epsilons=np.asarray(eps),accepted=np.zeros(100))
        config=dict(n=2,total_sweeps=3)
        root=Path(__file__).resolve().parents[1]/'.cache/tmp'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            directory=Path(tmp)
            save_chain(directory,state,config)
            loaded,_=load_chain(directory,config)
            expected=sweep(self.params,observed,paths,key,eps,1.)
            actual=sweep(self.params,observed,jnp.asarray(loaded['paths']),
                         jax.random.wrap_key_data(jnp.asarray(loaded['rng'])),jnp.asarray(loaded['epsilons']),1.)
            for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
                if jax.dtypes.issubdtype(a.dtype,jax.dtypes.prng_key):
                    a,b=jax.random.key_data(a),jax.random.key_data(b)
                np.testing.assert_array_equal(a,b)


if __name__=='__main__':
    unittest.main()
