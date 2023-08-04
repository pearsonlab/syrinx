
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import diffrax
import equinox as eqx
import optax
import time
from typing import Callable, Any
from tensorboardX import SummaryWriter
import uuid

# model parameters
eps1 = 1.25e8
eps2 = 7.5e9
beta1 = -2e3
beta2 = 5.3e5
C = 2e8
delta = 15e6

# simulation parameters
sr = 4e4  # data sampling rate (Hz)
T = 1.2
t_axis = jnp.arange(0, T, 1/sr)
params_true = jnp.array([eps1/1e8, eps2/1e8, beta1/1e3, beta2/1e3, C/1e8, delta/1e7])


def make_pulse_sequence(pulse_fun, arg_list):
    def applicator(carry, args):
       t = carry
       return t, pulse_fun(t, *args)
    
    def scan_and_sum(t):
        _, out = lax.scan(applicator, t, arg_list)
        return jnp.sum(out)

    return scan_and_sum

############### make pressure function
pfreq = 2 
plocs = jnp.array([0.3, 0.9])
pA = 0.025  # pressure amplitude (Volts)
p0 = -0.005 # pressure DC offset (Volts)
pwid = 0.08

ppulse = lambda t, loc: pA * jnp.exp(-0.5 * (t - loc)**2/pwid**2) + p0

P = jax.vmap(make_pulse_sequence(ppulse, (plocs,)))


############### make vS tension function

def make_tension_pulse_fn(shape=1, scale=1, peak=1):
    norm = peak * jnp.exp(shape - shape * jnp.log(shape) - shape * jnp.log(scale)) 

    fn = lambda t, loc: norm * jnp.exp((t - loc)/scale) * jnp.maximum((loc - t), 0)**shape 

    return fn
    
        
kshape = 5  # shape parameter of gamma function
kscale = 0.025  # rate parameter of gamma function (s)
kpeak = 0.06  # peak value (Volts)
klocs = jnp.array([0, 0.5, 1, 1.5])

pulse = make_tension_pulse_fn(shape=kshape, scale=kscale, peak=kpeak)
K = jax.vmap(make_pulse_sequence(pulse, (klocs,)))


############### make dTB tension function

dfreq = 2 
dlocs1 = jnp.arange(0.1, t_axis[-1], 1/dfreq)  # pressure pulse frequency (Hz)
dlocs2 = jnp.arange(0.45, t_axis[-1], 1/dfreq)  # pressure pulse frequency (Hz)
dlocs = jnp.sort(jnp.concatenate([dlocs1, dlocs2]))
dA = jnp.array([0.05, 0.02, 0.01, 0.05, 0.03,])
dwid = 0.01

gpulse = lambda t, loc, amp: amp * jnp.exp(-0.5 * (t - loc)**2/dwid**2)

D = jax.vmap(make_pulse_sequence(gpulse, (dlocs, dA)))


def gradfun(t, y, args):
    # params: eps1, eps2, beta1, beta2, C, delta
    params, extra_args = args
    K, D, P = extra_args
    t_arr = jnp.array([t])
    eps = (params[0] + params[1] * K(t_arr)) * 1e8
    B = (params[2] + params[3] * P(t_arr)) * 1e3
    C = params[4] * 1e8
    D0 = params[5] * D(t_arr) * 1e7

    xdot = y[1] 
    ydot = -eps * y[0] - C * y[0]**2 * y[1] + B * y[1] - D0

    return jnp.array((xdot, ydot[0]))


######### solve ODE
term = diffrax.ODETerm(gradfun)
solver = diffrax.Dopri5()
saveat = diffrax.SaveAt(ts=jnp.linspace(0, 1.2, int(sr)))
stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

soln = diffrax.diffeqsolve(term, solver, t0=0, t1=T, dt0=0.5/sr, y0=jnp.array((0, 0)), saveat=saveat,
                  stepsize_controller=stepsize_controller, args=(params_true, (K, D, P)), max_steps=int(1e6))

audio = jnp.interp(t_axis, soln.ts, soln.ys[:, 0])


########### make data set
ys_full = jnp.stack([jnp.interp(t_axis, soln.ts, soln.ys[:, idx]) for idx in range(soln.ys.shape[-1])]).T

T_snippet = 100  # samples
ys = ys_full.reshape((-1, T_snippet, soln.ys.shape[-1]))
ts = t_axis.reshape((-1, T_snippet))


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


class NeuralODE(eqx.Module):
    func: Callable
    params: jax.Array
    extra_args: Any

    def __init__(self, gradfun, params, *args, **kwargs):
        super().__init__(**kwargs)
        self.func = gradfun
        self.extra_args = args
        self.params = params

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=(ts[1] - ts[0]),
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            saveat=diffrax.SaveAt(ts=ts),
            args=(self.params, self.extra_args),
            max_steps=int(1e6)
        )
        return solution.ys

def main(
    init,
    batch_size=64,
    lr_strategy=(1e-3, 1e-3),
    steps_strategy=(500, 2500), 
    length_strategy=(0.1, 1),
    seed=5678,
    print_every=100,
    runnum=uuid.uuid4()
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    _, length_size, data_size = ys.shape
    
    # eps1, eps2, beta1, beta2, C, delta
    model = NeuralODE(gradfun, init, K, D, P)

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model)(ti, yi[:, 0])
        return jnp.mean((yi[:, :, 0] - y_pred[:, :, 0]) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    writer = SummaryWriter(f"logs/run{runnum}", flush_secs=1)

    globstep = 0
    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adam(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[:, : int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        for step, (ti, yi) in zip(
            range(steps), dataloader((_ts, _ys), batch_size, key=loader_key)
        ):
            start = time.time()
            loss, model, opt_state = make_step(ti, yi, model, opt_state)
            end = time.time()
            globstep += 1
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
                writer.add_scalar('loss', loss, globstep)
                writer.add_scalars('parameters', {
                    'eps1': model.params[0],
                    'eps2': model.params[1],
                    'beta1': model.params[2],
                    'beta2': model.params[3],
                    'C': model.params[4],
                    'delta': model.params[5],
                }, globstep, end)

    writer.close()

    return ts, ys, model

def simple_syrinx(seed, **kwargs):
    init_key, model_key = jr.split(jr.PRNGKey(seed))
    init = params_true * (1 + 0.05 * jr.normal(init_key, params_true.shape))
    ts, ys, model = main(init=init, **kwargs)
    print(params_true, init, model.params)

import fire
if __name__ == '__main__':
    fire.Fire(simple_syrinx)