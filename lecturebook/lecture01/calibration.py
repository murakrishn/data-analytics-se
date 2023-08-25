#~ovn!
"""Example for calibration

Author:
    R Murali Krishnan
    
Date:
    08.17.2023
    
"""

import torch
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.infer import MCMC, HMC, NUTS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from functools import partial

TT = torch.tensor

pyro.set_rng_seed(7)
pyro.clear_param_store()

# Forward Model
def forward_model(k, v, m, y0, L):
    
    # Calculate the angular velocity
    omega = 2. * torch.pi * v / L

    # Calculate the amplitude
    X = torch.abs(k * y0 / (k - m * omega ** 2))
    return X, omega

def generate_ground_truth():
    # Sample params from their distributions
    inputs = dist.Uniform(TT([159999., 80.0*1e3/3600., 100., 0.0*1e-3, 1.]),
                          TT([160001., 150.*1e3/3600., 200., 100*1e-3, 2.]))
    sample = inputs.sample()
    k, v, m, y0, L = sample[0], sample[1], sample[2], sample[3], sample[4]

    X, omega = forward_model(k, v, m, y0, L)
    return k, v, m, y0, L, X, omega

k_g, v_g, m_g, y0_g, L_g, X_g, omega_g = generate_ground_truth()
print("~~ovn!")

noise_scale = 1e-4

X_noisy = X_g + noise_scale * torch.randn(X_g.shape)

# MCMC model 1
def mcmc_model1(Xobs):
    y0 = pyro.sample("y0", dist.Uniform(TT([0.*1e-3]), TT([100.*1e-3])))
    L = pyro.sample("L", dist.Uniform(TT([1.]), TT([2.])))
    return pyro.sample("Xobs", dist.Normal(forward_model(k_g, v_g, m_g, y0_g, L_g)[0], scale=noise_scale), obs=Xobs)


# _data = torch.ones_like(X_noisy)

# pyro.render_model(mcmc_model1, model_args=(_data,), filename="mcmc_model1.pdf")

# hmc_kernel = HMC(mcmc_model1, num_steps=4)
# nuts_kernel = NUTS(mcmc_model1, adapt_step_size=True)
# n_samples = 200
# mcmc = MCMC(nuts_kernel, num_samples=int(n_samples * 3), warmup_steps=300, num_chains=1)
# mcmc.run(X_noisy)

# MCMC model 2
X_noise_scale = 1e-4
v_noise_scale = 0.1
m_noise_scale = 0.1


def forward_model_omega(v, L):
    # Calculate the angular velocity
    omega = 2. * torch.pi * v / L
    return omega

def forward_model_x(k, m, y0, omega):

    # omega = forward_model_omega(v, L)
    # Calculate the amplitude
    X = torch.abs(k * y0 / (k - m * omega ** 2))
    return X


def mcmc_model2(kdata, Xdata=None, vdata=None, mdata=None):
    v = pyro.sample("v", dist.Uniform(TT([80.*1e3/3600.]), TT([150.*1e3/3600.])))
    m = pyro.sample("m", dist.Uniform(TT([100.]), TT([200.])))
    y0 = pyro.sample("y0", dist.Uniform(TT([0.*1e-3]), TT([100.*1e-3])))
    L = pyro.sample("L", dist.Uniform(TT([1.]), TT([2.])))
    k = pyro.sample("k", dist.Uniform(TT([159999.]), TT([160001.])))
    # omega = pyro.deterministic("omega", forward_model_omega(v, L))
    omega = pyro.sample("omega", dist.Normal(forward_model_omega(v, L), scale=1e-9))
    # import pdb; pdb.set_trace()
    # k = pyro.sample("k", dist.Uniform(TT([159999.]), TT([160001.])), obs=kdata)
    pyro.sample("k_obs", dist.Normal(k, scale=TT([1e-9])), obs=kdata)
    pyro.sample("X_obs", dist.Normal(forward_model_x(k, m, y0, omega), scale=X_noise_scale), obs=Xdata)
    pyro.sample("v_obs", dist.Normal(v, scale=v_noise_scale), obs=vdata)
    pyro.sample("m_obs", dist.Normal(m, scale=m_noise_scale), obs=mdata)
    # return Xobs, vobs, mobs



# MCMC model 3

v_noisy = v_g + v_noise_scale * torch.randn(v_g.shape)
m_noisy = m_g + m_noise_scale * torch.randn(m_g.shape)
X_noisy = X_g + X_noise_scale * torch.randn(X_g.shape)

pyro.render_model(mcmc_model2, 
                  model_args=(torch.ones_like(k_g), 
                              torch.ones_like(X_noisy), 
                              torch.ones_like(v_noisy), 
                              torch.ones_like(m_noisy)), 
                  filename="mcmc_model2.pdf")


def model_MCMC(X_data, v_data, m_data):
    v = pyro.sample( "v", dist.Uniform(torch.tensor([80.0*1e3/3600.0]), torch.tensor([150.*1e3/3600.0])) )
    m = pyro.sample( "m", dist.Uniform(torch.tensor([100.]), torch.tensor([200.])) )
    y0 = pyro.sample( "y0", dist.Uniform(torch.tensor([0.*1e-3]), torch.tensor([100.*1e-3])) )
    L = pyro.sample( "L", dist.Uniform(torch.tensor([1.]), torch.tensor([2.])) )
    pyro.sample("Xobs", dist.Normal(forward_model(k_g, v, m, y0, L)[0], scale=X_noise_scale), obs=X_data)
    pyro.sample("vobs", dist.Normal(v, scale=v_noise_scale), obs=v_data)
    pyro.sample("mobs", dist.Normal(m, scale=m_noise_scale), obs=m_data)

pyro.render_model(model_MCMC, 
                  model_args=(torch.ones_like(X_noisy), torch.ones_like(v_noisy), torch.ones_like(m_noisy)), 
                  filename="mcmc_model3.pdf")

hmc_kernel = HMC(model_MCMC, step_size=0.0855, num_steps=4)
nuts_kernel= NUTS(model_MCMC, adapt_step_size=True)
n_samples = 300
mcmc = MCMC(nuts_kernel, num_samples=int(n_samples*3), warmup_steps=300, num_chains=1)
mcmc.run(X_noisy, v_noisy, m_noisy)

thin = 3 #adjacent MCMC samples are correlated so computationally it can make sense to throw some samples out.
v_samples_MCMC = mcmc.get_samples()['v'][::thin,:]
m_samples_MCMC = mcmc.get_samples()['m'][::thin,:]
y0_samples_MCMC = mcmc.get_samples()['y0'][::thin,:]
L_samples_MCMC = mcmc.get_samples()['L'][::thin,:]
print(v_samples_MCMC.shape, m_samples_MCMC.shape, y0_samples_MCMC.shape, L_samples_MCMC.shape)

print(mcmc.summary())
# import h5py
# import os

# if os.path.exists('data.h5'):
#     os.remove('data.h5')

# f = h5py.File('data.h5', 'w')
# grp = f.create_group('mcmc_results')
# mcmc_samples = mcmc.get_samples()
# for dset_name, data in mcmc_samples.items():
#     dset = grp.create_dataset(dset_name, data=data[::thin, :])
# f.close()
# import pdb; pdb.set_trace()


def pair_plot(df, xi_data):
    fig, axes = plt.subplots(len(xi_data[0]), len(xi_data[0]), figsize = (18, 12), sharex="col", tight_layout=True)

    COLUMNS = list(df.columns)
    COLUMNS.remove('Sample type')

    for i in range(len(COLUMNS)):
        for k in range(len(COLUMNS)):

            # If this is the lower-triangule, add a scatterlpot for each group.
            if i > k:
                a = sns.scatterplot(data=df, x=COLUMNS[k], y=COLUMNS[i],
                                  hue="Sample type", ax=axes[i, k], s=10, legend=False)
                a.set(xlabel=None)
                a.set(ylabel=None)

            # If this is the main diagonal, add kde plot
            if i == k:
                b = sns.kdeplot(data=df, x=COLUMNS[k], hue="Sample type",  common_norm=False, ax=axes[i, k])
                axes[i, k].axvline(x=xi_data[0][k], color = 'black', ls ='--')
                b.set(xlabel=None)
                b.set(ylabel=None)

                if k == 0:
                    sns.move_legend(b, "center right", bbox_to_anchor=(5.2,-1.25), title=None,frameon=True,)
                    #sns.move_legend(b, "lower center", bbox_to_anchor=(2.5, 1), ncol=3, title=None,frameon=True,)
                else:
                    axes[i, k].legend([],[], frameon=False)

            # If on the upper triangle
            if i < k:
                axes[i, k].remove()

    for i in range(len(COLUMNS)):
        k=0
        axes[i, k].set_ylabel(COLUMNS[i])

    for k in range(len(COLUMNS)):
        i=len(COLUMNS)-1
        axes[i, k].set_xlabel(COLUMNS[k])

    # See the chart now
    plt.show()
    # plt.close()

def X_samples_Prior():
  Xs = torch.empty(n_samples)
  for i in range(n_samples):
    inputs = dist.Uniform(torch.tensor([80.0*1e3/3600.0, 100., 0.*1e-3, 1.]), torch.tensor([150.*1e3/3600.0, 200., 100*1e-3, 2.]))
    sample = inputs.sample()
    v, m, y0, L = sample[0], sample[1], sample[2], sample[3]
    Xs[i] = forward_model(k_g, v, m, y0, L)[0]
  return Xs

def X_samples_MCMC():
  Xs = torch.empty(n_samples)
  for i in range(n_samples):
    k, v, m, y0, L = k_g, v_samples_MCMC[i], m_samples_MCMC[i], y0_samples_MCMC[i], L_samples_MCMC[i]
    Xs[i] = forward_model(k, v, m, y0, L)[0]
  return Xs

Prior_dict = dict()
Prior_dict[r'$v$'] = dist.Uniform(torch.tensor([80.0*1e3/3600.0]), torch.tensor([150.*1e3/3600.0])).sample([n_samples]).data.numpy().flatten()
Prior_dict[r'$m$'] = dist.Uniform(torch.tensor([100.]), torch.tensor([200.])).sample([n_samples]).data.numpy().flatten()
Prior_dict[r'$y0$'] = dist.Uniform(torch.tensor([0.*1e-3]), torch.tensor([100.*1e-3])).sample([n_samples]).data.numpy().flatten()
Prior_dict[r'$L$'] = dist.Uniform(torch.tensor([1.]), torch.tensor([2.])).sample([n_samples]).data.numpy().flatten()
Prior_dict[r'$X$'] = X_samples_Prior().data.numpy()

MCMC_dict = dict()
MCMC_dict[r'$v$'] = v_samples_MCMC.data.numpy().flatten()
MCMC_dict[r'$m$'] = m_samples_MCMC.data.numpy().flatten()
MCMC_dict[r'$y0$'] = y0_samples_MCMC.data.numpy().flatten()
MCMC_dict[r'$L$'] = L_samples_MCMC.data.numpy().flatten()
MCMC_dict[r'$X$'] = X_samples_MCMC().data.numpy()

df_Prior = pd.DataFrame(Prior_dict)
df_MCMC = pd.DataFrame(MCMC_dict)
df_Prior['Sample type'] = 'Prior'
df_MCMC['Sample type'] = 'MCMC'

df = pd.concat([df_Prior, df_MCMC])
df.reset_index(drop=True, inplace=True)
pair_plot(df, torch.tensor([v_g, m_g, y0_g, L_g, X_g]).reshape(1,-1))
