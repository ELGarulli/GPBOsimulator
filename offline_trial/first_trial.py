import numpy as np
from DataSet_Fake import DataSetFake
from SimGPBO import SimGPBO
import copy

c = 128
e = 3
# j = query actually tested?
t = 7
dummy = 2

emgs = np.array(["pc1", "pc2", "pc3"])
nChan = c
sorted_resp = np.random.rand(c, e, t, dummy)
sorted_respMean = np.mean(sorted_resp, axis=2)
sorted_isvalid = np.ones([c, e, t, dummy])
#x, y = np.linspace(0, 0.3*c/2, c//2), np.array([0, 1])
x, y = [i for i in range(0, c//2, 1)], np.array([0, 1])
ch2xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


dataset_fake = DataSetFake(emgs, nChan, sorted_resp, sorted_isvalid, ch2xy, sorted_respMean)

sim = SimGPBO(name = 'ttest_og_',
                    ds = copy.deepcopy(dataset_fake),
                    AF = 'UCB',
                    NB_REP = 1,
                    NB_IT = 15,
                    KAPPA = 6,
                    NB_RND = 1
                    )

# sim.select_emgs([0])

sim.run_simulations(gp_origin='gpytorch', response_type='valid',
                         hyperparams_storage=True, HP_estimation=True, manual_seed=True, outputscale=1., noise=0.01)

