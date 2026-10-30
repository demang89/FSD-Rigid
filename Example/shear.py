import sys
import hoomd
import hoomd.md
import hoomd.PSERigid
import numpy as np
import math
import gsd
import gsd.hoomd
import random
import itertools
from scipy.spatial.transform import Rotation
################
phi = float(sys.argv[1])
SRs = np.logspace(3,0,15)

SR = SRs[0]
k_n = 1e5
k_t = 1e5
tcont = 0.5/(1.5*k_n*np.sqrt(0.05))
tshear = 1e-3/SR
dt = min(tcont,tshear)

strain = 10
nrun = int(strain/SR/dt)
nframe = int(0.1/SR/dt)
nperiod = int(0.01/SR/dt)

SEED = 42#+int(sys.argv[2])
random.seed(SEED)
np.random.seed(SEED)

class CustomBoxVariant(hoomd.variant.box.BoxVariant):
    def __init__(self,dt,SR,L_X,L_xy):
        self.L_xy = L_xy
        self.dt=dt
        self.SR=SR
        self.L_X = L_X
        hoomd.variant.box.BoxVariant.__init__(self)

    def __call__(self, timestep):
        self.L_xy += self.dt*self.SR
        if(self.L_xy>=0.5):
            self.L_xy-=1.0
        return [self.L_X, self.L_X, self.L_X, self.L_xy, 0, 0]

nagg = 10
mass = 0.01
R_C1 = 1.0
I_ref = np.array([[2.0 / 5.0 * mass * R_C1**2, 0, 0],
                     [0, 2.0 / 5.0 * mass * R_C1**2, 0],
                     [0, 0, 2.0 / 5.0 * mass * R_C1**2]])
rigid = hoomd.md.constrain.Rigid()
with open('agg_nref36_std0.0_k0.5.xyz','r') as f:
  for i in range(nagg):
    natom = int(f.readline())
    f.readline()
    I_gen = np.zeros(shape=(3, 3))
    agg_pos=np.zeros((natom,3))
    for j in range(natom):
      agg_pos[j,:]=f.readline().split()[1:]
    agg_pos -= np.mean(agg_pos,axis=0)
    for j in range(natom):
      r = agg_pos[j,:]
      I_gen += I_ref + mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
    I_diagonal, E_vec = np.linalg.eig(I_gen)
    if np.linalg.det(E_vec) < 0: E_vec[:,0] *= -1.0
    R = E_vec.T
    diagonal_positions = np.dot(R, agg_pos[:,:].T).T
    rigid.body['agg'+str(i)] = {
      "constituent_types": ['A']*natom,
      "positions": diagonal_positions,
      "orientations": [[1,0,0,0]]*natom}

N_particles = 125
L_X = (4./3.*np.pi*N_particles*natom/phi)**(1./3.)

eta=1.0
kT = 1.0
T  = hoomd.variant.Constant(0)
A = 1.0*kT
Z = 0.25*kT
kappa = 1.0

rigid_centers_and_free = hoomd.filter.Rigid(("center",))
all_ = hoomd.filter.All()
typeA = hoomd.filter.Type(["A"])
nl = hoomd.md.nlist.Cell(buffer=0.1,default_r_cut=18.0)#, exclusions=['body']);

sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=SEED)
sim.timestep = 0
sim.create_state_from_gsd(filename='lattice.gsd',frame=-1)

shear_func = hoomd.PSERigid.shear_function.steady(dt,SR,0)
pse = hoomd.PSERigid.integrate.PSERigid(filter1 = rigid_centers_and_free, filter2=typeA, kT = T, nlist=nl,
    xi=0.25, error=1e-7, period=nperiod, F_rep=Z, F_att=A,
    kappa=kappa, k_n=k_n, epsq=1e-4, rcut=5.0, rot_diff=0.0,
    T_ext=0.0, omega_ext=0.0, F_ext=0, seed=SEED,
    shear_func=shear_func, max_strain=0.5, rfd_eps=10.0, delta=0, rk2=False, fric='contact', k_t=k_t, muf=1.0)

integrator=hoomd.md.Integrator(dt=dt,forces=[],methods=[pse],integrate_rotational_dof=True,rigid=rigid)
sim.operations.integrator = integrator

L_xy = sim.state.box.xy
box_resize = hoomd.update.BoxResize(trigger=hoomd.trigger.After(0),filter=all_,box=CustomBoxVariant(dt,SR,L_X,L_xy))
sim.operations.updaters.append(box_resize)

gsd_writer = hoomd.write.GSD(filename='shear.gsd',trigger=nframe,filter=all_,dynamic=['property','momentum','attribute'],mode='wb')
gsd_writer.maximum_write_buffer_size= 1024**2
sim.operations.writers.append(gsd_writer)
sim.run(nrun)
gsd_writer.flush()
