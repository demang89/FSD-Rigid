import hoomd
import hoomd.md
import numpy as np
import math
import gsd
import gsd.hoomd
import random
import itertools
from scipy.spatial.transform import Rotation
import random
import sys
################
phi = float(sys.argv[1])
SEED = 42 #+ int(sys.argv[2])
random.seed(SEED)
np.random.seed(SEED)

nagg = 10
mass = 1.0; R_C1=1.0
I_sys = np.zeros((nagg,3))
quat_sys = np.zeros((nagg,4))
Dg = np.zeros(nagg)
Rg_vec = np.zeros((nagg,3))
mass_sys = np.zeros(nagg)
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
    Dg[i] = 2.0*np.sqrt(sum(I_diagonal)/2.0/natom/mass+2.0)
    Rg_vec[i,0] = np.sqrt(5./2.*(I_diagonal[1]+I_diagonal[2]-I_diagonal[0])/natom/mass)
    Rg_vec[i,1] = np.sqrt(5./2.*(I_diagonal[0]+I_diagonal[2]-I_diagonal[1])/natom/mass)
    Rg_vec[i,2] = np.sqrt(5./2.*(I_diagonal[0]+I_diagonal[1]-I_diagonal[2])/natom/mass)
    if np.linalg.det(E_vec) < 0: E_vec[:,0] *= -1.0
    R = E_vec.T
    diagonal_positions = np.dot(R, agg_pos[:,:].T).T
    E_vec = Rotation.from_matrix(E_vec)
    quat  = E_vec.as_quat()
    I_sys[i,:] = I_diagonal
    quat_sys[i,:] =  quat
    mass_sys[i] = mass*natom
    rigid.body['agg'+str(i)] = {
      "constituent_types": ['A']*natom,
      "positions": diagonal_positions,
      "orientations": [[1,0,0,0]]*natom}

N_particles = 125
n_replicate = int(N_particles/nagg) + 1

index = np.arange(n_replicate*nagg)
np.random.shuffle(index)
typeid = np.tile(np.arange(nagg),n_replicate)[index]
I_sys = np.repeat(I_sys,n_replicate,axis=0)[index]
quat_sys = np.repeat(quat_sys,n_replicate,axis=0)[index]
mass_sys = np.repeat(mass_sys,n_replicate)[index]
Dg_sys = np.repeat(Dg,n_replicate)[index]
types = []
for i in range(nagg):
  types.append('agg'+str(i))
types.append('A')

L_Z = L_Y = L_X = (4./3.*np.pi*N_particles*natom/phi)**(1./3.)
m = 5
spacing = L_X/m
position = np.zeros((N_particles,3))
p = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            position[p] = [i*spacing,j*spacing,k*spacing]
            p+=1
position -= np.array([0.5*L_X,0.5*L_Y,0.5*L_Z])

frame = gsd.hoomd.Frame()
frame.configuration.box = [L_X, L_Y, L_Z, 0, 0, 0]
frame.particles.N = N_particles
frame.particles.types = types
frame.particles.typeid = typeid[0:N_particles]
frame.particles.position = position[0:N_particles, :]
frame.particles.diameter = Dg_sys[0:N_particles]
frame.particles.mass = mass_sys[0:N_particles]
frame.particles.moment_inertia = I_sys[0:N_particles]
frame.particles.orientation = quat_sys[0:N_particles]
with gsd.hoomd.open(name='fractal_centers.gsd', mode='w') as f:
    f.append(frame)

sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=SEED)
sim.create_state_from_gsd(filename='fractal_centers.gsd')
rigid.create_bodies(sim.state)

rigid_centers_and_free = hoomd.filter.Rigid(("center",))
all_ = hoomd.filter.All()
typeA = hoomd.filter.Type(["A"])

kT=1.0
kappa = 1.0
Z = 0.*kT
A = 0.*kT
kn = 1000*kT
nl = hoomd.md.nlist.Cell(buffer=0.1, exclusions=['body']);
dlvo = hoomd.md.pair.DLVO(nlist=nl, default_r_cut=1.0);
dlvo.params[('A', 'A')] = dict(A=A, kappa=kappa, Z=Z, a1=R_C1, a2=R_C1, kn=kn, epsq=1e-2)
dlvo.r_cut[('A', 'A')] = 6.0*R_C1
for i in range(nagg):
  dlvo.params[('agg'+str(i), 'A')] = dict(A=0,kappa=kappa,Z=0,a1=R_C1, a2=R_C1, kn=0, epsq=1e-4)
  dlvo.r_cut[('agg'+str(i), 'A')] = 0
  for j in range(i,nagg):
    dlvo.params[('agg'+str(i), 'agg'+str(j))] = dict(A=0, kappa=kappa, Z=0, a1=R_C1, a2=R_C1, kn=0, epsq=1e-4)
    dlvo.r_cut[('agg'+str(i), 'agg'+str(j))] = 0

eta = 1.0
brownian=hoomd.md.methods.Brownian(filter=rigid_centers_and_free, kT=kT)
for i in range(nagg):
  tau = 3*np.pi*eta*Dg[i]
  a = Rg_vec[i]
  Reff = np.sqrt([(a[1]**2+a[2]**2)/2,
                (a[0]**2+a[2]**2)/2,
                (a[0]**2+a[1]**2)/2])
  gamma_r = 8*np.pi*eta*Reff**3
  brownian.gamma['agg'+str(i)]= tau
  brownian.gamma_r['agg'+str(i)]=gamma_r

integrator=hoomd.md.Integrator(dt=dt,forces=[dlvo],methods=[brownian],integrate_rotational_dof=True,rigid=rigid)
sim.operations.integrator = integrator

gsd_writer = hoomd.write.GSD(filename='lattice.gsd',trigger=nframe,filter=all_,dynamic=['property','momentum','attribute'],mode='wb')
gsd_writer.maximum_write_buffer_size= 1024**2
sim.operations.writers.append(gsd_writer)

sim.run(nrun)
gsd_writer.flush()

del sim


