# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
from hoomd.PSERigid import _PSERigid
from hoomd.PSERigid import shear_function

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
from hoomd.md import _md
import hoomd
from hoomd import _hoomd
from hoomd.operation import AutotunedObject
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.filter import ParticleFilter
from hoomd.variant import Variant
from hoomd.md.methods import Method
import math

class PSERigid(Method):
    ## Specifies the integrator for Fast Stokesian Dynamics (FSD)
    #
    # filter1           Group of rigid bodies.
    # filter2           Group of primary particles
    # kT                 Temperature of the simulation (in energy units)
    # nlist             Neighbor list to use
    # xi                Ewald splitting parameter
    # error 		Error threshold to use for calculations (Spectral Ewald parameters are determined on the fly using this bound)
    # shear_func    Shear flow protocol
    # F_rep          max electrostatic repulsion magnitude
    # F_att          max vdw attraction magnitude
    # kappa          inverse Debye length          
    # k_n            collision spring constant      
    # period		Frequency of stresslet output
    # epsq          roughness of the primary particles
    # T_ext         External torque
    # omega_ext     Frequency of external torque
    # F_ext         External force
    
    def __init__(self, filter1, filter2, kT, nlist, xi, error, rcut, shear_func, F_rep=0,  F_att=0, kappa=0, k_n=0,
                epsq=0, rot_diff=0, T_ext=0, omega_ext=0, F_ext=0, period=0, seed=1234, max_strain=0.5, 
                rfd_eps=100.0, delta=0.0, fric=None, h0=0.005, alpha=0.01, k_t=0, muf=0):
        param_dict = ParameterDict(
                filter1 = ParticleFilter,
                filter2 = ParticleFilter,
                kT=Variant,
                nlist=hoomd.md.nlist.NeighborList,
                xi=float,
                error=float,
                rcut=float,
                F_rep=float,
                F_att=float,
                kappa=float,
                k_n=float,
                epsq=float,
                rot_diff=float,
                T_ext=float,
                omega_ext=float,
                F_ext=float,
                period=int,
                seed=int,
                max_strain=float,
                rfd_eps=float,
                delta=float,
                fric=str,
                h0=float,
                alpha=float,
                k_t=float,
                muf=float)

        param_dict.update(dict(filter1=filter1,filter2=filter2,kT=kT,nlist=nlist,xi=xi,error=error,
        rcut=rcut,F_rep=F_rep,F_att=F_att,kappa=kappa,k_n=k_n,epsq=epsq,rot_diff=rot_diff,
        T_ext=T_ext,omega_ext=omega_ext, F_ext=F_ext,period=period,seed=seed, max_strain=max_strain, 
        rfd_eps=rfd_eps, delta=delta, fric=fric, h0=h0, alpha=alpha, k_t=k_t, muf=muf))

        self._param_dict.update(param_dict)
        self.shear_func = shear_func

    def _attach_hook(self):
        if self.nlist._attached and self._simulation != self.nlist._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent neighbor list."
                f" This is happending since the force is moving to a new "
                f"simulation. Set a new nlist to suppress this warning.",
                RuntimeWarning,
            )
            self.nlist = copy.deepcopy(self.nlist)
        self.nlist._attach(self._simulation)
        if isinstance(self._simulation.device, hoomd.device.CPU):
            raise RuntimeError('Error creating Stokes');
        else:
            self.nlist._cpp_obj.setStorageMode(_md.NeighborList.storageMode.full)
            self._cpp_obj = _PSERigid.Stokes(self._simulation.state._cpp_sys_def,
                                          self._simulation.state._get_group(self.filter1),
                                          self._simulation.state._get_group(self.filter2),
                                          self.kT,
                                          self.nlist._cpp_obj,
                                          self.xi,
                                          self.error,
                                          self.rcut,
                                          self.F_rep,
                                          self.F_att,
                                          self.kappa,
                                          self.k_n,
                                          self.epsq,
                                          self.rot_diff,
                                          self.T_ext,
                                          self.omega_ext,
                                          self.F_ext,
                                          self.period,
                                          self.seed,
                                          self.rfd_eps,
                                          self.delta,
                                          self.fric,
                                          self.h0,
                                          self.alpha,
                                          self.k_t,
                                          self.muf);
        self._cpp_obj.setShear(self.shear_func._cpp_obj,self.max_strain)
        super()._attach_hook()

    def _detach_hook(self):
        self.nlist._detach()

    def _setattr_param(self, attr, value):
        if attr == "nlist":
            self._nlist_setter(value)
            return
        super()._setattr_param(attr, value)

    def _nlist_setter(self, new_nlist):
        if new_nlist is self.nlist:
            return
        if self._attached:
            raise RuntimeError("nlist cannot be set after scheduling.")
        self._param_dict._dict["nlist"] = new_nlist
