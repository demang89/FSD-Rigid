## \package PSEv3.variant
# classes representing the variant class to facilitate box_resize

from hoomd.PSERigid import _PSERigid
from hoomd.PSERigid import shear_function

from hoomd import variant
from hoomd.variant.scalar import Variant

from hoomd import _hoomd
import hoomd
import sys

## Variant class holding a functional form of shear field
# Used as an argument for box_resize class to deform the box
class shear_variant(Variant):
    ## Specify shear field represented by a function form with a limited timesteps
    #
    # \param function_form the functional form of the oscillatory shear
    # \param total_timestep the total timesteps of the shear, equal to shear_end_timestep - shear_start_timestep, must be positive
    # \param max_strain the maximum absolute value of the strain, use 0.5 in almost all the cases
    def __init__(self, function_form, total_timestep, max_strain = 0.5):
        # initialize the base class
        Variant.__init__(self)

	    # check total_timestep is positive
        if total_timestep <= 0:
            hoomd.context.msg.error("Cannot create a shear_variant with 0 or negative points\n")
            raise RuntimeError('Error creating variant')

        # create the c++ mirror class
        self.cpp_variant = _PSERigid.VariantShearFunction(function_form.cpp_function, int(total_timestep), -max_strain, max_strain)
