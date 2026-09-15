## \package PSEv3.shear_function
# classes representing shear functions, which can be input of an integrator and variant
# to shear the box of a simulation

from hoomd.PSERigid import _PSERigid

import hoomd

## shear function interface representing shear flow field described by a function
class _shear_function:
    ## Constructor and check the validity of zero param
    # \param zero Specify absolute time step number location for 0 in \a points. Use 0 to indicate the current step.
    def __init__(self, zero = 0):
        self._cpp_obj = None
        self._offset = zero

    ## Get shear rate at a certain time step, might be useful when switching strain field
    # \param timestep the timestep
    def get_shear_rate(self, timestep):
        return self._cpp_obj.getShearRate(timestep)

    ## Get the strain at a certain time step. The strain is not wrapped
    # \param timestep the timestep
    def get_strain(self, timestep):
        return self._cpp_obj.getStrain(timestep)

    ## Get the offset of this shear function
    def get_offset(self):
        return self._cpp_obj.getOffset()


## concrete class representing steady shear, no shear by default if shear_rate is not provided
class steady(_shear_function):
    ## Constructor of steady shear function
    # \param dt the time interval between each timestep, must be the same with the global timestep
    # \param shear_rate the shear rate of the shear, default is zero, should be zero or positive
    # \param zero the time offset
    def __init__(self, dt, shear_rate = 0, zero = 0):
        _shear_function.__init__(self, zero)
        self._cpp_obj = _PSERigid.SteadyShearFunction(shear_rate, self._offset, dt)


## concrete class representing simple sinusoidal oscillatory shear
class sine(_shear_function):
    ## Constructor of simple sinusoidal oscillatory shear
    # \param dt the time interval between each timestep, must be the same with the global timestep
    # \param shear_rate the maximum shear rate of the ocsillatory shear, must be positive
    # \param shear_freq the frequency (real frequency, not angular frequency) of the ocsillatory shear, must be positive
    # \param zero the time offset
    def __init__(self, dt, shear_rate, shear_freq, zero = 0):

        if shear_rate <= 0:
            raise RuntimeError("Shear rate must be positive (use steady class instead for zero shear)\n")
        if shear_freq <= 0:
            raise RuntimeError("Shear frequency must be positive (use steady class instead for steady shear)\n")

        _shear_function.__init__(self, zero)
        self._cpp_obj = _PSERigid.SinShearFunction(shear_rate, shear_freq, self._offset, dt)


## concrete class representing chirp oscillatory shear
class chirp(_shear_function):
    ## Constructor of chirp oscillatory shear
    # \param dt the time interval between each timestep, must be the same with the global timestep
    # \param amplitude the strain amplitude of Chirp oscillatory shear, must be positive
    # \param omega_0 minimum angular frequency, must be positive
    # \param omega_f maximum angular frequency, must be positive and larger than omega_0
    # \param periodT final time of chirp
    # \param zero the time offset
    def __init__(self, dt, amplitude, omega_0, omega_f, periodT, zero = 0):
        _shear_function.__init__(self, zero)
        self._cpp_obj = _PSERigid.ChirpShearFunction(amplitude, omega_0, omega_f, periodT, self._offset, dt)


## concrete class representing Tukey window function
class tukey_window(_shear_function):
    ## Constructor of Tukey window function
    # \param dt the time interval between each timestep, must be the same with the global timestep
    # \param periodT time length of the Tukey window function
    # \param tukey_param Tukey window function parameter, must be within (0, 1]
    # \param zero the time offset
    def __init__(self, dt, periodT, tukey_param, zero = 0):

        if tukey_param <= 0 or tukey_param > 1:
            raise RuntimeError("Tukey parameter must be within (0, 1]")

        _shear_function.__init__(self, zero)
        self._cpp_obj = _PSERigid.TukeyWindowFunction(periodT, tukey_param, self._offset, dt)


## concrete class represeting a windowed shear function
class windowed(_shear_function):
    ## Constructor of a windowed shear function
    # The strain of the resulting windowed shear function will be the product of the original shear function and
    # the provided window function
    # \param function_form the original shear function
    # \param window the window function. It is recommended to make sure the offset (zero) of the window function is the same with shear function
    def __init__(self, function_form, window):
        _shear_function.__init__(self, 0) # zero parameter is not used in windowed class anyways
        self._cpp_obj = _PSERigid.WindowedFunction(function_form._cpp_obj, window._cpp_obj)
