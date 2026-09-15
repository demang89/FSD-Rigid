# Maintainer: Andrew M. Fiore

set(COMPONENT_NAME PSERigid)

set(_${COMPONENT_NAME}_headers
   Stokes.h
   ShearFunction.h
   ShearFunctionWrap.h
   VariantShearFunction.h
   SpecificShearFunction.h
   Stokes.cuh
   Brownian_FarField.cuh
   Brownian_NearField.cuh
   Helper_Brownian.cuh
   Helper_Debug.cuh
   Helper_Integrator.cuh
   Helper_Mobility.cuh
   Helper_Precondition.cuh
   Helper_Stokes.cuh
   Helper_Saddle.cuh
   Integrator.cuh
   Lubrication.cuh
   Mobility.cuh
   Precondition.cuh
   Solvers.cuh
   rcm.hpp
   cublas_wrappers.hpp
   ContactFriction.cuh
   )

set(_${COMPONENT_NAME}_sources
    module.cc
    Stokes.cc
    Stokes_ResistanceTable.cc
    Stokes_SparseMath.cc
    ShearFunction.cc
    ShearFunctionWrap.cc
    SpecificShearFunction.cc
    VariantShearFunction.cc
    rcm.cpp
    )

set(_${COMPONENT_NAME}_cu_sources
    Brownian_FarField.cu
    Brownian_NearField.cu
    Helper_Brownian.cu
    Helper_Debug.cu
    Helper_Integrator.cu
    Helper_Mobility.cu
    Helper_Precondition.cu
    Helper_Stokes.cu
    Helper_Saddle.cu
    Integrator.cu
    Lubrication.cu
    Mobility.cu
    Precondition.cu
    Saddle.cu
    Stokes.cu
    Solvers.cu
    ContactFriction.cu
    )

if (ENABLE_HIP)
set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources})
set_source_files_properties(${_${COMPONENT_NAME}_cu_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
endif (ENABLE_HIP)

include_directories(/home/dmangal/HOOMD/cusplibrary)

## Build the module
hoomd_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_cuda_sources} ${_${COMPONENT_NAME}_headers} NO_EXTRAS)
# Alias into the HOOMD namespace so that external and symlinked components both work.
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# Find additional libraries to be linked for the plugin
find_library( LAPACK_LIBRARIES openblas PATHS /sw/pkgs/arc/stacks/gcc/10.3.0/OpenBLAS/0.3.23/lib )
if( LAPACK_LIBRARIES-NOTFOUND )
        message(FATAL_ERROR "lapack libraries not found")
endif( LAPACK_LIBRARIES-NOTFOUND )
message(STATUS  "found openblas libraries: ${LAPACK_LIBRARIES}")
set( LAPACK_LIBRARIES ${LAPACK_LIBRARIES})

# Find CUBLAS
find_library( CUBLAS_LIBRARY cublas PATHS /sw/pkgs/arc/cuda/11.8.0/lib64/ )
if( CUBLAS_LIBRARY-NOTFOUND )
        message(FATAL_ERROR "CUBLAS Library not found")
endif( CUBLAS_LIBRARY-NOTFOUND )
message(STATUS  "found CUBLAS library: ${CUBLAS_LIBRARY}")
set( CUBLAS_LIBRARY ${CUBLAS_LIBRARY} )

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME}
                     PRIVATE HOOMD::_hoomd
                     PRIVATE HOOMD::_md
                     PRIVATE ${LAPACK_LIBRARIES}
                     PRIVATE ${CUBLAS_LIBRARY}
		     PRIVATE curand
		     )

# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

set(files
    __init__.py
    integrate.py
    shear_function.py
    variant.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

copy_files_to_build("${files}" "${COMPONENT_NAME}" "*.py")

#if (BUILD_TESTING)
#    add_subdirectory(test-py)
#endif()
