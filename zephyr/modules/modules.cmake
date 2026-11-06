# SPDX-License-Identifier: Apache-2.0
# Integration glue for Zephyr modules that this repo's west groups fetch (module_ext_root, see ../module.yml).
#
# ExecuTorch (west group `executorch`) ships zephyr/module.yml with `kconfig-ext: True` and `cmake-ext: True`,
# which asks a module_ext_root to define ZEPHYR_EXECUTORCH_KCONFIG and ZEPHYR_EXECUTORCH_CMAKE_DIR. The pinned
# Zephyr has no such glue, so with the group on, every app in the workspace stopped at Kconfig
# (`osource "$(ZEPHYR_EXECUTORCH_KCONFIG)"` resolved to the Zephyr folder: EISDIR).
# executorch/mnist builds ExecuTorch itself from EXECUTORCH_DIR, so the module needs no Kconfig options and no
# CMake: give it an empty Kconfig and leave the CMake dir unset (Zephyr then adds nothing, as for a module
# without CMake).
set(ZEPHYR_EXECUTORCH_KCONFIG ${CMAKE_CURRENT_LIST_DIR}/executorch/Kconfig)
