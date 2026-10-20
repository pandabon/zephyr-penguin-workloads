# SPDX-License-Identifier: Apache-2.0
# make help                      the BSP's front door (build / run / doctor ...), see its list
# make build APP=basics/hello_world  build one app for the Penguin SoC   (ELF: build/<app>/zephyr/zephyr.elf)
# make build APP=atlas/upstream CMAKE_ARGS=-DATLAS_TEST=saturn_atlas_pause_resume
# make suite                     build (and run on Spike, if installed) every app in this repo
# make suite BOARD=chipyard_penguin/penguin/vcu118_ta ARGS='-f atlas'
# The targets live in the BSP (mk/user.mk); this file only says where the BSP is.
# Found as the west project chipyard-bsp (workspace made from this repo's west.yml), else as the
# workspace's manifest repo (the maintainers' workspace, whose manifest is the BSP itself).
BSP ?= $(shell west list -f '{abspath}' chipyard-bsp 2>/dev/null | head -1)
ifeq ($(BSP),)
BSP := $(shell west list -f '{abspath}' manifest 2>/dev/null | head -1)
endif
ifeq ($(BSP),)
$(error chipyard-bsp not found: run `west update` in the workspace, or set BSP=<dir>)
endif
DEFAULT_TARGET := help
include $(BSP)/mk/user.mk

.PHONY: suite
suite: need-env
	@echo "==> every scenario of this repo; ARGS='-f text' filters, SPIKE_EXTRA_ARGS adds simulator devices"
	$(BSP)/scripts/run_tests.py --root $(CURDIR) -j $(JOBS) --board $(BOARD) $(ARGS)
