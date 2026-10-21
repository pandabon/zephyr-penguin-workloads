// SPDX-License-Identifier: Apache-2.0
//
// Force-included (-include) into the C++ files of GoogleTest and of the test app. Zephyr's POSIX
// <unistd.h> lacks a few calls that GoogleTest's gtest-port.h/.cc use for things a bare-metal run
// never does (is stdout a terminal, change directory, capture a stream with dup/dup2). They are
// defined here as stubs that report "not supported", so GoogleTest itself stays unmodified.
// Nothing here runs in a normal test run: stream capture is only used by death tests and
// CaptureStdout(), which XNNPACK's microkernel tests do not use.
#pragma once

#include <errno.h>
#include <unistd.h>

#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif

static inline int isatty(int) { return 0; }
static inline int chdir(const char*) { errno = ENOSYS; return -1; }
static inline char* getcwd(char*, size_t) { errno = ENOSYS; return nullptr; }
static inline int dup(int) { errno = ENOSYS; return -1; }
static inline int dup2(int, int) { errno = ENOSYS; return -1; }
