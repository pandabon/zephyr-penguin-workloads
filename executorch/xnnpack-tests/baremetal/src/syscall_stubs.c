// SPDX-License-Identifier: Apache-2.0
//
// picolibc declares a few POSIX calls in <unistd.h> but, being OS-less, does not define them. GoogleTest's
// gtest-port.cc references them for things a test run on a bare chip never does (is stdout a terminal,
// change directory, capture a stream with dup/dup2); each is a stub that says "not supported", so GoogleTest
// itself stays unmodified. Same behaviour as src/zephyr_compat/gtest_posix_shim.h, which does this for
// Zephyr's <unistd.h> (that one lacks the declarations; here they exist and only the definitions are missing).
// Written for this project (no upstream code).
#include <errno.h>
#include <time.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

int isatty(int fd) { (void)fd; return 0; }
int chdir(const char* path) { (void)path; errno = ENOSYS; return -1; }
// No file system, so no working directory: report "/". GoogleTest (d72f9c8aea68, the version XNNPACK pins)
// aborts at start-up if getcwd() fails.
char* getcwd(char* buf, size_t size) {
  if (buf == NULL || size < 2) { errno = ERANGE; return NULL; }
  buf[0] = '/'; buf[1] = '\0';
  return buf;
}
int dup(int fd) { (void)fd; errno = ENOSYS; return -1; }
int dup2(int fd, int fd2) { (void)fd; (void)fd2; errno = ENOSYS; return -1; }

// The rest of what GoogleTest/libstdc++ reference at link time. No files, no clock: each reports failure or
// a constant (write() is in htif.c). gettimeofday() returns 0, so GoogleTest prints "0 ms" for every test and,
// with no --gtest_random_seed, its "time based" seed is a constant (as on Spike under Zephyr, whose clock is
// deterministic as well; XNN_TEST_SEED exists for a seed of your choice).
int open(const char* path, int flags, ...) { (void)path; (void)flags; errno = ENOENT; return -1; }
int close(int fd) { (void)fd; return 0; }
ssize_t read(int fd, void* buf, size_t n) { (void)fd; (void)buf; (void)n; return 0; }   // end of file
off_t lseek(int fd, off_t off, int whence) { (void)fd; (void)off; (void)whence; errno = ESPIPE; return -1; }
int stat(const char* path, struct stat* st) { (void)path; (void)st; errno = ENOENT; return -1; }
int mkdir(const char* path, mode_t mode) { (void)path; (void)mode; errno = ENOSYS; return -1; }
int gettimeofday(struct timeval* tv, void* tz) {
  (void)tz;
  if (tv) { tv->tv_sec = 0; tv->tv_usec = 0; }
  return 0;
}
// Same for clock_gettime() (XNNPACK's runtime times operators with CLOCK_MONOTONIC for profiling only): time 0.
int clock_gettime(clockid_t clock_id, struct timespec* tp) {
  (void)clock_id;
  if (tp) { tp->tv_sec = 0; tp->tv_nsec = 0; }
  return 0;
}
