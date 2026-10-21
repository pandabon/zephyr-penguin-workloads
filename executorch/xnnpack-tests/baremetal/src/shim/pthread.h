// SPDX-License-Identifier: Apache-2.0
//
// A single-hart, no-thread <pthread.h> for the bare-metal build (picolibc has none).
// XNNPACK (src/mutex.c) and pthreadpool include <pthread.h> and call mutex / condition-variable
// functions; with one hart and one thread (pthreadpool with a thread count of 1 starts no workers)
// every lock is trivially free. pthread_create() fails with EAGAIN, so a request for worker threads
// is refused instead of silently doing nothing. Written for this project (no upstream code).
#pragma once
#include <errno.h>
#include <stddef.h>
#include <stdint.h>

typedef struct { int locked; } pthread_mutex_t;
typedef struct { int unused; } pthread_cond_t;
typedef struct { int unused; } pthread_attr_t;
typedef struct { int unused; } pthread_mutexattr_t;
typedef struct { int unused; } pthread_condattr_t;
typedef int pthread_t;

static inline int pthread_mutex_init(pthread_mutex_t* m, const pthread_mutexattr_t*) { m->locked = 0; return 0; }
static inline int pthread_mutex_destroy(pthread_mutex_t*) { return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t* m) { m->locked = 1; return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t* m) { m->locked = 0; return 0; }
static inline int pthread_cond_init(pthread_cond_t*, const pthread_condattr_t*) { return 0; }
static inline int pthread_cond_destroy(pthread_cond_t*) { return 0; }
static inline int pthread_cond_signal(pthread_cond_t*) { return 0; }
static inline int pthread_cond_broadcast(pthread_cond_t*) { return 0; }
// Waiting would block forever with no other thread to signal: report it as an error.
static inline int pthread_cond_wait(pthread_cond_t*, pthread_mutex_t*) { return EDEADLK; }
static inline int pthread_create(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*) { return EAGAIN; }
static inline int pthread_join(pthread_t, void**) { return ESRCH; }

// pthreadpool's busy-wait loops call sched_yield(); with one thread there is nobody to yield to.
static inline int sched_yield(void) { return 0; }

// pthread_once / static initializers (cpuinfo's init.c, XNNPACK's init-once.h).
typedef int pthread_once_t;
#define PTHREAD_ONCE_INIT 0
#define PTHREAD_MUTEX_INITIALIZER { 0 }
#define PTHREAD_COND_INITIALIZER { 0 }
static inline int pthread_once(pthread_once_t* once, void (*fn)(void)) {
  if (*once == 0) { *once = 1; fn(); }
  return 0;
}
