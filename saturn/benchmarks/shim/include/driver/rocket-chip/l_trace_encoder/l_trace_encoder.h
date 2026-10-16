/* No-op stub of baremetal-ide's l_trace_encoder.h (outside generators/saturn; not on Penguin).
   Only vec-mx-axpy actually calls it; the other benchmarks merely #include it. */
#ifndef L_TRACE_ENCODER_STUB_H
#define L_TRACE_ENCODER_STUB_H
typedef struct { int dummy; } LTraceEncoderType;
#define BRANCH_MODE_TARGET 0
static inline unsigned long get_hart_id(void){ return 0; }
static inline LTraceEncoderType *l_trace_encoder_get(unsigned long h){ static LTraceEncoderType e; (void)h; return &e; }
static inline void l_trace_encoder_configure_branch_mode(LTraceEncoderType *e,int m){(void)e;(void)m;}
static inline void l_trace_encoder_start(LTraceEncoderType *e){(void)e;}
static inline void l_trace_encoder_stop(LTraceEncoderType *e){(void)e;}
#endif
