#ifndef __GET_REAL_TIME_H__
#define __GET_REAL_TIME_H__

#include <unistd.h>	  // POSIX flags
#include <time.h>	    // clock_gettime(), time()
#include <sys/time.h>	// gethrtime(), gettimeofday()

// Returns the real time, in seconds, or -1.0 if an error occurred.
// 
// Time is measured since an arbitrary and OS-dependent start time.
// The returned real time is only useful for computing an elapsed time
// between two calls to this function.
double get_real_time() {
  struct timeval tm;
  gettimeofday(&tm, NULL);
  return (double)tm.tv_sec + (double)tm.tv_usec / 1000000.0;
}

#endif // __GET_REAL_TIME_H__
