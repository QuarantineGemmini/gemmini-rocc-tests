// See LICENSE for license details.

#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>
#include <stdlib.h>

#define MAX(a,b)                  \
  ({ __typeof__ (a) _a = (a);     \
      __typeof__ (b) _b = (b);    \
    _a > _b ? _a : _b; })
    
#define MIN(a,b)                  \
  ({ __typeof__ (a) _a = (a);     \
      __typeof__ (b) _b = (b);    \
    _a < _b ? _a : _b; })

#define PRINT(fmt, ...)            \
  do {                             \
    printf(fmt, ##__VA_ARGS__);    \
    printf("\n");                  \
  } while(0)

#ifndef NODEBUG
#define DEBUG(fmt, ...)            \
  PRINT(fmt, ##__VA_ARGS__)

#define ERROR(fmt, ...)            \
  do {                             \
    PRINT(fmt, ##__VA_ARGS__);     \
    exit(1);                       \
  } while(0)

#define ASSERT(predicate, fmt, ...)\
  do {                             \
    if (!(predicate)) {            \
      ERROR(fmt, ##__VA_ARGS__);   \
    }                              \
  } while(0)

#else 
#define DEBUG(fmt, ...) 
#define ERROR(fmt, ...)
#define ASSERT(predicate, fmt, ...)
#endif // NODEBUG

#endif // __UTIL_H__
