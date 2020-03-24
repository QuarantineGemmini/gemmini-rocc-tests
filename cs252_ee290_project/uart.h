// See LICENSE for license details.

#ifndef __UART_H__
#define __UART_H__

#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>

#define UART_ADDR_TX_BYTE 0x54000000
#define UART_ADDR_TX_EN   0x54000008
#define UART_ADDR_DIV     0x54000018

void uart_write(const char *str, const size_t len);
int kprintf(const char* fmt, ...);

#endif //__UART_H__
