#ifndef __MY_UART_H
#define __MY_UART_H

#include "stm32f10x.h"
#include "my_gpio.h"

extern uint8_t UART_Rx_Data;
extern uint8_t UART_Rx_Flag;

void my_uart1_init(void);
void uart_send_byte(uint8_t data);
void uart_send_array(uint8_t* array, uint16_t length);
void uart_send_string(char* str);
uint32_t uart_pow(uint32_t X, uint32_t Y);
void uart_send_num(uint8_t num, uint16_t length);
uint8_t uart_get_rx_flag(void);
uint8_t uart_get_rx_data(void);
void USART1_IRQHandler(void);

#endif