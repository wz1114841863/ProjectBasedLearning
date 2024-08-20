#ifndef __MY_SPI_H
#define __MY_SPI_H

#include "stm32f10x.h"
#include "my_gpio.h"

void my_spi_w_cs(uint8_t bit_value);
void my_spi_init(void);
void my_spi_start(void);
void my_spi_stop(void);
uint8_t my_spi_swap_byte(uint8_t byte_send);

#endif
