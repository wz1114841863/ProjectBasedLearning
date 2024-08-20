#ifndef __MY_SOFTWARE_SPI_H
#define __MY_SOFTWARE_SPI_H

#include "stm32f10x.h"
#include "my_gpio.h"

void my_soft_spi_w_cs(uint8_t bit_value);
void my_soft_spi_w_sck(uint8_t bit_value);
void my_soft_spi_w_mosi(uint8_t bit_value);
uint8_t my_soft_spi_r_miso(void);
void my_soft_spi_init(void);
void my_soft_spi_start(void);
void my_soft_spi_stop(void);
uint8_t my_soft_spi_swap_byte(uint8_t send_byte);

#endif
