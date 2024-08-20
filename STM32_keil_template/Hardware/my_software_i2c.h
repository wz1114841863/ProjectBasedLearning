#ifndef __MY_SOFTWARE_I2C_H__
#define __MY_SOFTWARE_I2C_H__

#include "stm32f10x.h"
#include "my_gpio.h"
#include "my_delay.h"

void my_soft_i2c_w_scl(uint8_t bit_value);
void my_soft_i2c_w_sda(uint8_t bit_value);
uint8_t my_soft_i2c_r_sda(void);
void my_soft_i2c_init(void);
void my_soft_i2c_start(void);
void my_soft_i2c_stop(void);
void my_soft_i2c_send_byte(uint8_t data);
uint8_t my_soft_i2c_recv_byte(void);
void my_soft_i2c_send_ack(uint8_t ack);
uint8_t my_soft_i2c_recv_ack(void);

#endif