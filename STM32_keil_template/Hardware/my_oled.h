#ifndef __MY_OLED_H__
#define __MY_OLED_H__

#include "stm32f10x.h"
#include "my_gpio.h"
#include "my_delay.h"

void my_oled_w_scl(uint8_t bit_value);
void my_oled_w_sda(uint8_t bit_value);
void my_oled_i2c_init(void);
void my_oled_i2c_start(void);
void my_oled_i2c_stop(void);
void my_oled_i2c_send_byte(uint8_t data);
void my_oled_w_command(uint8_t command);
void my_oled_w_data(uint32_t data);
void my_oled_set_cursor(uint8_t Y, uint8_t X);
void my_oled_clear(void);
void my_oled_show_char(uint8_t line, uint8_t column, char c);
void my_oled_show_string(uint8_t line, uint8_t column, char* str);
uint32_t my_oled_pow(uint32_t X, uint32_t Y);
void my_oled_show_num(uint8_t line, uint8_t column, uint32_t num, uint8_t len);
void my_oled_show_signed_num(uint8_t line, uint8_t column, int32_t num, uint8_t len);
void my_oled_show_hex_num(uint8_t line, uint8_t column, uint32_t num, uint8_t len);
void my_oled_show_bin_num(uint8_t line, uint8_t column, uint32_t num, uint8_t len);
void my_oled_init(void);

#endif