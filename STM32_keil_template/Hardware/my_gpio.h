#ifndef __MY_GPIO_H
#define __MY_GPIO_H

#include "stm32f10x.h"

void my_gpioA_init(uint16_t GPIO_Pin, GPIOMode_TypeDef GPIO_Mode);
void my_gpioB_init(uint16_t GPIO_Pin, GPIOMode_TypeDef GPIO_Mode);
void my_gpioC_init(uint16_t GPIO_Pin, GPIOMode_TypeDef GPIO_Mode);

#endif
