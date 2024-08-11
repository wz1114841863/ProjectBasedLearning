#ifndef __MY_ADC_H
#define __MY_ADC_H

#include "stm32f10x.h"
#include "my_gpio.h"

void my_adc_software_init(void);
uint16_t ADC_GetValue(void);

#endif 

