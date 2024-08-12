#ifndef __TEST_DMA_H
#define __TEST_DMA_H

#include "stm32f10x.h"
#include "my_adc.h"
#include "my_gpio.h"
#include "my_dma.h"

void dma_adc_test(uint32_t addr);
void adc_get_value(void);

#endif