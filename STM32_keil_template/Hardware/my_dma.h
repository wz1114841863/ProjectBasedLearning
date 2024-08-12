#ifndef __MY_DMA_H
#define __MY_DMA_H

#include "stm32f10x.h"

void my_dma_init(uint32_t addr_a, uint32_t addr_b, uint16_t size);
void my_dma_transfer(uint16_t size);

#endif
