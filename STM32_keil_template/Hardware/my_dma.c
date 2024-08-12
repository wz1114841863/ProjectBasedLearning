#include "my_dma.h"

void my_dma_init(uint32_t addr_a, uint32_t addr_b, uint16_t size) {
	// 开启时钟
	RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
	// 结构体初始化
	DMA_InitTypeDef dma_init_stru;
	dma_init_stru.DMA_BufferSize = size;  // 传输计数器
	dma_init_stru.DMA_DIR = DMA_DIR_PeripheralSRC; // 传输方向
	dma_init_stru.DMA_Mode = DMA_Mode_Normal; // 是否重装
	dma_init_stru.DMA_M2M = DMA_M2M_Enable;  // 选择触发方式
	
	dma_init_stru.DMA_MemoryBaseAddr = addr_b;  // 外设配置
	dma_init_stru.DMA_MemoryDataSize = DMA_PeripheralDataSize_Byte;  // 传输位数
	dma_init_stru.DMA_MemoryInc = DMA_PeripheralInc_Disable;  // 是否自增
	
	dma_init_stru.DMA_PeripheralBaseAddr = addr_a;  // 寄存器配置
	dma_init_stru.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte;
	dma_init_stru.DMA_PeripheralInc = DMA_PeripheralInc_Enable;
	
	dma_init_stru.DMA_Priority = DMA_Priority_High;  // 优先级
	DMA_Init(DMA1_Channel1, &dma_init_stru);
}

void my_dma_transfer(uint16_t size) {
	DMA_Cmd(DMA1_Channel1, DISABLE);  
	DMA_SetCurrDataCounter(DMA1_Channel1, size);  // 循环次数
	DMA_Cmd(DMA1_Channel1, ENABLE);
	
	while(DMA_GetFlagStatus(DMA1_FLAG_TC1) == RESET);  // 获取状态标志位 
	DMA_ClearFlag(DMA1_FLAG_TC1);
}
