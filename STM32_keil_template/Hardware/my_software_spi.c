#include "my_software_spi.h"

void my_soft_spi_w_cs(uint8_t bit_value) {
	// 片选信号
	GPIO_WriteBit(GPIOA, GPIO_Pin_4, (BitAction)bit_value);
}

void my_soft_spi_w_sck(uint8_t bit_value) {
	// 设施SCK引脚电平
	GPIO_WriteBit(GPIOA, GPIO_Pin_5, (BitAction)bit_value);
}

void my_soft_w_mosi(uint8_t bit_value) {
	// 写mosi引脚
	GPIO_WriteBit(GPIOA, GPIO_Pin_7, (BitAction)bit_value);
}


uint8_t my_soft_r_miso(void) {
	// 读miso引脚
	return GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_6);
}

