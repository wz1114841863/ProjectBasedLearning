#include "my_software_spi.h"

void my_soft_spi_w_cs(uint8_t bit_value) {
	// 片选信号
	GPIO_WriteBit(GPIOA, GPIO_Pin_4, (BitAction)bit_value);
}

void my_soft_spi_w_sck(uint8_t bit_value) {
	// 设施SCK引脚电平
	GPIO_WriteBit(GPIOA, GPIO_Pin_5, (BitAction)bit_value);
}

void my_soft_spi_w_mosi(uint8_t bit_value) {
	// 写mosi引脚
	GPIO_WriteBit(GPIOA, GPIO_Pin_7, (BitAction)bit_value);
}


uint8_t my_soft_spi_r_miso(void) {
	// 读miso引脚
	return GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_6);
}

void my_soft_spi_init(void) {
	// 端口配置
	my_gpioA_init(GPIO_Pin_4 | GPIO_Pin_5 | GPIO_Pin_7, GPIO_Mode_Out_PP);
	my_gpioA_init(GPIO_Pin_6, GPIO_Mode_IPU);
	
	// 设置默认电平
	my_soft_spi_w_cs(1);
	my_soft_spi_w_sck(0);
}

void my_soft_spi_start(void) {
	// 片选信号置低
	my_soft_spi_w_cs(0);
}

void my_soft_spi_stop(void) {
	my_soft_spi_w_cs(1);
}

uint8_t my_soft_spi_swap_byte(uint8_t send_byte) {
	uint8_t recv_byte = 0x00;  // 定义接收的数据
	for (uint8_t i = 0; i < 8; ++i) {  // 循环八次，依次交换每一位数据
		my_soft_spi_w_mosi(send_byte & (0x80 >> i));  // 使用掩码的方式发送给指定一位数据
		my_soft_spi_w_sck(1);  // 拉高sck，上升沿移出数据
		if (my_soft_spi_r_miso() == 1) {
			recv_byte |= (0x80 >> i);  // 读取数据并存储
		}
		my_soft_spi_w_sck(0);  // 拉低sck，下降沿移入数据
	}
	return recv_byte;
}

