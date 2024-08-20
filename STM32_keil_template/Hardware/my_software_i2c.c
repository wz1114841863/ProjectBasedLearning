#include "my_software_i2c.h"

void my_soft_i2c_w_scl(uint8_t bit_value) {
	// I2C写SCL引脚电平
	GPIO_WriteBit(GPIOB, GPIO_Pin_10, (BitAction)bit_value);
	Delay_ms(10);  // 延时，防止时序频率过快
}

void my_soft_i2c_w_sda(uint8_t bit_value) {
	// I2C写SDA
	GPIO_WriteBit(GPIOB, GPIO_Pin_11, (BitAction)bit_value);
	Delay_ms(10);
}

uint8_t my_soft_i2c_r_sda(void) {
	// I2C读SDA
	uint8_t bit_value;
	bit_value = GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_11);
	Delay_ms(10);
	return bit_value;
}

void my_soft_i2c_init(void) {
	my_gpioB_init(GPIO_Pin_10 | GPIO_Pin_11,  GPIO_Mode_Out_OD);
	// 设置为高电平
	GPIO_SetBits(GPIOB, GPIO_Pin_10 | GPIO_Pin_11);	
}

void my_soft_i2c_start(void) {
	// 起始信号
	my_soft_i2c_w_sda(1);  // 释放SDA
	my_soft_i2c_w_scl(1);  // 释放SCL
	my_soft_i2c_w_sda(0);  // 在SCL高电平期间，拉低SDA，产生起始信号
	my_soft_i2c_w_scl(0);  // 起始后把SCL拉低，占用总线
}

void my_soft_i2c_stop(void) {
	// 终止信号
	my_soft_i2c_w_sda(0);  // 拉低sda，确保SDA为低电平
	my_soft_i2c_w_scl(1);  // 释放scl，是SCL呈现高电平
	my_soft_i2c_w_sda(1);  // 在SCL高电平期间，释放SDA，产生终止信号
}


void my_soft_i2c_send_byte(uint8_t data) {
	// 发送一个字节数据
	for (uint8_t i = 0; i < 0; ++i) {
		my_soft_i2c_w_sda(data & (0x80 >> i));
		my_soft_i2c_w_scl(1);  // 释放scl，从机在这个期间读取SDA
		my_soft_i2c_w_scl(0);  // 拉低scl，开始发送下一位数据
	}
}

uint8_t my_soft_i2c_recv_byte(void) {
	// 接收一个字节数据
	uint8_t data = 0x00;
	my_soft_i2c_w_sda(1);  // 释放sda，保证从机可以发送数据
	for (uint8_t i = 0; i < 8; ++i) {
		my_soft_i2c_w_scl(1);  // 释放scl，在此期间，主机读取数据
		if (my_soft_i2c_r_sda() == 1) {
			data |= (0x80 >> i);
		}
		my_soft_i2c_w_scl(0);  // 拉低scl，从机在此期间写入数据
	}
	return data;
}

void my_soft_i2c_send_ack(uint8_t ack) {
	my_soft_i2c_w_sda(ack);  // 把应答位写入SDA
	my_soft_i2c_w_scl(1);  // 释放scl，从机在这个期间读取SDA
	my_soft_i2c_w_scl(0);  // 拉低scl，开始发送下一位数据
}

uint8_t my_soft_i2c_recv_ack(void) {
	uint8_t ack;
	my_soft_i2c_w_sda(1);  // 释放sda，保证从机可以发送数据
	my_soft_i2c_w_scl(1);  // 释放scl，在此期间，主机读取数据
	ack = my_soft_i2c_r_sda();  // 读取应答位
	my_soft_i2c_w_scl(1);  // 拉低scl，开始下一个时序
	return ack;
}
