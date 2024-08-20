#include "my_oled.h"
#include "my_oled_font.h"

void my_oled_w_scl(uint8_t bit_value) {
	// 写 SCL 电平
	GPIO_WriteBit(GPIOB, GPIO_Pin_8, (BitAction)(bit_value));
	// Delay_ms(10);
}

void my_oled_w_sda(uint8_t bit_value) {
	// 写 SDA 电平
	GPIO_WriteBit(GPIOB, GPIO_Pin_9, (BitAction)(bit_value));
	// Delay_ms(10);
}

void my_oled_i2c_init(void) {
	my_gpioB_init(GPIO_Pin_8 | GPIO_Pin_9, GPIO_Mode_Out_OD);

	my_oled_w_scl(1);
	my_oled_w_sda(1);
}

void my_oled_i2c_start(void) {
	// 起始信号
	my_oled_w_sda(1);
	my_oled_w_scl(1);
	my_oled_w_sda(0);
	my_oled_w_scl(0);
}


void my_oled_i2c_stop(void) {
	// 终止信号
	my_oled_w_sda(0);
	my_oled_w_scl(1);
	my_oled_w_sda(1);
}

void my_oled_i2c_send_byte(uint8_t data) {
	// 发送一个字节的数据
	for (uint8_t i = 0; i < 8; ++i) {
		my_oled_w_sda(data & (0x80 >> i));
		my_oled_w_scl(1);
		my_oled_w_scl(0);
	}
	// 额外的一个时钟，不处理应答
	my_oled_w_scl(1);
	my_oled_w_scl(0);	
}

void my_oled_w_command(uint8_t command) {
	// 发送命令
	my_oled_i2c_start();
	my_oled_i2c_send_byte(0x78);  // 从机地址
	my_oled_i2c_send_byte(0x00);  // 写命令
	my_oled_i2c_send_byte(command);
	my_oled_i2c_stop();
}

void my_oled_w_data(uint32_t data) {
	// 发送数据
	my_oled_i2c_start();
	my_oled_i2c_send_byte(0x78);  // 从机地址
	my_oled_i2c_send_byte(0x40);  // 写命令
	my_oled_i2c_send_byte(data);
	my_oled_i2c_stop();	
}


void my_oled_set_cursor(uint8_t Y, uint8_t X) {
	// 设置光标位置，左上角为原点
	my_oled_w_command(0xB0 | Y);  // 设置Y位置
	my_oled_w_command(0x10 | ((X & 0xF0) >> 4));  // 设置X位置高4位
	my_oled_w_command(0x00 | (X & 0x0F));  // 设置X位置低4位
}
	
void my_oled_clear(void) {
	// 清屏
	for (uint8_t j = 0; j < 8; ++j) {
		my_oled_set_cursor(j, 0);
		for (uint8_t i = 0; i < 128; ++i) {
			my_oled_w_data(0x00);
		}
	}
}

void my_oled_show_char(uint8_t line, uint8_t column, char c) {
	// 显示一个字符
	uint8_t i;
	// 设置光标位置在上半部分
	my_oled_set_cursor((line - 1) * 2, (column - 1) * 8);
	// 显示上半部分内容
	for (i = 0; i < 8; ++i) {
		my_oled_w_data(OLED_F8x16[c - ' '][i]);
	}
	// 设置光标位置在下半部分
	my_oled_set_cursor((line - 1) * 2 + 1, (column - 1) * 8);
	// 显示下半部分内容
	for (i = 0; i < 8; ++i) {
		my_oled_w_data(OLED_F8x16[c - ' '][i + 8]);
	}	
}

void my_oled_show_string(uint8_t line, uint8_t column, char* str) {
	// 显示字符串
	for (uint8_t i = 0; str[i] != '\0'; ++i) {
		my_oled_show_char(line, column, str[i]);
	}
}


uint32_t my_oled_pow(uint32_t X, uint32_t Y) {
	// 次方函数
	uint32_t res = 1;
	while (Y--) {
		res *= X;
	}
	return res;
}

void my_oled_show_num(uint8_t line, uint8_t column, uint32_t num, uint8_t len) {
	// 显示无符号数
	for (uint8_t i = 0; i < len; ++i) {
		my_oled_show_char(line, column + i, num / my_oled_pow(10, len - i - 1) % 10 + '0');
	}
}

void my_oled_show_signed_num(uint8_t line, uint8_t column, int32_t num, uint8_t len) {
	// 显示有符号数
	uint32_t tmp_num;
	if (num > 0) {
		my_oled_show_char(line, column, '+');
		tmp_num = num;
	}else  {
		my_oled_show_char(line, column, '-');
		tmp_num = -num;
	}
	
	for (uint8_t i = 0; i < len; ++i) {
		my_oled_show_char(line, column + i + 1, tmp_num / my_oled_pow(10, len - i - 1) % 10 + '0');
	}
}

void my_oled_show_hex_num(uint8_t line, uint8_t column, uint32_t num, uint8_t len) {
	// 显示十六进制数
	uint8_t i, single_num;
	for (i = 0; i < len; ++i) {
		single_num = num / my_oled_pow(16, len - i - 1) % 16;
		if (single_num < 10) {
			my_oled_show_char(line, column + i, single_num + '0');
		}else {
			my_oled_show_char(line, column + i, single_num - 10 + 'A');
		}
	}
}

void my_oled_show_bin_num(uint8_t line, uint8_t column, uint32_t num, uint8_t len) {
	// 显示二进制数
	for (uint8_t i = 0; i < len; ++i) {
		my_oled_show_char(line, column + i, num / my_oled_pow(2, len - i - 1) % 2 + '0');
	}
}

void my_oled_init(void) {
	// oled初始化
	uint32_t i, j;
	for (i = 0; i < 1000; ++i) {  //上电延时
		for (j = 0; j < 1000; ++j);
	}
	my_oled_i2c_init();  // 端口初始化
	
	my_oled_w_command(0xAE);	//关闭显示
	
	my_oled_w_command(0xD5);	//设置显示时钟分频比/振荡器频率
	my_oled_w_command(0x80);
	
	my_oled_w_command(0xA8);	//设置多路复用率
	my_oled_w_command(0x3F);
	
	my_oled_w_command(0xD3);	//设置显示偏移
	my_oled_w_command(0x00);
	
	my_oled_w_command(0x40);	//设置显示开始行
	
	my_oled_w_command(0xA1);	//设置左右方向，0xA1正常 0xA0左右反置
	
	my_oled_w_command(0xC8);	//设置上下方向，0xC8正常 0xC0上下反置

	my_oled_w_command(0xDA);	//设置COM引脚硬件配置
	my_oled_w_command(0x12);
	
	my_oled_w_command(0x81);	//设置对比度控制
	my_oled_w_command(0xCF);

	my_oled_w_command(0xD9);	//设置预充电周期
	my_oled_w_command(0xF1);

	my_oled_w_command(0xDB);	//设置VCOMH取消选择级别
	my_oled_w_command(0x30);

	my_oled_w_command(0xA4);	//设置整个显示打开/关闭

	my_oled_w_command(0xA6);	//设置正常/倒转显示

	my_oled_w_command(0x8D);	//设置充电泵
	my_oled_w_command(0x14);

	my_oled_w_command(0xAF);	//开启显示
	
	my_oled_clear();
}

