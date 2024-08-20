#include "test_oled.h"

void oled_test(void) {

	my_oled_init();  // oled初始化
	// 显示字符
	my_oled_show_char(1, 1, 'A');

	// 显示字符串
	my_oled_show_string(1, 3, "HelloWorld!");
	// 显示十进制数字
	my_oled_show_num(2, 1, 12345, 5);
	// 显示有符号十进制数字
	my_oled_show_signed_num(2, 7, -66, 2);
	// 显示十六进制数
	my_oled_show_hex_num(3, 1, 0xAA55, 4);
	// 显示二进制数
	my_oled_show_bin_num(4, 1, 0xAA55, 16);

}
