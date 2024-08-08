#include "test_gpio.h"

void gpio_test() {
	// 测试GPIOA、B和C引脚输出
	my_gpioA_init(GPIO_Pin_10, GPIO_Mode_Out_PP);
	GPIO_ResetBits(GPIOA, GPIO_Pin_10);
	
	my_gpioB_init(GPIO_Pin_10, GPIO_Mode_Out_PP);
	GPIO_ResetBits(GPIOB, GPIO_Pin_10);
	
	my_gpioC_init(GPIO_Pin_13, GPIO_Mode_Out_PP);
	GPIO_ResetBits(GPIOC, GPIO_Pin_13);
}
