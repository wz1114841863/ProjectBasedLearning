#include "stm32f10x.h"                  // Device header
#include "test_oled.h"

int main(void) {
	
	my_gpioC_init(GPIO_Pin_13, GPIO_Mode_Out_PP);
	GPIO_WriteBit(GPIOC, GPIO_Pin_13, (BitAction)(0));
	
	oled_test();
	while (1) {

	}
}


