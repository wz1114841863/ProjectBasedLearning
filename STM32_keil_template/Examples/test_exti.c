#include "test_exti.h"

// 中断函数名唯一，不需要自己去定义
void EXTI15_10_IRQHandler(void) {
	// 检测触发中断的引脚
	if (EXTI_GetITStatus(EXTI_Line13) == SET) {
		if (GPIO_ReadOutputDataBit(GPIOC, GPIO_Pin_13) == 0) {
			GPIO_SetBits(GPIOC, GPIO_Pin_13);
		}else {
			GPIO_ResetBits(GPIOC, GPIO_Pin_13);
		}	 	
		EXTI_ClearITPendingBit(EXTI_Line13);
	}
}

void exti_test(void) {
	// 开启GPIOC，pin13， 通过外部中断改别引脚输出
	my_gpioC_init(GPIO_Pin_13, GPIO_Mode_Out_PP);
	GPIO_SetBits(GPIOC, GPIO_Pin_13);
	// 初始化exti
	my_exti_init();
}

