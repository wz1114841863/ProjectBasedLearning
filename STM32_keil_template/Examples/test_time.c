#include "test_time.h"

void time_test(void) {
	my_gpioC_init(GPIO_Pin_13, GPIO_Mode_Out_PP);
	GPIO_ResetBits(GPIOC, GPIO_Pin_13);
	my_time2_init();
}
	
void TIM2_IRQHandler(void) {
	// 检查并清楚中断标志位
	if (TIM_GetITStatus(TIM2, TIM_IT_Update) == SET) {
		if (GPIO_ReadOutputDataBit(GPIOC, GPIO_Pin_13) == 0) {
			GPIO_SetBits(GPIOC, GPIO_Pin_13);
		}else {                                                                          
			GPIO_ResetBits(GPIOC, GPIO_Pin_13);
		}
		TIM_ClearITPendingBit(TIM2, TIM_IT_Update);
	}
}
