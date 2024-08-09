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

void time_oc_test(void) {
	my_time2_OC_init();
	while(1) {
		for (int i = 0; i <= 100; ++i) {
			TIM_SetCompare1(TIM2, i);
			Delay_ms(10);
		}
	}
}

void time_oc_afio_test(void) {
	my_time2_OC_AFIO_init();
	while(1) {
		for (int i = 0; i <= 100; ++i) {
			TIM_SetCompare1(TIM2, i);
			Delay_ms(10);
		}
	}
}

void time_ic_test(void) {
	my_time2_OC_init();
	my_time3_ic_init();
}

uint16_t IC_Get_Freq(void) {
	return (uint16_t)(10000 / (TIM_GetCapture1(TIM3) + 1));
}

void time_pwmi_test(void) {
	my_time2_OC_init();
	my_time3_pwmi_init();
}

uint32_t IC_GetDuty(void) {
	return (TIM_GetCapture2(TIM3) + 1) * 100 / (TIM_GetCapture1(TIM3) + 1);
}
