#include "my_exti.h"

#define OPEN_GPIO() my_gpioB_init(GPIO_Pin_13, GPIO_Mode_IPU)
// 需要配置相应的GPIO端口为输入模式，这里通过宏定义配置
// 同时需要修改AFIO口的配置

void my_exti_init() {
	// 配置IO端口
	OPEN_GPIO();
	// 开启AFIO时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO, ENABLE);
	// 配置AFIO
	GPIO_EXTILineConfig(GPIO_PortSourceGPIOB, GPIO_PinSource13);
	
	// 配置EXTI
	EXTI_InitTypeDef exti_stru;
	exti_stru.EXTI_Line = EXTI_Line13;
	exti_stru.EXTI_LineCmd = ENABLE;
	exti_stru.EXTI_Mode = EXTI_Mode_Interrupt;
	exti_stru.EXTI_Trigger = EXTI_Trigger_Falling;
	EXTI_Init(&exti_stru);
	
	// 配置NVIC，NVIC不需要开启时钟
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	NVIC_InitTypeDef nvic_stru;
	nvic_stru.NVIC_IRQChannel = EXTI15_10_IRQn;
	nvic_stru.NVIC_IRQChannelCmd = ENABLE;
	nvic_stru.NVIC_IRQChannelPreemptionPriority = 1;
	nvic_stru.NVIC_IRQChannelSubPriority = 1;
	NVIC_Init(&nvic_stru);
}
