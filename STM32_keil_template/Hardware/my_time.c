#include "my_time.h"

// 定时中断
void my_time2_init(void) {
	// 开启定时器2时钟
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE);
	// 配置定时器2时钟源、时基单元
	TIM_InternalClockConfig(TIM2);
	TIM_TimeBaseInitTypeDef tim2_stru;
	tim2_stru.TIM_ClockDivision = TIM_CKD_DIV1;
	tim2_stru.TIM_CounterMode = TIM_CounterMode_Up;
	tim2_stru.TIM_Period = 10000 - 1;  // 重装寄存器
	tim2_stru.TIM_Prescaler = 7200 - 1;  // 预分频器
	tim2_stru.TIM_RepetitionCounter = 0;
	TIM_TimeBaseInit(TIM2, &tim2_stru);
	
	// 先清除一次状态寄存器中的更新标志位
	TIM_ClearFlag(TIM2, TIM_FLAG_Update);
	// 中断输出控制使能
	TIM_ITConfig(TIM2, TIM_IT_Update, ENABLE);
	
	// 配置NVIC
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	NVIC_InitTypeDef nvic_stru;
	nvic_stru.NVIC_IRQChannel = TIM2_IRQn;
	nvic_stru.NVIC_IRQChannelCmd = ENABLE;
	nvic_stru.NVIC_IRQChannelPreemptionPriority = 1;
	nvic_stru.NVIC_IRQChannelSubPriority = 1;
	NVIC_Init(&nvic_stru);
	
	// 开启定时器
	TIM_Cmd(TIM2, ENABLE);
};


// 输出比较
void my_time2_OC_init(void) {
	// 开启定时器2时钟
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE);
	// 配置定时器2时钟源、时基单元
	TIM_InternalClockConfig(TIM2);
	TIM_TimeBaseInitTypeDef tim2_stru;
	tim2_stru.TIM_ClockDivision = TIM_CKD_DIV1;
	tim2_stru.TIM_CounterMode = TIM_CounterMode_Up;
	tim2_stru.TIM_Period = 1000 - 1;  // 重装寄存器
	tim2_stru.TIM_Prescaler = 720 - 1;  // 预分频器
	tim2_stru.TIM_RepetitionCounter = 0;
	TIM_TimeBaseInit(TIM2, &tim2_stru);

	// 配置GPIO端口
	my_gpioA_init(GPIO_Pin_0, GPIO_Mode_AF_PP);
	
	// 配置输出比较单元
	TIM_OCInitTypeDef time_oc_init;
	TIM_OCStructInit(&time_oc_init);
	time_oc_init.TIM_OCMode = TIM_OCMode_PWM1;
	time_oc_init.TIM_OCPolarity = TIM_OCPolarity_High;
	time_oc_init.TIM_OutputState = TIM_OutputState_Enable;
	time_oc_init.TIM_Pulse = 500;	// CCR
	TIM_OC1Init(TIM2, &time_oc_init);
	// 开启定时器
	TIM_Cmd(TIM2, ENABLE);
	
};

// 输出比较，但是复用引脚
void my_time2_OC_AFIO_init(void) {
	// 开启定时器2时钟
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE);
	// 配置定时器2时钟源、时基单元
	TIM_InternalClockConfig(TIM2);
	TIM_TimeBaseInitTypeDef tim2_stru;
	tim2_stru.TIM_ClockDivision = TIM_CKD_DIV1;
	tim2_stru.TIM_CounterMode = TIM_CounterMode_Up;
	tim2_stru.TIM_Period = 1000 - 1;  // 重装寄存器
	tim2_stru.TIM_Prescaler = 720 - 1;  // 预分频器
	tim2_stru.TIM_RepetitionCounter = 0;
	TIM_TimeBaseInit(TIM2, &tim2_stru);

	// 配置GPIO端口
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO, ENABLE);
	// AFIO 引脚重映射, 将引脚PA0映射为PA15，并关闭PA15的原功能
	GPIO_PinRemapConfig(GPIO_PartialRemap1_TIM2, ENABLE);
	GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);
	my_gpioA_init(GPIO_Pin_15, GPIO_Mode_AF_PP);
	
	// 配置输出比较单元
	TIM_OCInitTypeDef time_oc_init;
	TIM_OCStructInit(&time_oc_init);
	time_oc_init.TIM_OCMode = TIM_OCMode_PWM1;
	time_oc_init.TIM_OCPolarity = TIM_OCPolarity_High;
	time_oc_init.TIM_OutputState = TIM_OutputState_Enable;
	time_oc_init.TIM_Pulse = 500;	// CCR
	TIM_OC1Init(TIM2, &time_oc_init);
	// 开启定时器
	TIM_Cmd(TIM2, ENABLE);
	
};

// 定时器三通道一输入捕获
void my_time3_ic_init(void) {
	// 开启时钟
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE);  // 接收PWM
	// 配置定时器时钟源，时基单元
	TIM_InternalClockConfig(TIM3);
	TIM_TimeBaseInitTypeDef tim3_init_stru;
	tim3_init_stru.TIM_ClockDivision = TIM_CKD_DIV1;
	tim3_init_stru.TIM_CounterMode = TIM_CounterMode_Up;
	tim3_init_stru.TIM_Period = (uint16_t)(166536 - 1);  // ARR
	tim3_init_stru.TIM_Prescaler = 72 - 1;  // PSC, 计数标准频率
	tim3_init_stru.TIM_RepetitionCounter = 0; 
	TIM_TimeBaseInit(TIM3, &tim3_init_stru);
	// 配置输入捕获模块
	TIM_ICInitTypeDef time3_ic_init;
	TIM_ICStructInit(&time3_ic_init);
	time3_ic_init.TIM_Channel = TIM_Channel_1;
	time3_ic_init.TIM_ICFilter = 0xF;
	time3_ic_init.TIM_ICPolarity = TIM_ICPolarity_Rising;
	time3_ic_init.TIM_ICPrescaler = TIM_ICPSC_DIV1;
	time3_ic_init.TIM_ICSelection = TIM_ICSelection_DirectTI;
 	TIM_ICInit(TIM3, &time3_ic_init);
	// 配置触发源和从模式
	TIM_SelectInputTrigger(TIM3, TIM_TS_TI1FP1);
	TIM_SelectSlaveMode(TIM3, TIM_SlaveMode_Reset);  // 从模式选择复位
													 // 即TI1产生上升沿时，会触发CNT归零
	// 配置GPIO端口
	my_gpioA_init(GPIO_Pin_6, GPIO_Mode_IPU);
	
	TIM_Cmd(TIM3, ENABLE);
}

// PWMI模式
// 定时器三通道一输入捕获
void my_time3_pwmi_init(void) {
	// 开启时钟
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE);  // 接收PWM
	// 配置定时器时钟源，时基单元
	TIM_InternalClockConfig(TIM3);
	TIM_TimeBaseInitTypeDef tim3_init_stru;
	tim3_init_stru.TIM_ClockDivision = TIM_CKD_DIV1;
	tim3_init_stru.TIM_CounterMode = TIM_CounterMode_Up;
	tim3_init_stru.TIM_Period = (uint16_t)(166536 - 1);  // ARR
	tim3_init_stru.TIM_Prescaler = 72 - 1;  // PSC, 计数标准频率
	tim3_init_stru.TIM_RepetitionCounter = 0; 
	TIM_TimeBaseInit(TIM3, &tim3_init_stru);
	// 配置输入捕获模块
	TIM_ICInitTypeDef time3_ic_init;
	TIM_ICStructInit(&time3_ic_init);
	time3_ic_init.TIM_Channel = TIM_Channel_1;
	time3_ic_init.TIM_ICFilter = 0xF;
	time3_ic_init.TIM_ICPolarity = TIM_ICPolarity_Rising;
	time3_ic_init.TIM_ICPrescaler = TIM_ICPSC_DIV1;
	time3_ic_init.TIM_ICSelection = TIM_ICSelection_DirectTI;
 	TIM_ICInit(TIM3, &time3_ic_init);
	// PWMIm模式配置
	TIM_PWMIConfig(TIM3, &time3_ic_init);
	//	tim_ic_init.TIM_Channel = TIM_Channel_2;
	//	tim_ic_init.TIM_ICFilter = 0xF;
	//	tim_ic_init.TIM_ICPolarity = TIM_ICPolarity_Falling;
	//	tim_ic_init.TIM_ICPrescaler = TIM_ICPSC_DIV1;
	//	tim_ic_init.TIM_ICSelection = TIM_ICSelection_IndirectTI;
	// 	TIM_ICInit(TIM3, &tim_ic_init);
	// 配置触发源和从模式
	TIM_SelectInputTrigger(TIM3, TIM_TS_TI1FP1);
	TIM_SelectSlaveMode(TIM3, TIM_SlaveMode_Reset);
	// 配置GPIO端口
	my_gpioA_init(GPIO_Pin_6, GPIO_Mode_IPU);  // 使用一个引脚
	TIM_Cmd(TIM3, ENABLE);
}