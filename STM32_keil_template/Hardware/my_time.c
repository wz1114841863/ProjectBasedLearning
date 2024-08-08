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
