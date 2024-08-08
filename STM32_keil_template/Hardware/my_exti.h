#ifndef __MY_EXTI_H
#define __MY_EXTI_H

#include "stm32f10x.h"
#include "my_gpio.h"

void my_exti_init();
// 需要去实现对应的中断函数定义
// void EXTI15_10_IRQHandler(void);

#endif
