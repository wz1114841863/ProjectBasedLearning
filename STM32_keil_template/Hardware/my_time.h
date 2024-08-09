#ifndef __MY_TIME_H
#define __MY_TIME_H

#include "stm32f10x.h"
#include "my_gpio.h"
#include "my_delay.h"

void my_time2_init(void);
void my_time2_OC_init(void);
void my_time2_OC_AFIO_init(void);
void my_time3_ic_init(void);
void my_time3_pwmi_init(void);
	
#endif