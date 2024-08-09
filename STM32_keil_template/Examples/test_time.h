#ifndef __TEST_TIME_H
#define __TEST_TIME_H

#include "stm32f10x.h"
#include "my_time.h"
#include "my_gpio.h"

void time_test(void);
void time_oc_test(void);
void time_oc_afio_test(void);
void time_ic_test(void);
void time_pwmi_test(void);
uint32_t IC_GetDuty(void);
uint16_t IC_Get_Freq(void);

#endif