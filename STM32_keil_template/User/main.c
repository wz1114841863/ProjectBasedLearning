#include "stm32f10x.h"                  // Device header
#include "test_adc.h"

int main(void) {
	adc_test();
	while (1) {
		adc_get_ADValue();
		Delay_ms(100);
	}
}


