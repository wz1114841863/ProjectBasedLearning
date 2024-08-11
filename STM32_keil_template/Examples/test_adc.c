#include "test_adc.h"

void adc_test(void) {
	my_adc_software_init();
}

float adc_get_ADValue(void) {
    uint16_t value = ADC_GetValue();  // 获取AD转换的值
    float voltage = (float)value / 4095 * 3.3;  // 将AD值线性变换到0~3.3的范围，表示电压
	return voltage;
}
