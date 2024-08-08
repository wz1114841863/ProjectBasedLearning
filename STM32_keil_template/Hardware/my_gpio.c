#include "my_gpio.h"


void my_gpioA_init(uint16_t GPIO_Pin, GPIOMode_TypeDef GPIO_Mode) {
	// 开启GPIOA时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	
	GPIO_InitTypeDef gpioa_init;
	gpioa_init.GPIO_Mode = GPIO_Mode;
	gpioa_init.GPIO_Pin = GPIO_Pin;
	gpioa_init.GPIO_Speed = GPIO_Speed_50MHz;
	
	// 使能gpio
	GPIO_Init(GPIOA, &gpioa_init);														
}

void my_gpioB_init(uint16_t GPIO_Pin, GPIOMode_TypeDef GPIO_Mode) {
	// 开启GPIOB时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	
	GPIO_InitTypeDef gpiob_init;
	gpiob_init.GPIO_Mode = GPIO_Mode;
	gpiob_init.GPIO_Pin = GPIO_Pin;
	gpiob_init.GPIO_Speed = GPIO_Speed_50MHz;
	
	// 使能gpio
	GPIO_Init(GPIOB, &gpiob_init);														
}

void my_gpioC_init(uint16_t GPIO_Pin, GPIOMode_TypeDef GPIO_Mode) {
	// 开启GPIOC时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);
	
	GPIO_InitTypeDef gpioc_init;
	gpioc_init.GPIO_Mode = GPIO_Mode;
	gpioc_init.GPIO_Pin = GPIO_Pin;
	gpioc_init.GPIO_Speed = GPIO_Speed_50MHz;
	
	// 使能gpio
	GPIO_Init(GPIOC, &gpioc_init);														
}
