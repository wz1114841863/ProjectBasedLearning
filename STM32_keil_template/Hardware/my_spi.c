#include "my_spi.h"

void my_spi_w_cs(uint8_t bit_value) {
	GPIO_WriteBit(GPIOA, GPIO_Pin_4, (BitAction)bit_value);
}

void my_spi_init(void) {
	// 开启时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_SPI1, ENABLE);
	// GPIO初始化
	my_gpioA_init(GPIO_Pin_4, GPIO_Mode_Out_PP);
	my_gpioA_init(GPIO_Pin_5 | GPIO_Pin_7, GPIO_Mode_AF_PP);
	my_gpioA_init(GPIO_Pin_6, GPIO_Mode_IPU);
	
	// SPI初始化
	SPI_InitTypeDef spi_init_stru;
	spi_init_stru.SPI_Mode = SPI_Mode_Master;  // SPI主模式
	spi_init_stru.SPI_Direction = SPI_Direction_2Lines_FullDuplex;  // 方向，选择两线全双工
	spi_init_stru.SPI_DataSize = SPI_DataSize_8b;  // 数据宽度，8位数据位
	spi_init_stru.SPI_FirstBit = SPI_FirstBit_MSB;  // 先行位，选择高位先行
	spi_init_stru.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_128;  // 波特率分频，选择128分频
	
	// 极性和相位决定选择SPI模式0
	spi_init_stru.SPI_CPHA = SPI_CPHA_1Edge; // SPI相位，选择第一个时钟边沿采样，
	spi_init_stru.SPI_CPOL = SPI_CPOL_Low;  // SPI极性，选择低极性

	
	spi_init_stru.SPI_NSS = SPI_NSS_Soft;  //NSS，选择由软件控制;
	spi_init_stru.SPI_CRCPolynomial = 7;  // CRC校验值
	
	// 结构体初始化
	SPI_Init(SPI1, &spi_init_stru);  
		
	// SPI使能
	SPI_Cmd(SPI1, ENABLE);
	my_spi_w_cs(1);
}

void my_spi_start(void) {
	// 起始信号
	my_spi_w_cs(0);
}

void my_spi_stop(void) {
	// 终止信号
	my_spi_w_cs(1);
}

uint8_t my_spi_swap_byte(uint8_t byte_send) {
	// 等待发送数据寄存器空
	while(SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) != SET);
	// 写入数据到发送寄存器
	SPI_I2S_SendData(SPI1, byte_send);
	// 等待接收数据寄存器非空
	while(SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) != SET);
	return SPI_I2S_ReceiveData(SPI1);
}
