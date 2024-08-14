#include "my_uart.h"

uint8_t UART_Rx_Data;		//定义串口接收的数据变量
uint8_t UART_Rx_Flag;		//定义串口接收的标志位变量

void my_uart1_init(void) {
	// 开启时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
	
	// GPIO 初始化
	my_gpioA_init(GPIO_Pin_9, GPIO_Mode_AF_PP);
	my_gpioA_init(GPIO_Pin_10, GPIO_Mode_IPU);
	
	// USART初始化
	USART_InitTypeDef usart_init;
	usart_init.USART_BaudRate =9600 ;
	usart_init.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	usart_init.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;
	usart_init.USART_Parity = USART_Parity_No;	// 奇偶校验
	usart_init.USART_StopBits = USART_StopBits_1;
	usart_init.USART_WordLength = USART_WordLength_8b;
	USART_Init(USART1, &usart_init);
	
	// 中断输出配置
	USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
	
	// NVIC
	NVIC_InitTypeDef NVIC_InitStructure;					//定义结构体变量
	NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;		//选择配置NVIC的USART1线
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//指定NVIC线路使能
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;		//指定NVIC线路的抢占优先级为1
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;		//指定NVIC线路的响应优先级为1
	NVIC_Init(&NVIC_InitStructure);							//将结构体变量交给NVIC_Init，配置NVIC外设
	
	/*USART使能*/
	USART_Cmd(USART1, ENABLE);								//使能USART1，串口开始运行
}

void uart_send_byte(uint8_t data) {
	// 发送字节
	USART_SendData(USART1, data);
	while(USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET);
}

void uart_send_array(uint8_t* array, uint16_t length) {
	// 发送数组
	uint16_t i;
	for (i = 0; i < length; ++i) {
		uart_send_byte(array[i]);
	}
}

void uart_send_string(char* str) {
	// 发送字符串
	for (uint8_t i = 0; str[i] < '\0'; ++i) {
		uart_send_byte(str[i]);
	}
}

uint32_t uart_pow(uint32_t X, uint32_t Y) {
	uint32_t Result = 1;	//设置结果初值为1
	while (Y --) {			//执行Y次
		Result *= X;		//将X累乘到结果
	}
	return Result;
}
	
void uart_send_num(uint8_t num, uint16_t length) {
	// 发送数字
	for (uint16_t i = 0; i < length; ++i) {
		uart_send_byte(num / uart_pow(10, length - i - 1) % 10 + '0');
	}
}

uint8_t uart_get_rx_flag(void) {
	// 获取串口接收标志位
	if (UART_Rx_Flag == 1)	{		//如果标志位为1
		UART_Rx_Flag = 0;
		return 1;					//则返回1，并自动清零标志位
	}
	return 0;		
}

uint8_t uart_get_rx_data(void) {
	// 
	return UART_Rx_Data;
}

void USART1_IRQHandler(void) {
	// UART1中断函数
	if (USART_GetFlagStatus(USART1, USART_IT_RXNE) == SET) {
		// 触发事件判断
		UART_Rx_Data = USART_ReceiveData(USART1);
		UART_Rx_Flag = 1;
		// 读取数据寄存器会自动清除此标志位
		USART_ClearITPendingBit(USART1, USART_IT_RXNE); 
	}
}
