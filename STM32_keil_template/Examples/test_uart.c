#include "test_uart.h"

void uart_test() {
	my_uart1_init();
	uint8_t rx_data;
	while(1) {
		if (uart_get_rx_flag() == 1) {			//检查串口接收数据的标志位
			rx_data = uart_get_rx_data();		//获取串口接收的数据
			uart_send_byte(rx_data);			    //串口将收到的数据回传回去，用于测试
		}
	}	
}
