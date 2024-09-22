/****************************************

  Filename            : uart_rx_ctrl.v
  Description         : Send Ctrl Signal By Uart.
  Author              : wz
  Date                : 22-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_rx_ctrl (
    input wire clk,      // 系统时钟信号
    input wire reset_n,  // 系统复位信号
    input wire uart_rx,  // 接收的串口数据

    output reg signal    // 输出信号
);
    parameter baud_set = 3'd4;
    wire [7: 0] data_byte;
    wire [7: 0] ctrl_set;
    wire [31: 0] time_set;
    wire rx_done;

    uart_byte_rx  uart_byte_rx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .baud_set(baud_set),
        .uart_rx(uart_rx),
        .data_byte(data_byte),
        .rx_done(rx_done)
    );



    uart_cmd  uart_cmd_inst (
        .clk(clk),
        .reset_n(reset_n),
        .rx_done(rx_done),
        .rx_data(data_byte),
        .ctrl_set(ctrl_set),
        .time_set(time_set)
    );


    uart_rx_counter  uart_rx_counter_inst (
        .clk(clk),
        .reset_n(reset_n),
        .ctrl_set(ctrl_set),
        .time_set(time_set),
        .signal(signal)
    );

endmodule //uart_rx_ctrl
