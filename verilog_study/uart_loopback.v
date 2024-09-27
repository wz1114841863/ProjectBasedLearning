/****************************************

  Filename            : uart_loopback.v
  Description         : 顶层封装:例化多字节串口收发模块
                        将接收模块的输出与发送模块的输入端相连, 实现串口回环
  Author              : wz
  Date                : 26-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_loopback #(
    parameter DATA_WIDTH = 32,
    parameter MSB_FIRST = 0
) (
    input wire clk,
    input wire reset_n,
    input wire uart_rx,

    output [2: 0] flag,
    output uart_tx
);
    // parameter DATA_WIDTH = 32;
    // parameter MSB_FIRST = 0;

    wire [DATA_WIDTH - 1: 0] data;
    wire rx_done;
    wire [7: 0] data_byte;

    uart_data_rx #(
        .DATA_WIDTH(DATA_WIDTH),
        .MSB_FIRST(MSB_FIRST)
    ) uart_data_rx(
        .clk(clk),
        .reset_n(reset_n),
        .uart_rx(uart_rx),
        .data(data),
        .rx_done(rx_done),
        .timeout_flag(flag[0]),
        .baud_set(3'd4)
    );

    uart_data_tx #(
        .DATA_WIDTH(DATA_WIDTH),
        .MSB_FIRST(MSB_FIRST)
    ) uart_data_tx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .data(data),
        .send_en(rx_done),
        .baud_set(baud_set),
        .uart_tx(uart_tx),
        .tx_done(flag[1]),
        .uart_state(flag[2])
    );
endmodule //uart_loopback
