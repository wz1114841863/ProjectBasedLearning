/****************************************

  Filename            : uart_dpram.v
  Description         : 顶层模块,通过串口将数据发送到FPGA, 在RAM中进行存储. 当需要时, 通过按键将RAM中存储的数据发出.
  Author              : wz
  Date                : 01-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module uart_dpram (
    input wire clk,      // 系统时钟50MHz
    input wire reset_n,  // 系统复位信号, 低电平有效
    input wire uart_rx,  // 串行数据输入
    input wire key_in,   // 按键输入
    output wire uart_tx  // 串行数据输出
);
    wire key_flag;
    wire key_state;

    wire rx_done;
    wire tx_done;
    wire send_en;
    wire wea;
    wire [7: 0] addra;
    wire [7: 0] addrb;

    wire [7: 0] rx_data;
    wire [7: 0] tx_data;

    wire uart_state;

    parameter [2: 0] baud_set = 3'd0;

    key_filter  key_filter_inst (
        .clk(clk),
        .reset_n(reset_n),
        .key_in(key_in),
        .key_flag(key_flag),
        .key_state(key_state)
    );

    uart_ctrl  uart_ctrl_inst (
        .clk(clk),
        .reset_n(reset_n),
        .key_flag(key_flag),
        .key_state(key_state),
        .rx_done(rx_done),
        .tx_done(tx_done),
        .send_en(send_en),
        .wea(wea),
        .addra(addra),
        .addrb(addrb)
    );

    uart_byte_rx  uart_byte_rx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .baud_set(baud_set),
        .uart_rx(uart_rx),
        .data_byte(rx_data),
        .rx_done(rx_done)
    );

    blk_mem_ram_ip blk_mem_ram_ip_inst (
        .clka(clk),    // input wire clka
        .wea(wea),      // input wire [0: 0] wea
        .addra(addra),  // input wire [7: 0] addra
        .dina(rx_data),    // input wire [7: 0] dina

        .clkb(clk),    // input wire clkb
        .addrb(addrb),  // input wire [7: 0] addrb
        .doutb(tx_data)   // output wire [7: 0] doutb
    );

    uart_byte_tx  uart_byte_tx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .data_byte(tx_data),
        .send_en(send_en),
        .baud_set(baud_set),
        .uart_tx(uart_tx),
        .tx_done(tx_done),
        .uart_state(uart_state)
    );

endmodule // uart_dpram
