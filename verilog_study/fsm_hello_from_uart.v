/****************************************

  Filename            : fsm_hello_from_uart.v
  Description         : 检测从uart传来的字符串是不是"hello"
  Author              : wz
  Date                : 23-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module fsm_hello_from_uart (
    input wire clk,      // 系统时钟
    input wire reset_n,  // 复位信号
    input wire uart_rx,  // 串口输入信号

    output reg signal    // 输出信号
);
    assign reset = ~reset_n;

    //====================== 串口输入数据 ======================//
    reg [2: 0] baud_set = 3'd0;
    wire [7: 0] data_byte;
    wire rx_done;

    uart_byte_rx  uart_byte_rx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .baud_set(baud_set),
        .uart_rx(uart_rx),
        .data_byte(data_byte),
        .rx_done(rx_done)
    );


    //====================== 字串串检测 ======================//
    wire check_ok;

    fsm_hello  fsm_hello_inst (
        .clk(clk),
        .reset_n(reset_n),
        .data_in(data_in),
        .data_in_valid(data_in_valid),
        .check_ok(check_ok)
    );

    //====================== 每检测到一次, 输出翻转 ======================//
    always @(posedge clk or posedge reset) begin
        if (reset)
            signal <= 1'b0;
        else if (check_ok)
            signal <= ~signal;
        else
            signal <= signal;
    end
endmodule //fsm_hello_from_uart
