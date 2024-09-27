/****************************************

  Filename            : uart_loopback_tb.v
  Description         : Test bench for uart loopback module.
  Author              : wz
  Date                : 26-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_loopback_tb;
    parameter DATA_WIDTH = 256;
    parameter MSB_FIRST = 0;

    reg clk;
    reg reset_n;
    reg [DATA_WIDTH - 1 : 0] data;
    reg send_en;
    reg [2: 0] baud_set;
    wire uart_tx_test;
    wire tx_done_test;
    wire uart_state_test;

    wire  uart_rx;
    wire [2: 0] flag;
    wire uart_tx;

    assign uart_rx = uart_tx_test;

    uart_data_tx #(
        .DATA_WIDTH(DATA_WIDTH),
        .MSB_FIRST(MSB_FIRST)
    ) uart_data_tx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .data(data),
        .send_en(send_en),
        .baud_set(baud_set),
        .uart_tx(uart_tx_test),
        .tx_done(tx_done_test),
        .uart_state(uart_state_test)
    );

    uart_loopback  uart_loopback_inst (
        .clk(clk),
        .reset_n(reset_n),
        .uart_rx(uart_rx),
        .flag(flag),
        .uart_tx(uart_tx)
    );

    initial  begin
        clk = 1;
        baud_set = 3'd4;
    end
    always #10 clk = ~clk;


    initial begin
        reset_n = 0;
        data = 0;
        send_en = 0;

        #201;
        reset_n= 1;

        #2000;
        data = 256'h890abcdef12312345abcdef674567890cba0987654fed365432121fedcba0987;
        send_en = 1;

        #20;
        send_en = 0;

        #20;
        @(posedge tx_done_test);

        #1000000;
        data = 256'hcba0987654fed365432121fedcba0987ba09876fe321dc54b321dc6fe54a0978;
        send_en = 1;

        #20;
        send_en = 0;

        #20;
        @(posedge tx_done_test);

        #1000000;
        data = 256'h890abcdef123123454fed365432121fedcba09875abcdef674567890cba09876;
        send_en = 1;

        #20;
        send_en = 0;

        #20;
        @(posedge tx_done_test);

        #1;
        #4000000;
        $stop;
    end

endmodule
