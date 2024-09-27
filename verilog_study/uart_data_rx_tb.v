/****************************************

  Filename            : uart_data_rx_tb.v
  Description         : testbench for uart data rx module.
  Author              : wz
  Date                : 26-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_data_rx_tb;
    parameter DATA_WIDTH = 32;
    parameter MSB_FIRST = 0;

    reg  clk;
    reg  reset_n;

    reg [DATA_WIDTH - 1 : 0] data;
    reg send_en;
    reg [2: 0] baud_set;
    wire uart_tx;
    wire tx_done;
    wire uart_state;

    wire  uart_rx;
    wire rx_done;
    wire timeout_flag;
    wire [DATA_WIDTH - 1: 0] rx_data;

    assign uart_rx = uart_tx;

    uart_data_tx #(
        .DATA_WIDTH(DATA_WIDTH),
        .MSB_FIRST(MSB_FIRST)
    ) uart_data_tx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .data(data),
        .send_en(send_en),
        .baud_set(baud_set),
        .uart_tx(uart_tx),
        .tx_done(tx_done),
        .uart_state(uart_state)
    );

    uart_data_rx #(
        .DATA_WIDTH(DATA_WIDTH),
        .MSB_FIRST(MSB_FIRST)
    ) uart_data_rx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .uart_rx(uart_rx),
        .baud_set(baud_set),
        .rx_done(rx_done),
        .timeout_flag(timeout_flag),
        .data(rx_data)
    );

    initial begin
        clk = 1;
        baud_set = 3'd4;
    end
    always #10 clk = ~clk;

    initial begin
        reset_n = 0;
        data = 0;
        send_en = 0;

        #201;
        reset_n = 1;

        #2000;
        data = 32'h12345678;
        send_en = 1;

        #20;
        send_en = 0;

        #20;
        @(posedge tx_done);

        #1;
        #1000000;
        data = 32'h87654321;
        send_en = 1;

        #20;
        send_en = 0;
        #20;

        @(posedge tx_done);
        #1;
        #1000000;
        data = 32'h24680135;
        send_en = 1;

        #20;
        send_en = 0;

        #20;
        @(posedge tx_done);

        #1;
        #400000;
        $stop;
    end
endmodule
