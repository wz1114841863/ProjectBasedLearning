`timescale 1ns / 1ps
`define CLOCK_PERIOD 20
/****************************************

  Filename            : uart_data_tx_tb.v
  Description         : testbench for uart_data_tx module
  Author              : wz
  Date                : 24-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_data_tx_tb;

    parameter DATA_WIDTH = 32;
    parameter MSB_FIRST = 0;

    reg clk;
    reg reset_n;
    reg [DATA_WIDTH - 1 : 0] data;
    reg send_en;
    reg [2: 0] baud_set;
    wire uart_tx;
    wire tx_done;
    wire uart_state;

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

    initial clk = 1;
    always #(`CLOCK_PERIOD / 2) clk = ~clk;

    initial begin
        reset_n = 0;
        data = 0;
        send_en = 0;
        baud_set = 0;

        #(`CLOCK_PERIOD * 10 + 1)
        reset_n = 1;
        baud_set = 3'd4;

        #(`CLOCK_PERIOD * 100)
        data = 32'h01234567;
        send_en = 1;

        #(`CLOCK_PERIOD)
        send_en = 0;

        #(`CLOCK_PERIOD)
        @(posedge tx_done);

        #1;
        data = 32'h12345678;
        send_en = 1;

        #(`CLOCK_PERIOD);
        send_en = 0;

        #(`CLOCK_PERIOD);
        @(posedge tx_done);
        #1;
        data = 32'h23456789;
        send_en = 1;

        #(`CLOCK_PERIOD);
        send_en = 0;

        #(`CLOCK_PERIOD);
        @(posedge tx_done);

        #1;
        #(`CLOCK_PERIOD * 5000);
         $stop;
    end
endmodule
