`timescale 1ns / 1ps
`define CLOCK_PERIOD 20
/****************************************

  Filename            : uart_byte_tx_tb.v
  Description         : TestBench for uart_buye_tx
  Author              : wz
  Date                : 21-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module uart_byte_tx_tb;

    // Ports
    reg  clk;
    reg  reset_n;
    reg [7: 0] data_byte;
    reg  send_en;
    reg [2: 0] baud_set;
    wire  uart_tx;
    wire  tx_done;
    wire  uart_state;

    uart_byte_tx  uart_byte_tx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .data_byte(data_byte),
        .send_en(send_en),
        .baud_set(baud_set),
        .uart_tx(uart_tx),
        .tx_done(tx_done),
        .uart_state(uart_state)
    );

    initial clk = 1;
    always #(`CLOCK_PERIOD / 2) clk = ~clk;

    initial begin
        reset_n = 1'b0;
        data_byte = 8'd0;
        send_en = 1'b0;
        baud_set = 3'd4;
        # (`CLOCK_PERIOD * 500 + 1);
        reset_n = 1'b1;
        # (`CLOCK_PERIOD * 50)

        // send first byte
        data_byte = 8'haa;
        send_en = 1'd1;
        # `CLOCK_PERIOD;
        send_en = 1'd0;

        @(posedge tx_done)
        #(`CLOCK_PERIOD * 5000)

        // send second byte
        data_byte = 8'h55;
        send_en = 1'd1;
        # `CLOCK_PERIOD;
        send_en = 1'd0;

        @(posedge tx_done)
        #(`CLOCK_PERIOD * 5000)
        $stop;
    end

endmodule
