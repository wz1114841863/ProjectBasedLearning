`timescale 1ns / 1ps
`define CLOCK_PERIOD 20
/****************************************

  Filename            : bin_counter_tb.v
  Description         : This is testbench for bin_counter.
  Author              : wz
  Date                : 20-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/


module bin_counter_tb;
    //Ports
    reg   clk;
    reg   reset_n;
    wire  flag;

    bin_counter  bin_counter_inst (
        .clk(clk),
        .reset_n(reset_n),
        .flag(flag)
    );

    initial clk = 1;
    always #(`CLOCK_PERIOD / 2) clk = ~clk;

    initial begin
        reset_n = 1'b0;
        #(`CLOCK_PERIOD * 200 + 1);
        reset_n = 1'b1;
        #100000;
        $stop;
    end

endmodule
