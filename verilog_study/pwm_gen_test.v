`timescale 1ns/1ps
`define CLK_PERIOD 20
/****************************************

  Filename            : pwm_gen_test.v
  Description         : Testbench for pwm module.
  Author              : wz
  Date                : 07-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module pwm_gen_tb;
    reg clk;
    reg reset_n;
    reg pwm_gen_en;
    reg [31: 0] counter_arr;
    reg [31: 0] counter_compare;
    wire pwm_out;

    pwm_gen  pwm_gen_inst (
        .clk(clk),
        .reset_n(reset_n),
        .pwm_gen_en(pwm_gen_en),
        .counter_arr(counter_arr),
        .counter_compare(counter_compare),
        .pwm_out(pwm_out)
    );

    initial clk = 1;
    always #(`CLK_PERIOD / 2) clk = ~clk;

    initial begin
		reset_n = 0;
		pwm_gen_en = 0;
		counter_arr = 0;
		counter_compare = 0;

		#(`CLK_PERIOD * 20 + 1);
		reset_n = 1;

		#(`CLK_PERIOD * 10 + 1);
		counter_arr = 1000;
		counter_compare = 400;
		#(`CLK_PERIOD * 10);
		pwm_gen_en = 1;
		#100050;

		counter_compare = 700;
		#100050;
		pwm_gen_en = 0;

		counter_arr = 500;
		counter_compare = 250;
		#(`CLK_PERIOD * 10);
		pwm_gen_en = 1;
		#50050;
		counter_compare = 100;
		#50050;
		pwm_gen_en = 0;

		#200;
		$stop;
	end
endmodule
