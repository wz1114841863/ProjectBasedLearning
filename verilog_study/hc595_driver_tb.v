`timescale 1ns/1ns
`define CLK_PERIOD 20
/****************************************

  Filename            : hc595_driver_tb.v
  Description         : testbench for hc595_driver module.
  Author              : wz
  Date                : 30-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module hc595_driver_tb;

	reg clk;
	reg reset_n;
	reg [15: 0] data;
	reg chip_en;
    wire ds;	 // shift serial data
	wire sh_cp;	 // shift clock
	wire st_cp;	 // latch data clock

	hc595_driver hc595_driver(
		.clk(clk),
		.reset_n(reset_n),
		.data(data),
		.chip_en(chip_en),
		.sh_cp(sh_cp),
		.st_cp(st_cp),
		.ds(ds)
	);
	initial clk = 1;
	always#(`CLK_PERIOD/2) clk = ~clk;

	initial begin
		reset_n = 1'b0;
		chip_en = 1;
		data = 16'b1010_1111_0110_0101;
		#(`CLK_PERIOD*20);

		reset_n = 1;
		#(`CLK_PERIOD*20);

		#5000;
		data = 16'b0101_0101_1010_0101;

		#5000;
		$stop;
	end
endmodule
