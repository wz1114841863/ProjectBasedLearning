`timescale 1ns/1ns
`define CLK_PERIOD 20
/****************************************

  Filename            : hex8_tb.v
  Description         : testbench for hex8 module.
  Author              : wz
  Date                : 30-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module hex8_tb;

	reg clk;
	reg reset_n;
	reg disp_en;
	reg [31: 0] data;
	wire [7: 0] sel;
	wire [7: 0] seg;

	hex8 hex8(
		.clk(clk),
		.reset_n(reset_n),
		.disp_en(disp_en),
		.data(data),
		.sel(sel),
		.seg(seg)
	);

	initial clk = 1;
	always#(`CLK_PERIOD/2) clk = ~clk;

	initial begin
		reset_n = 1'b0;
		disp_en = 1;
		data = 32'h12345678;
		#(`CLK_PERIOD*20);

		reset_n = 1'b1;
		#(`CLK_PERIOD*20);

		#20000000;
		data = 32'h87654321;

		#20000000;
		data = 32'h89abcdef;

		#20000000;
		$stop;
	end

endmodule
