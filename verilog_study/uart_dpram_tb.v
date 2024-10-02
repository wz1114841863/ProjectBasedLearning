`timescale 1ns/1ns
`define CLK_PERIOD 20
/****************************************

  Filename            : uart_dpram_tb.v
  Description         : 顶层测试模块, 需要额外例化key_model和uart_tx模块.
  Author              : wz
  Date                : 01-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module uart_dpram_tb();

	reg clk;
	reg reset_n;
	reg key_press;
	reg [7: 0] tx_data;
	reg send_en;

	wire key_in;
	wire uart_tx;
	wire tx_done;

	initial clk = 1;
	always #(`CLK_PERIOD / 2) clk = ~clk;

	initial begin
		reset_n   = 1'b0;
		key_press = 1'b0;
		tx_data   = 8'h00;
		send_en   = 1'b0;
		#(`CLK_PERIOD*20);
		reset_n = 1;

        // 1st byte
		tx_data = 8'haa;
		send_en = 1'b1;
		#(`CLK_PERIOD);
		send_en = 1'b0;

		@(posedge tx_done);
		#(`CLK_PERIOD*5);

        // 2nd byte
		tx_data = 8'hbb;
		send_en = 1'b1;
		#(`CLK_PERIOD);
		send_en = 1'b0;

		@(posedge tx_done);
		#(`CLK_PERIOD*5);

        // 3rd byte
		tx_data = 8'h55;
		send_en = 1'b1;
		#(`CLK_PERIOD);
		send_en = 1'b0;

		@(posedge tx_done);
		#(`CLK_PERIOD*5);

        // change state
		key_press = 1'b1;
		#(`CLK_PERIOD);
		key_press = 1'b0;

		#35000000;
		$stop;
	end

	key_model key_model_inst(
		.key_press(key_press),
		.key_out(key_in)
	);

	uart_byte_tx uart_byte_tx_inst(
		.clk(clk),
		.reset_n(reset_n),

		.data_byte(tx_data),
		.send_en(send_en),
		.baud_set(3'd0),

		.uart_tx(uart_tx),
		.tx_done(tx_done),
		.uart_state()
	);

	uart_dpram uart_dpram_inst(
		.clk(clk),
		.reset_n(reset_n),

		.key_in(key_in),
		.uart_rx(uart_tx),
		.uart_tx()
	);


endmodule
