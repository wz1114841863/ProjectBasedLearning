`timescale 1ns/1ns
`define CLK_PERIOD 20
/****************************************

  Filename            : fsm_hello_tb.v
  Description         : TestBench for fsm hello.
  Author              : wz
  Date                : 23-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module fsm_hello_tb;
    reg  clk;
    reg  reset_n;
    reg [7: 0] data_in;
    reg  data_in_valid;
    wire check_ok;

    fsm_hello  fsm_hello_inst (
        .clk(clk),
        .reset_n(reset_n),
        .data_in(data_in),
        .data_in_valid(data_in_valid),
        .check_ok(check_ok)
    );

    initial clk = 1;
    always #(`CLK_PERIOD / 2) clk = ~clk;

    initial begin
        reset_n = 0;
        data_in_valid = 0;
        data_in = 0;
        #(`CLK_PERIOD * 20);
        reset_n = 1;
        #(`CLK_PERIOD * 20 + 1);

        repeat(2) begin
			gen_char("I");
			#(`CLK_PERIOD);
			gen_char("A");
			#(`CLK_PERIOD);
			gen_char("h");
			#(`CLK_PERIOD);
			gen_char("e");
			#(`CLK_PERIOD);
		    gen_char("l");
			#(`CLK_PERIOD);
			gen_char("l");
			#(`CLK_PERIOD);
			gen_char("l");
			#(`CLK_PERIOD);
			gen_char("h");
			#(`CLK_PERIOD);
			gen_char("e");
			#(`CLK_PERIOD);
			gen_char("l");
			#(`CLK_PERIOD);
		    gen_char("l");
			#(`CLK_PERIOD);
			gen_char("o");
			#(`CLK_PERIOD);
			gen_char("e");
			#(`CLK_PERIOD);
			gen_char("h");
			#(`CLK_PERIOD);
			gen_char("h");
			#(`CLK_PERIOD);
			gen_char("o");
			#(`CLK_PERIOD);
		end

        #200;
        $stop;
    end

    task gen_char;
        input [7: 0] char;
        begin
            data_in = char;
            data_in_valid = 1'b1;
            #(`CLK_PERIOD);
            data_in_valid = 1'b0;
        end
    endtask
endmodule
