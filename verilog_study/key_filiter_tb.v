`timescale 1ns/1ns
`define CLK_PERIOD 20
/****************************************

  Filename            : key_filiter.v
  Description         : 模拟按键抖动进行测试
  Author              : wz
  Date                : 27-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module key_filter_tb;

    reg  clk;
    reg  reset_n;
    reg  key_in;
    wire key_flag;
    wire key_state;

    key_filter  key_filter_inst (
        .clk(clk),
        .reset_n(reset_n),
        .key_in(key_in),
        .key_flag(key_flag),
        .key_state(key_state)
    );

    // reg key_out;
    reg key_press;
	reg [15: 0] myrand;

    initial begin
        clk = 1;
        key_in = 1'b1;
        while (1) begin
            @(posedge key_press);
            key_gen;
        end
    end
    always #10 clk = ~clk;

	task key_gen; begin
        key_in = 1'b1;
        repeat (50) begin
            myrand = {$random} % 65536;  // 0~65535;
            #myrand key_in = ~key_in;
        end
        key_in = 0;
        #25000000;

        repeat (50) begin
            myrand = {$random} % 65536;  // 0~65535;
            #myrand key_in = ~key_in;
        end
        key_in = 1;
        #25000000;
	end
	endtask

    initial begin
        reset_n = 1'b0;
        key_press = 1'b0;
        # (`CLK_PERIOD * 10);
        reset_n = 1'b1;
        # (`CLK_PERIOD * 10 + 1);

        key_press = 1'b1;
        #(`CLK_PERIOD);
		key_press = 1'b0;
		#60000000;

        key_press = 1'b1;
		#(`CLK_PERIOD);
		key_press = 1'b0;
		#60000000;

        $stop;
    end

endmodule
