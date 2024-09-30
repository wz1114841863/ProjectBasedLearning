/****************************************

  Filename            : key_led.v
  Description         : 顶层模块, 根据按键信号控制led
  Author              : wz
  Date                : 28-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module key_led (
    input wire clk,                 // 系统时钟
    input wire reset_n,             // 复位信号
    input wire key_in0,             // 自加按键
    input wire key_in1,             // 自减按键
    input wire key_in2,             // 左移按键
    input wire key_in3,             // 右移按键

    output wire [7: 0] led          // 显示信号
);
    wire key_flag0, key_state0;
    wire key_flag1, key_state1;
    wire key_flag2, key_state2;
    wire key_flag3, key_state3;
    wire key_add;
    wire key_sub;
    wire key_shift_left;
    wire key_shift_right;

    assign key_add = key_flag0 && (~key_state0);
    assign key_sub = key_flag1 && (~key_state1);
    assign key_shift_left = key_flag2 && (~key_state2);
    assign key_shift_right = key_flag3 && (~key_state3);

    key_filter key_filter0(
		.clk(clk),
		.reset_n(reset_n),
		.key_in(key_in0),
		.key_flag(key_flag0),
		.key_state (key_state0)
	);

    key_filter key_filter1(
		.clk(clk),
		.reset_n(reset_n),
		.key_in(key_in1),
		.key_flag(key_flag1),
		.key_state (key_state1)
	);

    key_filter key_filter2(
		.clk(clk),
		.reset_n(reset_n),
		.key_in(key_in2),
		.key_flag(key_flag2),
		.key_state(key_state2)
	);

    key_filter key_filter3(
		.clk(clk),
		.reset_n(reset_n),
		.key_in(key_in3),
		.key_flag(key_flag3),
		.key_state(key_state3)
	);

	led_ctrl led_ctrl0(
		.clk(clk),
		.reset_n(reset_n),
		.key_add(key_add),
		.key_sub(key_sub),
		.key_shift_left(key_shift_left),
		.key_shift_right(key_shift_right),
		.led(led)
	);
endmodule //key_led
