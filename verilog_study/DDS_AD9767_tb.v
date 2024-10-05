`timescale 1ns / 1ps
/****************************************

  Filename            : DDS_AD9767_tb.v
  Description         : 顶层模块的测试文件
  Author              : wz
  Date                : 05-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module DDS_AD9767_tb;
    reg clk;
    reg reset_n;
    reg [1: 0] model_selA;
    reg [1: 0] model_selB;
    reg [3: 0] key;
    wire [13: 0] dataA;
    wire clkA;
    wire WRTA;
    wire [13: 0] dataB;
    wire clkB;
    wire WRTB;

    DDS_AD9767  DDS_AD9767_inst (
        .clk(clk),
        .reset_n(reset_n),
        .model_selA(model_selA),
        .model_selB(model_selB),
        .key(key),
        .dataA(dataA),
        .clkA(clkA),
        .WRTA(WRTA),
        .dataB(dataB),
        .clkB(clkB),
        .WRTB(WRTB)
    );

    initial clk = 1;
    always #10 clk = ~clk;

    initial begin
        reset_n = 0;

        #201;
        key = 4'b1111;
        model_selA = 2'b00;
        model_selB = 2'b01;
		reset_n = 1;

        #1000000;
        $stop;
    end
endmodule
