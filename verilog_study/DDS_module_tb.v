`timescale 1ns / 1ps
/****************************************

  Filename            : DDS_module_tb.v
  Description         : testbench for DDS_MODULE module.
  Author              : wz
  Date                : 05-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module DDS_Module_tb;

    reg clk;
    reg reset_n;
    reg [1: 0] model_selA, model_selB;
    reg [31: 0] f_wordA, f_wordB;
    reg [11: 0] p_wordA, p_wordB;
    wire [13: 0] dataA, dataB;

    DDS_Module  DDS_Module_instA (
        .clk(clk),
        .reset_n(reset_n),
        .model_sel(model_selA),
        .f_word(f_wordA),
        .p_word(p_wordA),
        .data(dataA)
    );

    DDS_Module  DDS_Module_instB (
        .clk(clk),
        .reset_n(reset_n),
        .model_sel(model_selB),
        .f_word(f_wordB),
        .p_word(p_wordB),
        .data(dataB)
    );

    initial clk = 1;
    always #10 clk = ~clk;

    initial begin
        reset_n = 0;
        f_wordA = 65536;
        p_wordA = 0;
        f_wordB = 65536;
        p_wordB = 1024;
        model_selA = 2'b00;
        model_selB = 2'b00;

        #201;
        reset_n = 1;

        #5000000;
        f_wordA = 65536 * 1024;
        f_wordB = 65536 * 1024;
        p_wordA = 0;
        p_wordB = 2048;

        #1000000;
        $stop;
    end
endmodule
