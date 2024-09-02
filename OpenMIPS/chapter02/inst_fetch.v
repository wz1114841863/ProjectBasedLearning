`timescale 1ns / 1ps
/*
    顶层文件, 用于模块例化
    通过调用各个功能单元, 将其按照一定方式连接在一起, 从而实现最终电路
*/
module inst_fetch(
    input wire clk,
    input wire rst,
    output wire [31: 0] inst_o
);
    wire [5: 0] pc;
    wire        rom_ce;

    // PC模块的例化
    pc pc_instance(
        .clk(clk),
        .rst(rst),
        .pc(pc),
        .ce(rom_ce)
    );

    // ROM模块的例化
    rom rom_instance(
        .ce(rom_ce),
        .addr(pc),
        .inst(inst_o)
    );
endmodule
