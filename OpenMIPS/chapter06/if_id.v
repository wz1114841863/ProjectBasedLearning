`timescale 1ns / 1ps
`include "defines.v"

/*
    将PC的值,在下一个时钟传递到译码阶段
*/

module if_id(
    input wire clk,
    input wire rst,

    input wire [`InstAddrBus] if_pc,
    input wire [`InstBus] if_inst,

    output reg [`InstAddrBus] id_pc,
    output reg [`InstBus] id_inst
);

    always @(posedge clk) begin
        if (rst == `RstEnable) begin
            id_pc <= `ZeroWord;
            id_inst <= `ZeroWord;
        end else begin  // 单纯的向下传递取值阶段的值
            id_pc <= if_pc;
            id_inst <= if_inst;
        end
    end

endmodule
