`timescale 1ns / 1ps
`include "defines.v"

/*
    将访存阶段的运算结果,在下一个时钟传递到回写阶段
*/

module mem_wb(
    input wire clk,
    input wire rst,

    input wire [`RegAddrBus] mem_wd,
    input wire mem_wreg,
    input wire [`RegBus] mem_wdata,

    input wire [`RegBus] mem_hi,
    input wire [`RegBus] mem_lo,
    input wire mem_whilo,

    input wire [5: 0] stall,

    output reg [`RegAddrBus] wb_wd,
    output reg wb_wreg,
    output reg [`RegBus] wb_wdata,

    output reg [`RegBus] wb_hi,
    output reg [`RegBus] wb_lo,
    output reg wb_whilo
);

    always @(posedge clk) begin
        if (rst == `RstEnable) begin
            wb_wd <= `NOPRegAddr;
            wb_wreg <= `WriteDisable;
            wb_wdata <= `ZeroWord;
            wb_hi <= `ZeroWord;
            wb_lo <= `ZeroWord;
            wb_whilo <= `WriteDisable;
        end else if (stall[4] == `Stop && stall[5] == `NoStop) begin
            // 访存阶段暂停, 而回写阶段继续
            wb_wd <= `NOPRegAddr;
            wb_wreg <= `WriteDisable;
            wb_wdata <= `ZeroWord;
            wb_hi <= `ZeroWord;
            wb_lo <= `ZeroWord;
            wb_whilo <= `WriteDisable;
        end else if (stall[4] == `NoStop) begin
            // 访存阶段继续
            wb_wd <= mem_wd;
            wb_wreg <= mem_wreg;
            wb_wdata <= mem_wdata;
            wb_hi <= mem_hi;
            wb_lo <= mem_lo;
            wb_whilo <= mem_whilo;
        end  // 其余情况, 保持输出不变
    end

endmodule
