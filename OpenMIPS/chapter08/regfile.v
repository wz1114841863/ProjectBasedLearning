`timescale 1ns / 1ps
`include "defines.v"

/*
    Regfile 模块实现了32个32位的通用整数寄存器
    可以同时进行两个寄存器的读操作和一个寄存器的写操作
*/

module regfile(
    input wire clk,
    input wire rst,

    // 写端口
    input wire we,
    input wire [`RegAddrBus] waddr,
    input wire [`RegBus] wdata,

    // 读端口1
    input wire re1,
    input wire [`RegAddrBus] raddr1,
    output reg [`RegBus] rdata1,

    // 读端口2
    input wire re2,
    input wire [`RegAddrBus] raddr2,
    output reg [`RegBus] rdata2
);

    // 定义32个32位的寄存器
    reg [`RegBus] regs[0: `RegNum - 1];

    // 写操作
    always @(posedge clk) begin
        if (rst == `RstDisable) begin
            // 寄存器0其数值只能是0
            if ((we == `WriteEnable) && (waddr != `RegNumLog2'h0)) begin
                regs[waddr] <= wdata;
            end
        end
    end

    // 读操作
    always @(*) begin
        if (rst == `RstEnable) begin
            rdata1 <= `ZeroWord;
        end else if (raddr1 == `RegNumLog2'h0) begin
            rdata1 <= `ZeroWord;
        end else if ((raddr1 == waddr) && (we == `WriteEnable) && (re1 == `ReadEnable)) begin
            // 读写位置相同
            rdata1 <= wdata;
        end else if (re1 == `ReadEnable) begin
            rdata1 <= regs[raddr1];
        end else begin
            rdata1 <= `ZeroWord;
        end
    end

    always @(*) begin
        if (rst == `RstEnable) begin
            rdata2 <= `ZeroWord;
        end else if (raddr2 == `RegNumLog2'h0) begin
            rdata2 <= `ZeroWord;
        end else if ((raddr2 == waddr) && (we == `WriteEnable) && (re2 == `ReadEnable)) begin
            // 读写位置相同
            rdata2 <= wdata;
        end else if (re2 == `ReadEnable) begin
            rdata2 <= regs[raddr2];
        end else begin
            rdata2 <= `ZeroWord;
        end
    end

endmodule
