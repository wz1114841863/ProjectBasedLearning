`timescale 1ns / 1ps
`include "defines.v"
/*
    取值阶段取出指令寄存器的指令, 同时, PC值递增, 准备取下一条指令
*/

module pc_reg(
    input wire clk,
    input wire rst,
    input wire [5: 0] stall,
    output reg [`InstAddrBus] pc,
    output reg                ce
);

    always @(posedge clk) begin
        if (rst == `RstEnable) begin
            ce <= `ChipDisable;
        end else begin
            ce <= `ChipEnable;
        end
    end

    always @(posedge clk) begin
        if (ce == `ChipDisable) begin
            pc <= 32'h00000000;
        end else if (stall[0] == `NoStop) begin
            pc <= pc + 4'h4;  // 指令存储器使能的时候, PC的值每时钟周期加4
        end // 没有复位信号以及流水线请求暂停时, PC保持不变(寄存器值不变).
    end

endmodule
