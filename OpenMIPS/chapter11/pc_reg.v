`timescale 1ns / 1ps
`include "defines.v"
/*
    取值阶段取出指令寄存器的指令, 同时, PC值递增, 准备取下一条指令
*/

module pc_reg(
    input wire clk,
    input wire rst,

    // 来自控制模块的指令
    input wire [5: 0] stall,

    // 来自译码阶段ID模块的信息
    input wire branch_flag_i,
    input wire [`RegBus] branch_target_address_i,

    input wire flush,
    input wire [`RegBus] new_pc,

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
        end else begin
            if (flush == 1'b1) begin
                // 输入信号flush为1表示异常发生
                // 将从CTRL模块给出的异常处理例程入口地址new_pc处取指执行
                pc <= new_pc;
            end else if (stall[0] == `NoStop) begin
                if (branch_flag_i == `Branch) begin
                    pc <= branch_target_address_i;  // 跳转指令, 转移地址
                end else begin
                    pc <= pc + 4'h4;  // 指令存储器使能的时候, PC的值每时钟周期加4
                end
            end // 没有复位信号以及流水线请求暂停时, PC保持不变(寄存器值不变).
        end
    end

endmodule
