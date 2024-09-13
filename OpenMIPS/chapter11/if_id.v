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

    input wire [5: 0] stall,

    input wire flush,

    output reg [`InstAddrBus] id_pc,
    output reg [`InstBus] id_inst
);

    always @(posedge clk) begin
        if (rst == `RstEnable) begin
            id_pc <= `ZeroWord;
            id_inst <= `ZeroWord;
        end else if (flush == 1'b1) begin
            // flush == 1表示异常发生, 要清除流水线
            // 所以复位id_pc, id_inst寄存器的值
            id_pc <= `ZeroWord;
            id_inst <= `ZeroWord;
        end else if (stall[1] == `Stop && stall[2] == `NoStop) begin
            // 取值阶段暂停, 而译码阶段继续
            id_pc <= `ZeroWord;
            id_inst <= `ZeroWord;
        end else if (stall[1] == `NoStop) begin
            // 取值阶段继续
            id_pc <= if_pc;
            id_inst <= if_inst;
        end  // 其他情况, 输出保持不变
    end

endmodule
