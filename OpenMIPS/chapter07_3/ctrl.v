`timescale 1ns / 1ps
`include "defines.v"

/*
    对暂停请求信号进行判断,然后输出流水线暂停信号stall
*/

module ctrl(
    input wire rst,
    input wire stallreq_from_id,
    input wire stallreq_from_ex,

    output reg[5: 0] stall
);
    always @(*) begin
        if (rst == `RstEnable) begin
            stall <= 6'b00_0000;
        end else if (stallreq_from_ex == `Stop) begin
            // 要求取值, 译码, 执行阶段暂停, 而访存, 回写阶段继续
            stall <= 6'b00_1111;
        end else if (stallreq_from_id == `Stop) begin
            // 要求取值, 译码阶段暂停, 而执行, 访存, 回写阶段继续
            stall <= 6'b00_0111;
        end else begin
            // 不设置暂停流水线
            stall <= 6'b00_0000;
        end
    end

endmodule
