`timescale 1ns / 1ps
`include "defines.v"

/*
    对暂停请求信号进行判断,然后输出流水线暂停信号stall
*/

module ctrl(
    input wire rst,
    input wire stallreq_from_id,
    input wire stallreq_from_ex,

    // 来自MEM模块
    input wire [31: 0] excepttype_i,
    input wire [`RegBus] cp0_epc_i,

    output reg [`RegBus] new_pc,
    output reg flush,
    output reg[5: 0] stall
);
    always @(*) begin
        if (rst == `RstEnable) begin
            stall <= 6'b000000;
			flush <= 1'b0;
			new_pc <= `ZeroWord;
        end else if(excepttype_i != `ZeroWord) begin
            flush <= 1'b1;
            stall <= 6'b000000;
            case (excepttype_i)
                32'h00000001: begin  // interrupt
                    new_pc <= 32'h00000020;
                end
                32'h00000008: begin  // syscall
                    new_pc <= 32'h00000040;
                end
                32'h0000000a: begin  // inst_invalid
                    new_pc <= 32'h00000040;
                end
                32'h0000000d: begin  // trap
                    new_pc <= 32'h00000040;
                end
                32'h0000000c: begin  // ov
                    new_pc <= 32'h00000040;
                end
                32'h0000000e: begin  // eret
                    new_pc <= cp0_epc_i;
                end
                default	: begin
                end
            endcase
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
