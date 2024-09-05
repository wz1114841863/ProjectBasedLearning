`timescale 1ns / 1ps
`include "defines.v"

/*
    执行阶段: 依据译码阶段的结果, 对源操作数1, 源操作树2进行指定的运算, 执行阶段包括EX, EX/MEM两个模块
*/


module ex(
    input wire rst,

    input wire [`AluOpBus] aluop_i,
    input wire [`AluSelBus] alusel_i,
    input wire [`RegBus] reg1_i,
    input wire [`RegBus] reg2_i,
    input wire [`RegAddrBus] wd_i,
    input wire wreg_i,

    output reg [`RegAddrBus] wd_o,
    output reg wreg_o,
    output reg [`RegBus] wdata_o
);

    // 保存逻辑运算的结果
    reg [`RegBus] logicout;

    // 依据aluop_i指示的运算子类型进行运算, 此处只有逻辑 "或" 运算
    always @(*) begin
        if (rst == `RstEnable) begin
            logicout <= `ZeroWord;
        end else begin
            case (aluop_i)
                `EXE_OR_OP: begin
                    logicout <= reg1_i | reg2_i;
                end
                default: begin
                    logicout <= `ZeroWord;
                end
            endcase
        end
    end

    // 依据alusel_i指示的运算类型,选择一个运算结果作为最终结果,此处只有逻辑运算结果
    always @(*) begin
        wd_o <= wd_i;
        wreg_o <= wreg_i;
        case (alusel_i)
            `EXE_RES_LOGIC: begin
                wdata_o  <= logicout;
            end
            default: begin
                wdata_o <= `ZeroWord;
            end
        endcase
    end

endmodule
