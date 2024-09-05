`timescale 1ns / 1ps
`include "defines.v"

/*
    访存阶段: 由于ori指令不需要访问数据存储器,所以在访存阶段目前不需要做任何事
*/

module mem(
    input wire rst,

    input wire [`RegAddrBus] wd_i,
    input wire wreg_i,
    input wire [`RegBus] wdata_i,

    output reg [`RegAddrBus] wd_o,
    output reg wreg_o,
    output reg [`RegBus] wdata_o
);

    always @(*) begin
        if (rst == `RstEnable) begin
            wd_o <= `NOPRegAddr;
            wreg_o <= `WriteDisable;
            wdata_o <= `ZeroWord;
        end else begin
            wd_o <= wd_i;
            wreg_o <= wreg_i;
            wdata_o <= wdata_i;
        end
    end

endmodule
