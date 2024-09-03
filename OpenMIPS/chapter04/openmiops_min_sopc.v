`timescale 1ns / 1ps
`include "defines.v"

/*
    最小SOPC实现,仅包含OpenMIPS 和 指令存储器 ROM
*/

module openmiops_min_sopc(
    input wire clk,
    input wire rst
);

    // 连接指令存储器
    wire [`InstAddrBus] inst_addr;
    wire [`InstBus] inst;
    wire rom_ce;

    // 例化处理器OpenMIPS
    OpenMIPS openmiso_instance(
        .clk(clk),
        .rst(rst),
        .rom_addr_o(inst_addr),
        .rom_data_i(inst),
        .rom_ce_o(rom_ce)
    );

    // 例化指令存储器ROM
    inst_rom inst_rom_instance(
        .ce(rom_ce),
        .addr(inst_addr),
        .inst(inst)
    );

endmodule
