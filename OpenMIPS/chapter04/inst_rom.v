`timescale 1ns / 1ps
`include "defines.v"

/*
    指令存储器ROM的实现, 只读
*/

module inst_rom(
    input wire ce,
    input wire [`InstAddrBus] addr,
    output reg [`InstBus] inst
);

    // 定义一个数组, 大小是InstMemNum, 元素宽度是InstBus
    reg [`InstBus] inst_mem[0: `InstMemNum - 1];

    // 使用文件inst_rom.data初始化指令存储器
    initial begin
        $readmemh ("D:/Code/Vivado/OpenMIPS/OpenMIPS.srcs/sources_1/new/inst_rom.data", inst_mem);
    end

    // 当复位信号无效时, 依据输入的地址, 给出指令存储器ROM中对应的元素
    always @(*) begin
        if (ce == `ChipDisable) begin
            inst <= `ZeroWord;
        end else begin
            inst <= inst_mem[addr[`InstMemNumLog2 + 1: 2]];  // 指令地址右移两位
        end
    end

endmodule
