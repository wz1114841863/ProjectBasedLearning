`timescale 1ns / 1ps
/*
    指令存储器ROM, 存储指令, 并依据输入的地址, 给出对应的指令
*/
module rom(
    input wire ce,
    input wire [5: 0] addr,
    output reg[31: 0] inst
);
    reg[31: 0] rom[63: 0];  // 使用二维向量定义存储器

    initial $readmemh ("D:/Code/Vivado/inst_fetch/inst_fetch.srcs/sources_1/new/rom.data", rom);  // 不可综合, 读取数据至ROM

    always @(*) begin
        if (ce == 1'b0) begin
            inst <= 32'h0;
        end else begin
            inst <= rom[addr];
        end
    end

endmodule
