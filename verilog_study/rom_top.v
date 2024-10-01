`define DIST_MEM
/****************************************

  Filename            : rom_top.v
  Description         : 顶层模块, 用于查看ROM的具体资源使用情况
  Author              : wz
  Date                : 01-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module rom_top(
    input wire clk,
    input [7: 0] addr,
    output [7: 0] dout
);
`ifdef DIST_MEM
    dist_mem_rom_ip rom (
        .a(addr),       // input wire [7: 0] a
        .clk(clk),      // input wire clk
        .spo(dout),     // output wire [7: 0] spo
        .qspo()         // output wire [7: 0] qspo
    );
`elsif BLK_MEM
    blk_mem_rom_ip rom (
        .clka(clk),      // input wire clka
        .addra(addr),    // input wire [7: 0] addra
        .douta(dout)     // output wire [7: 0] douta
    );
`endif
endmodule
