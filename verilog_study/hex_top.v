/****************************************

  Filename            : hex_top.v
  Description         : 顶层模块, 实现数码管显示
  Author              : wz
  Date                : 30-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module hex_top (
    input wire clk,         // 系统时钟
    input wire reset_n,     // 复位信号

    output wire ds,         // 串行数据输出
    output wire sh_cp,      // 移位寄存器的时钟输出
    output wire st_cp       // 存储寄存器的时钟输出
);
    wire [31: 0] disp_data;
    wire [7: 0] sel;
    wire [6: 0] seg;

    hc595_driver hc595_driver(
        .clk(clk),
        .reset_n(reset_n),
        .data({1'd0,seg,sel}),
        .chip_en(1'b1),
        .ds(ds),
        .sh_cp(sh_cp),
        .st_cp(st_cp)
    );

    hex8 hex8(
        .clk(clk),
        .reset_n(reset_n),
        .disp_en(1'b1),
        .data(disp_data),
        .sel(sel),
        .seg(seg)
    );
endmodule //hex_top
