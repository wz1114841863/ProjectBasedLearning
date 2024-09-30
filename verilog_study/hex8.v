/****************************************

  Filename            : hex8.v
  Description         : 8个8位数码管动态扫描实现
  Author              : wz
  Date                : 30-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module hex8 (
    input wire clk,             // 系统时钟
    input wire reset_n,         // 复位信号
    input wire [31: 0] data,    // 数据
    input wire disp_en,         // 数码管使能信号

    output wire [7: 0] sel,     // 数码管位选, 选择当前要显示的数码管
    output reg [7: 0] seg       // 数码管段选, 当前要显示的内容
);
    assign reset = ~reset_n;

    //====================== 1KHz-分频时钟 ======================//
    reg [14: 0] divider_cnt;  // 25000 - 1, 计数值
	reg clk_1KHz;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            divider_cnt <= 15'd0;
        end else if (!disp_en) begin
            divider_cnt <= 15'd0;
        end else if (divider_cnt == 24999) begin
            divider_cnt <= 15'd0;
        end else begin
            divider_cnt <= divider_cnt + 1'b1;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            clk_1KHz <= 1'b0;
        end else if (divider_cnt == 24999) begin
            clk_1KHz <= ~clk_1KHz;
        end else begin
            clk_1KHz <= clk_1KHz;
        end
    end

    //====================== 移位寄存器, 用于实现数码管位选 ======================//
    reg [7: 0] sel_r;  // 位选信号
    reg [3: 0] data_tmp;  // 对应的段选信号

    always @(posedge clk_1KHz or posedge reset) begin
        if (reset) begin
            sel_r <= 8'b0000_0001;
        end else if (sel_r == 8'b1000_0000) begin
            sel_r <= 8'b0000_0001;
        end else begin
            sel_r <= sel_r << 1;
        end
    end

    always @(*) begin
        case (sel_r)
            8'b0000_0001: data_tmp = data[3: 0];
            8'b0000_0010: data_tmp = data[7: 4];
            8'b0000_0100: data_tmp = data[11: 8];
            8'b0000_1000: data_tmp = data[15: 12];
            8'b0001_0000: data_tmp = data[19: 16];
            8'b0010_0000: data_tmp = data[23: 20];
            8'b0100_0000: data_tmp = data[27: 24];
            8'b1000_0000: data_tmp = data[31: 28];
            default: data_tmp = 4'b000;
        endcase
    end

    //====================== 译码器 ======================//
    always @(*) begin
        case(data_tmp)
            4'h0: seg = 7'b1000000;
            4'h1: seg = 7'b1111001;
            4'h2: seg = 7'b0100100;
            4'h3: seg = 7'b0110000;
            4'h4: seg = 7'b0011001;
            4'h5: seg = 7'b0010010;
            4'h6: seg = 7'b0000010;
            4'h7: seg = 7'b1111000;
            4'h8: seg = 7'b0000000;
            4'h9: seg = 7'b0010000;
            4'ha: seg = 7'b0001000;
            4'hb: seg = 7'b0000011;
            4'hc: seg = 7'b1000110;
            4'hd: seg = 7'b0100001;
            4'he: seg = 7'b0000110;
            4'hf: seg = 7'b0001110;
            default: seg = 7'b1000000;
        endcase
    end

    assign sel = (disp_en) ? sel_r : 8'b0000_0000;

endmodule //hex8
