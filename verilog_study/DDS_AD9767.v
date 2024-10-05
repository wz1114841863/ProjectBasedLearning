/****************************************

  Filename            : DDS_AD9767.v
  Description         : 顶层模块, 实例化按键, ROM和驱动模块
  Author              : wz
  Date                : 03-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module DDS_AD9767 (
    input wire clk,                // 系统时钟
    input wire reset_n,            // 复位信号
    input wire [1: 0] model_selA,  // 通道A模式选择
    input wire [1: 0] model_selB,  // 通道B模式选择
    input wire [3: 0] key,         // 按键输入

    output wire [13: 0] dataA,     // 通道A数据输出
    output wire clkA,              // 通道A时钟信号
    output wire WRTA,              //

    output wire [13: 0] dataB,     // 通道B数据输出
    output wire clkB,              // 通道B时钟信号
    output wire WRTB
);
    assign reset = ~reset_n;

    wire CLK_125M;
    assign clkA = CLK_125M;
    assign clkB = CLK_125M;
    assign WRTA = clkA;
    assign WRTB = clkB;

    reg [31: 0] f_wordA, f_wordB;
    reg [11: 0] p_wordA, p_wordB;

    //====================== 模块例化 ======================//

    clk_wiz_0 clk_wiz_inst (
        .clk_in1(clk),          // input clk_in1
        .clk_out1(CLK_125M),    // output clk_out1
        .resetn(reset_n),       // input resetn
        .locked()               // output locked
    );

    DDS_Module DDS_ModuleA(
        .clk(CLK_125M),
        .reset_n(reset_n),
        .model_sel(model_selA),
        .f_word(f_wordA),
        .p_word(p_wordA),
        .data(dataA)
    );

    DDS_Module DDS_ModuleB(
        .clk(CLK_125M),
        .reset_n(reset_n),
        .model_sel(model_selB),
        .f_word(f_wordB),
        .p_word(p_wordB),
        .data(dataB)
    );

    wire [3: 0] key_flag;
    wire [3: 0] key_state;

    key_filter key_filter0(
        .clk(CLK_125M),
        .reset_n(reset_n),
        .key_in(key[0]),
        .key_flag(key_flag[0]),
        .key_state(key_state[0])
   );

    key_filter key_filter1(
        .clk(CLK_125M),
        .reset_n(reset_n),
        .key_in(key[1]),
        .key_flag(key_flag[1]),
        .key_state(key_state[1])
    );

    key_filter key_filter2(
        .clk(CLK_125M),
        .reset_n(reset_n),
        .key_in(key[2]),
        .key_flag(key_flag[2]),
        .key_state(key_state[2])
    );

    key_filter key_filter3(
        .clk(CLK_125M),
        .reset_n(reset_n),
        .key_in(key[3]),
        .key_flag(key_flag[3]),
        .key_state(key_state[3])
    );

    //====================== 频率和相位控制 ======================//
    reg [2: 0] CHA_fword_sel;
    reg [2: 0] CHB_fword_sel;

    reg [2: 0] CHA_pword_sel;
    reg [2: 0] CHB_pword_sel;
    always @(posedge CLK_125M or posedge reset) begin
        if (reset) begin
            CHA_fword_sel <= 4;
        end else if (key_flag[0] && key_state[0] == 0) begin
            CHA_fword_sel <= CHA_fword_sel + 1'd1;
        end
    end

    always @(posedge CLK_125M or posedge reset) begin
        if (reset) begin
            CHB_fword_sel <= 4;
        end else if (key_flag[1] && key_state[1] == 0) begin
            CHB_fword_sel <= CHB_fword_sel + 1'd1;
        end
    end

    always @(posedge CLK_125M or posedge reset) begin
        if (reset) begin
            CHA_pword_sel <= 3;
        end else if (key_flag[2] && key_state[2] == 0) begin
            CHA_pword_sel <= CHA_pword_sel + 1'd1;
        end
    end

    always @(posedge CLK_125M or posedge reset) begin
        if (reset) begin
            CHB_pword_sel <= 3;
        end else if (key_flag[3] && key_state[3] == 0) begin
            CHB_pword_sel <= CHB_pword_sel + 1'd1;
        end
    end

    //频率控制字
    //如果把周期完整的一个波形等分成2的32次方份,在时钟频率为125M次/秒的条件下,
    //如果希望1秒钟输出一个完成的周期,那么每一拍递进多少份?
    always @(*) begin
        case(CHA_fword_sel)
            0: f_wordA = 34;//2**32 / 1250000000;     34.35
            1: f_wordA = 344;//2**32 / 125000000;
            2: f_wordA = 3436;//2**32 / 12500000;
            3: f_wordA = 34360;//2**32 / 1250000;
            4: f_wordA = 343597;//2**32 / 125000;
            5: f_wordA = 3435974;//2**32 / 12500;
            6: f_wordA = 34359738;//2**32 / 1250;
            7: f_wordA = 343597384;//2**32 / 125;
        endcase
    end
    always @(*) begin
        case(CHB_fword_sel)
            0: f_wordB = 34;//2**32 / 1250000000;     34.35
            1: f_wordB = 344;//2**32 / 125000000;
            2: f_wordB = 3436;//2**32 / 12500000;
            3: f_wordB = 34360;//2**32 / 1250000;
            4: f_wordB = 343597;//2**32 / 125000;
            5: f_wordB = 3435974;//2**32 / 12500;
            6: f_wordB = 34359738;//2**32 / 1250;
            7: f_wordB = 343597384;//2**32 / 125;
        endcase
    end
    always @(*) begin
        case(CHA_pword_sel)
            0: p_wordA = 0;       // 0
            1: p_wordA = 341;     // 30
            2: p_wordA = 683;     // 60
            3: p_wordA = 1024;    // 90
            4: p_wordA = 1707;    // 150
            5: p_wordA = 2048;    // 180
            6: p_wordA = 3072;    // 270
            7: p_wordA = 3641;    // 320
        endcase
    end
    always @(*) begin
        case(CHB_pword_sel)
            0: p_wordB = 0;     // 0
            1: p_wordB = 341;   // 30
            2: p_wordB = 683;   // 60
            3: p_wordB = 1024;  // 90
            4: p_wordB = 1707;  // 150
            5: p_wordB = 2048;  // 180
            6: p_wordB = 3072;  // 270
            7: p_wordB = 3641;  // 320
        endcase
    end
endmodule //DDS_AD9767
