/****************************************

  Filename            : DDS_Module.v
  Description         : DDS ADM9767 驱动模块
  Author              : wz
  Date                : 03-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module DDS_Module(
    input wire clk,               // 系统时钟
    input wire reset_n,           // 复位信号
    input wire [1: 0] model_sel,  // 模式选择
    input wire [31: 0] f_word,    // 频率控制字
    input wire [11: 0] p_word,    // 相位控制字

    output reg [13: 0] data       // 输出信号
);
    assign reset = ~reset_n;
    //====================== 寄存器同步 ======================//
    reg [31: 0] reg_f_word;
    reg [11: 0] reg_p_word;
    always @(posedge clk) begin
        reg_f_word <= f_word;
        reg_p_word <= p_word;
    end

    //====================== 根据控制字查表 ======================//
    reg [31: 0] freq_acc;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            freq_acc <= 0;
        end else begin
            freq_acc <= reg_f_word + freq_acc;
        end
    end

    wire [11: 0] rom_addr;
    assign rom_addr = freq_acc[31: 20] + reg_p_word;

    //====================== 输出选择 ======================//
    wire [13:0] data_sine, data_square, data_triangular;
    rom_sine rom_sine(
      .clka(clk),
      .addra(rom_addr),
      .douta(data_sine)
    );

    rom_square rom_square(
      .clka(clk),
      .addra(rom_addr),
      .douta(data_square)
    );

    rom_triangular rom_triangular(
      .clka(clk),
      .addra(rom_addr),
      .douta(data_triangular)
    );

    always @(*) begin
        case (model_sel)
            0: data = data_sine;
            1: data = data_square;
            2: data = data_triangular;
            3: data = 8192;
            default: data = 8192;
        endcase
    end
endmodule
