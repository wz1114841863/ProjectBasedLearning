/****************************************

  Filename            : pwm_gen.v
  Description         : PWM 波形产生模块.
  Author              : wz
  Date                : 07-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module pwm_gen (
    input wire clk,                      // 系统时钟
    input wire reset_n,                  // 复位信号
    input wire pwm_gen_en,               // 使能信号
    input wire [31: 0] counter_arr,      // 32bit预重装值, 用于确定频率
    input wire [31: 0] counter_compare,  // 32bit输出比较值, 用于确定占空比
    output reg pwm_out                   // PWM波形输出
);
    wire reset;
    assign reset = ~reset_n;
    //====================== 32bit计数器 ======================//
    reg [31: 0]pwm_gen_cnt;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            pwm_gen_cnt <= 32'd1;
        end else if (pwm_gen_en) begin
            if (pwm_gen_cnt <= 32'd1) begin
                pwm_gen_cnt <= counter_arr;  // 加载预重装寄存器
            end else begin
                pwm_gen_cnt <= pwm_gen_cnt - 1'b1;
            end
        end else begin
            pwm_gen_cnt <= counter_arr;
        end
    end

    //====================== 比较器 ======================//
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            pwm_out <= 1'b0;
        end else if (pwm_gen_cnt <= counter_compare) begin
            pwm_out <= 1'b1;
        end else begin
            pwm_out <= 1'b0;
        end
    end
endmodule //pwm_gen
