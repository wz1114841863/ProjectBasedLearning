`timescale 1ns / 1ps
`include "defines.v"

/*
    DIV模块: 使用试商法, 需要32个时钟周期
    采用状态机实现, 总共有四个状态:
        DivFree: 除法模块空闲
        DivByZero: 除数是0
        DivOn: 除法运算进行中
        DivEnd: 除法运算结束
*/

module div(
    input wire clk,
    input wire rst,

    input wire signed_div_i,
    input wire [31: 0] opdata1_i,
    input wire [31: 0] opdata2_i,
    input wire start_i,  // 是否开始除法运算
    input wire annul_i,  // 是否取消除法运算, 为1表示取消

    output reg [63: 0] result_o,  // 除法运算结果
    output reg ready_o  // 除法运算是否结束
);
    wire [32: 0] div_temp;
    reg [5: 0] cnt;  // 记录试商法进行了几轮, 当cnt == 32时, 试商法结束
    reg [64: 0] dividend;
    reg [1: 0] state;
    reg [31: 0] divisor;
    reg [31: 0] temp_op1;
    reg [31: 0] temp_op2;

    // dividend的低32位保存的是被除数和中间结果
    // 第k次迭代结束的时候, dividend[k: 0]保存的就是当前得到的中间结果
    // dividend[31: k + 1]保存的就是被除数中还没有参与运算的数据
    // dividend高32位是每次迭代时的被减数
    // divisor是除数
    assign div_temp = {1'b0, dividend[63: 32]} - {1'b0, divisor};  // minuend - n 运算

    always @(posedge clk) begin
        if (rst == `RstEnable) begin
            state <= `DivFree;
            ready_o <= `DivResultNotReady;
            result_o <= {`ZeroWord, `ZeroWord};
        end else begin
            case (state)
                `DivFree: begin
                    // 分三种情况:
                    //    1) 开始除法运算, 但除数为0  -> `DivByZero
                    //    2) 开始除法运算, 且除数不为0, -> `DivOn, 初始化cnt为0
                    //       如果是有符号除法, 且被除数或者除数为负数, 就对被除数或者除数取补码
                    //       除数保存到divisor中, 将被除数的最高位保存到dividen的第32位
                    //    3) 没有开始除法运算, 保持ready_o为`DivResultNotReady, 保持result_o为0
                    if (start_i == `DivStart && annul_i == 1'b0) begin
                        if (opdata2_i == `ZeroWord) begin
                            state <= `DivByZero;
                        end else begin
                            state <= `DivOn;
                            cnt <= 6'b00_0000;
                            if (signed_div_i == 1'b1 && opdata1_i[31] == 1'b1) begin
                                temp_op1 = ~opdata1_i + 1;  // 被除数取补码
                            end else begin
                                temp_op1 = opdata1_i;
                            end
                            if (signed_div_i == 1'b1 && opdata2_i[31] == 1'b1) begin
                                temp_op2 = ~opdata2_i + 1;  // 被除数取补码
                            end else begin
                                temp_op2 = opdata2_i;
                            end
                            dividend <= {`ZeroWord, `ZeroWord};
                            dividend[32: 1] <= temp_op1;
                            divisor <= temp_op2;
                        end
                    end else begin
                        ready_o <= `DivResultNotReady;
                        result_o <= {`ZeroWord, `ZeroWord};
                    end
                end

                `DivByZero: begin
                    // 进入DivByZero状态, 直接进入DivEnd状态, 除法结束, 且结果为0
                    dividend <= {`ZeroWord, `ZeroWord};
                    state <= `DivEnd;
                end

                `DivOn: begin
                    // 分三种情况:
                    // 1) 如果取消除法计算, 直接返回DivFree状态
                    // 2) 如果cnt != 32, 表示试商法还未结束, 此时如果减法结果为负数, 那么此次迭代结果为0,
                    //    如果减法结果为正, 那么此次结果迭代是1, dividend的最低位保存每次的迭代结果, 同时保存DivON状态, cnt + 1
                    // 3) 如果cnt == 32, 表示试商法结束, 根据是否是否为有符号数以及正负性对商和余数都要进行补码操作
                    //    商保存再dividend的低32位, 余数保存在dividend的高32位, 同时进入DivEnd状态
                    if (annul_i == 1'b0) begin
                        if (cnt != 6'b10_0000) begin  // cnt != 32
                            if (div_temp[32] == 1'b1) begin
                                // 如果div_temp[32]为1, 表示minuend - n的结果小于0
                                // 将dividend向左移一位, 这样就将被除数还没有参与运算的
                                // 最高位加入到下一次迭代的被减数中, 同时将0追加到中间结果
                                dividend <= {dividend[63: 0], 1'b0};
                            end else begin
                                // 如果div_temp[32]为0, 表示minuend - n的结果大于等于0
                                // 将dividend向左移一位, 这样就将被除数还没有参与运算的
                                // 最高位加入到下一次迭代的被减数中, 同时将1追加到中间结果
                                dividend <= {dividend[31: 0], dividend[31: 0], 1'b1};
                            end
                            cnt <= cnt + 1;
                        end else begin  // cnt == 32, 试商法结束
                            if ((signed_div_i == 1'b1) && ((opdata1_i[31] ^ opdata2_i[31])) == 1'b1) begin
                                dividend[31: 0] <= (~dividend[31: 0] + 1);  // 对商求补码
                            end
                            if ((signed_div_i == 1'b1) && ((opdata1_i[31] ^ dividend[64])) == 1'b1) begin
                                dividend[64: 33] <= (~dividend[64: 33] + 1);  // 对余数求补码, 余数的符号位应该与被除数一致
                            end
                            state <= `DivEnd;  // -> DivEnd
                            cnt <= 6'b00_0000;  // cnt清零
                        end
                    end else begin
                        state <= `DivFree;  // 取消除法操作, -> DivFree
                    end
                end

                `DivEnd: begin
                    // 除法运算结束
                    // result_o的宽度是64位, 其中高32位存储余数, 低32位存储商
                    // 设置输出信号ready_o为DivResultReady, 表示除法结束, 然后等待EX模块发送DivStop信号
                    // 当EX模块送来DivStop信号时, Div模块回到DivFree状态
                    result_o <= {dividend[64: 33], dividend[31: 0]};
                    ready_o <= `DivResultReady;
                    if (start_i == `DivStop) begin
                        state <= `DivFree;
                        ready_o <= `DivResultNotReady;
                        result_o <= {`ZeroWord, `ZeroWord};
                    end
                end
            endcase
        end
    end

endmodule
