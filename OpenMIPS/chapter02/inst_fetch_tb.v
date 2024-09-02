`timescale 1ns / 1ps

/*
    Test Bench
        1. 只有模块名,没有端口列表, 激励信号必须为reg类型, 以保持信号值, 待观测信号必须为wire类型
        2. Test Bench中要调用被测试模块, 也就是元件例化
        3. Test Bench一般会使用initial, always过程块来定义和描述激励信号
*/

module inst_fetch_tb;

    reg CLOCK;          // 激励信号clock, 时钟信号
    reg rst;            // 激励信号rst, 复位信号
    wire [31: 0] inst;  // 显示信号inst, 取出的指令

    initial begin
        CLOCK = 1'b0;
        forever #10 CLOCK = ~CLOCK;
    end

    initial begin
        rst = 1'b1;  // 初始值为1, 表示复位信号有效
        #195 rst = 1'b0;  // 195ns后,复位信号无效
        #1000 $stop;  // 停止仿真
    end

    inst_fetch  inst_fetch_instance(
        .clk(CLOCK),
        .rst(rst),
        .inst_o(inst)
    );

endmodule
