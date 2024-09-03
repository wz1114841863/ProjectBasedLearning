`timescale 1ns / 1ps

/*
    Test bench
*/

module openmips_min_sopc_tb();

    reg CLOCK_50;
    reg rst;

    // 每隔10ns, CLOCK_50信号翻转一次, 周期时20ms
    initial begin
        CLOCK_50 = 1'b0;
        forever #10 CLOCK_50 = ~CLOCK_50;
    end

    // 最初时刻, 复位信号有效, 在195ns时,复位信号无效, 最小SOPC开始运行
    // 运行1000ns后,停止仿真
    initial begin
        rst = 1'b1;
        #195 rst = 1'b0;
        #1000 $stop;
    end

    // 例化最小SOPC
    openmiops_min_sopc openmiops_min_sopc_instance(
        .clk(CLOCK_50),
        .rst(rst)
    );

endmodule
