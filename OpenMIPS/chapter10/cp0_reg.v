`timescale 1ns / 1ps
`include "defines.v"

/*
    CP0寄存器实现
*/

module cp0_reg(
    input wire clk,
    input wire rst,

    input wire we_i,
    input wire [4: 0] waddr_i,
    input wire [4: 0] raddr_i,
    input wire [`RegBus] data_i,

    input wire [5: 0] int_i,

    output reg [`RegBus] data_o,
    output reg [`RegBus] count_o,
    output reg [`RegBus] compare_o,
    output reg [`RegBus] status_o,
    output reg [`RegBus] cause_o,
    output reg [`RegBus] epc_o,
    output reg [`RegBus] config_o,
    output reg [`RegBus] prid_o,

    output reg timer_int_o
);

    // 写操作
    always @(posedge clk) begin
        if (rst == `RstEnable) begin
            // Counter 寄存器的初始值, 为0
            count_o <= `ZeroWord;
            // Compare寄存器的初始值, 为0
            compare_o <= `ZeroWord;
            // Status寄存器的初始值,  其中CU字段为4'b0001, 表示协处理器CP0存在
            status_o <= 32'b0001_0000_0000_0000_0000_0000_0000_0000;
            // Cause寄存器的初始值
            cause_o <= `ZeroWord;
            // EPC寄存器的初始值
            epc_o <= `ZeroWord;
            // Config寄存器的初始值, 其中BE字段为1, 表示工作再大端模式
            config_o <= 32'b0000_0000_0000_0000_1000_0000_0000_0000;
            // PRId寄存器的初始值,
            prid_o <= 32'b0000_0000_0100_1100_0000_0001_0000_0010;

            timer_int_o <= `InterruptNotAssert;
        end else begin
            count_o <= count_o + 1;  // Count寄存器的值在每个时钟周期加1
            cause_o[15: 10] <= int_i;  // Cause的第10 ~ 15bit保存外部中断声明

            // 当Compare寄存器不为0, 且Conut寄存器的值等同于Compare寄存器的值时
            // 将输出信号timer_int_o的置为1, 表示时钟中断发生
            if (compare_o != `ZeroWord && count_o == compare_o) begin
                timer_int_o <= `InterruptAssert;
            end
            if (we_i == `WriteEnable) begin
                case (waddr_i)
                    `CP0_REG_COUNT: begin
                        count_o <= data_i;
                    end
                    `CP0_REG_COMPARE: begin
                        compare_o <= data_i;
                        timer_int_o <= `InterruptNotAssert;
                    end
                    `CP0_REG_STATUS: begin
                        status_o <= data_i;
                    end
                    `CP0_REG_EPC: begin
                        epc_o <= data_i;
                    end
                    `CP0_REG_CAUSE: begin
                        cause_o[9: 8] <= data_i[9: 8];
                        cause_o[23] <= data_i[23];
                        cause_o[22] <= data_i[22];
                    end
                    default: begin
                    end
                endcase
            end
        end
    end

    // 读操作
    always @ (*) begin
		if(rst == `RstEnable) begin
			data_o <= `ZeroWord;
		end else begin
            case (raddr_i)
                `CP0_REG_COUNT: begin
                    data_o <= count_o ;
                end
                `CP0_REG_COMPARE: begin
                    data_o <= compare_o ;
                end
                `CP0_REG_STATUS: begin
                    data_o <= status_o ;
                end
                `CP0_REG_CAUSE: begin
                    data_o <= cause_o ;
                end
                `CP0_REG_EPC: begin
                    data_o <= epc_o ;
                end
                `CP0_REG_PrId: begin
                    data_o <= prid_o ;
                end
                `CP0_REG_CONFIG: begin
                    data_o <= config_o ;
                end
                default: begin
                end
            endcase
		end
	end

endmodule
