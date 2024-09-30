/****************************************

  Filename            : hc595_driver.v
  Description         : hc595芯片驱动信号生成
  Author              : wz
  Date                : 30-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module hc595_driver (
    input wire clk,             // 系统时钟
    input wire reset_n,         // 复位信号
    input wire [15: 0] data,    // 并行数据, 包括位选信号和段选信号
    input wire chip_en,         // 芯片使能引脚

    output reg ds,              // 串行数据输出
    output reg sh_cp,           // 移位寄存器的时钟输出
    output reg st_cp            // 存储寄存器的时钟输出
);
    assign reset = ~reset_n;

    //====================== 12.5MHz-工作时钟 ======================//
    parameter CNT_MAX = 2;  // 四分频
    reg [15: 0] r_data;
    reg [7: 0] divider_cnt;

    always @(posedge clk) begin
        r_data <= data;
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            divider_cnt <= 0;
        end else if (divider_cnt == CNT_MAX - 1'b1) begin
            divider_cnt <= 0;
        end else begin
            divider_cnt <= divider_cnt + 1'b1;
        end
    end

    //====================== 利用查找表实现数据串行输出 ======================//
    wire sck_plus;
    assign sck_plus = (divider_cnt == CNT_MAX - 1'b1);
    reg [5: 0] SHCP_EDGE_CNT;  // 对sck_plus进行计数

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            SHCP_EDGE_CNT <= 0;
        end else if (sck_plus) begin
            if (SHCP_EDGE_CNT == 6'd32)
                SHCP_EDGE_CNT <= 0;
            else
                SHCP_EDGE_CNT <= SHCP_EDGE_CNT + 1'b1;
        end else begin
            SHCP_EDGE_CNT <= SHCP_EDGE_CNT;
        end
    end

    always@(posedge clk or posedge reset) begin
        if(reset)begin
            ds <= 1'b0;
            st_cp <= 1'b0;
            sh_cp <= 1'd0;
        end
	    else begin
            case (SHCP_EDGE_CNT)
                0: begin sh_cp <= 0; st_cp <= 1'd0; ds <= r_data[15]; end
                1: begin sh_cp <= 1; st_cp <= 1'd0; end
                2: begin sh_cp <= 0; ds <= r_data[14]; end
                3: begin sh_cp <= 1; end
                4: begin sh_cp <= 0; ds <= r_data[13]; end
                5: begin sh_cp <= 1; end
                6: begin sh_cp <= 0; ds <= r_data[12]; end
                7: begin sh_cp <= 1; end
                8: begin sh_cp <= 0; ds <= r_data[11]; end
                9: begin sh_cp <= 1; end
                10: begin sh_cp <= 0; ds <= r_data[10]; end
                11: begin sh_cp <= 1; end
                12: begin sh_cp <= 0; ds <= r_data[9]; end
                13: begin sh_cp <= 1; end
                14: begin sh_cp <= 0; ds <= r_data[8]; end
                15: begin sh_cp <= 1; end
                16: begin sh_cp <= 0; ds <= r_data[7]; end
                17: begin sh_cp <= 1; end
                18: begin sh_cp <= 0; ds <= r_data[6]; end
                19: begin sh_cp <= 1; end
                20: begin sh_cp <= 0; ds <= r_data[5]; end
                21: begin sh_cp <= 1; end
                22: begin sh_cp <= 0; ds <= r_data[4]; end
                23: begin sh_cp <= 1; end
                24: begin sh_cp <= 0; ds <= r_data[3]; end
                25: begin sh_cp <= 1; end
                26: begin sh_cp <= 0; ds <= r_data[2]; end
                27: begin sh_cp <= 1; end
                28: begin sh_cp <= 0; ds <= r_data[1]; end
                29: begin sh_cp <= 1; end
                30: begin sh_cp <= 0; ds <= r_data[0]; end
                31: begin sh_cp <= 1; end
                32: st_cp <= 1'd1;
                default:
                    begin
                        st_cp <= 1'b0;
                        ds <= 1'b0;
                        sh_cp <= 1'd0;
                    end
            endcase
	    end
    end
endmodule //hc595_driver
