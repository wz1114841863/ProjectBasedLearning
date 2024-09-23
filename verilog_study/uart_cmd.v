/****************************************

  Filename            : uart_cmd.v
  Description         : Uart Command Parsing
  Author              : wz
  Date                : 22-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_cmd (
    input wire clk,              // 系统时钟
    input wire reset_n,          // 复位信号
    input wire rx_done,          // 接收结束信号
    input wire [7: 0] rx_data,   // 接收到的数据

    output reg [7: 0] ctrl_set,      // 状态控制字
    output reg [31: 0] time_set  // 周期控制字
);
    assign reset = ~reset_n;

    //====================== 移位缓存 ======================//
    reg [7: 0] data_str [7: 0];  // 存储8 * 8 bit 数据
    always @(posedge clk) begin
        if (rx_done) begin
            data_str[7] <= rx_data;
            data_str[6] <= data_str[7];
            data_str[5] <= data_str[6];
            data_str[4] <= data_str[5];
            data_str[3] <= data_str[4];
            data_str[2] <= data_str[3];
            data_str[1] <= data_str[2];
            data_str[0] <= data_str[1];
        end
    end

    //====================== 帧结构判断 ======================//
    reg rx_done_delay;
    always @(posedge clk) begin
        // 这里由于最高位的缓存数据与输入数据间存在一个时钟周期的延迟,因
        // 需要对rx_done信号打一拍后再作为条件去判断缓存中的数据.
        rx_done_delay <= rx_done;
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            ctrl_set <= 8'd0;
            time_set <= 32'd0;
        end else if (rx_done_delay) begin
            if((data_str[0] == 8'h55) && (data_str[1] == 8'hA5) && (data_str[7] == 8'hF0)) begin
                time_set[31:24] <= data_str[2];
                time_set[23:16] <= data_str[3];
                time_set[15:8] <= data_str[4];
                time_set[7:0] <= data_str[5];
                ctrl_set <= data_str[6];
            end
        end
    end
endmodule //uart_cmd
