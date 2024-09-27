/****************************************

  Filename            : uart_data_rx.v
  Description         : 多字节串口接收模块
  Author              : wz
  Date                : 25-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_data_rx #(
    parameter DATA_WIDTH = 16,  // 数据宽度
    parameter MSB_FIRST = 1
)(
    clk,           // 系统时钟信号
    reset_n,       // 复位信号
    uart_rx,       // 串口接收串行数据
    baud_set,      // 波特率设置

    rx_done,       // 串口数据接收完毕标志位
    timeout_flag,  // 超时标志位
    data           // 串口输出并行数据
);
    input wire clk;
    input wire reset_n;
    input wire uart_rx;
    input wire [2: 0] baud_set;

    output reg rx_done;
    output reg timeout_flag;
    output reg [DATA_WIDTH - 1: 0] data;

    assign reset = ~reset_n;

    localparam S0 = 0;
    localparam S1 = 1;
    localparam S2 = 2;
    reg [1: 0] state;
    reg [DATA_WIDTH - 1 : 0] data_r;
    reg [8: 0] cnt;  // 接收byte数计数器

    //====================== 单字节接收模块例化 ======================//
    wire [7: 0] data_byte;
    wire byte_rx_done;

    uart_byte_rx  uart_byte_rx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .baud_set(baud_set),
        .uart_rx(uart_rx),
        .data_byte(data_byte),
        .rx_done(byte_rx_done)
    );

    //====================== 接收超时模块 ======================//
    // 这里参考modubus, 传输一旦超过3.5字符就视作是新的报文
    reg [31: 0] timeout_cnt;  // 传输超时计数器
    reg to_state;  // 计数器工作信号
    wire [19: 0] TIMEOUT;
    assign TIMEOUT = (baud_set == 3'd0) ? 20'd182291:
                     (baud_set == 3'd1) ? 20'd91145:
                     (baud_set == 3'd2) ? 20'd45572:
                     (baud_set == 3'd3) ? 20'd30381:
                                        20'd15190;

    always @(posedge clk or posedge reset) begin
        if (reset)
            timeout_flag <= 1'b0;
        else if (timeout_cnt >= TIMEOUT)
            timeout_flag <= 1'b1;
        else if (state == S0)
            timeout_flag <= 1'b0;
        else
            timeout_flag <= 1'b0;
    end

    always @(posedge clk or posedge reset) begin
        if (reset)
            to_state <= 0;
        else if (uart_rx == 1'b0)
            // 利用串口传输时的低电平起始信号
            to_state <= 1;
        else if (byte_rx_done == 1'b1)
            to_state <= 0;
        else
            to_state <= to_state;
    end

    always @(posedge clk or posedge reset) begin
        if (reset)
            timeout_cnt <= 32'd0;
        else if (to_state) begin
            if (byte_rx_done)
                timeout_cnt <= 32'd0;
            else if (timeout_cnt >= TIMEOUT)
                if (byte_rx_done)
                    timeout_cnt <= 32'd0;
                else if (timeout_cnt >= TIMEOUT)
                    timeout_cnt <= TIMEOUT;
                else
                    timeout_cnt <= timeout_cnt + 1'd1;
        end
    end

    //====================== 接收控制状态机 ======================//
    always @(posedge clk or posedge reset) begin
        if (reset == 1'b1) begin
            state <= S0;
        end else begin
            case (state)
                S0: begin
                    if (DATA_WIDTH == 8)
                        state <= S0;
                    else if (byte_rx_done)
                        state <= S1;
                    else
                        state <= S0;
                end
                S1: begin
                    if (timeout_flag)
                        state <= S0;
                    else if(byte_rx_done)
                        state <= S2;
                    else
                        state <= S1;
                end
                S2: begin
                    if (cnt >= DATA_WIDTH) begin
                        state <= S0;
                    end else begin
                        state <= S1;
                    end
                end
                default: state <= S0;
            endcase
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            data_r <= 0;
            data <= 0;
            cnt <= 0;
        end else begin
            case (state)
                S0: begin
                    rx_done <= 0;
                    data_r <= 0;
                    if (DATA_WIDTH == 8) begin
                        data <= data_byte;
						rx_done <= byte_rx_done;
                    end else if(byte_rx_done)begin
						cnt <= cnt + 9'd8;
						if(MSB_FIRST == 1)
							data_r <= {data_r[DATA_WIDTH - 1 - 8 : 0], data_byte};
						else
							data_r <= {data_byte, data_r[DATA_WIDTH - 1 : 8]};
					end
                end

                S1: begin
                    if (timeout_flag) begin
                        rx_done <= 1'b1;
                    end else if(byte_rx_done) begin
                        cnt <= cnt + 9'd8;
						if(MSB_FIRST == 1)
							data_r <= {data_r[DATA_WIDTH - 1 - 8 : 0], data_byte};
						else
							data_r <= {data_byte, data_r[DATA_WIDTH - 1 : 8]};
                    end
                end

                S2: begin
                    if(cnt >= DATA_WIDTH)begin
                        cnt <= 0;
                        data <= data_r;
                        rx_done <= 1;
                    end
                    else begin
                        rx_done <= 0;
                    end
                end

                default: begin
                    data_r <= 0;
                    data <= 0;
                    cnt <= 0;
                end
            endcase
        end
    end
endmodule //uart_data_rx
