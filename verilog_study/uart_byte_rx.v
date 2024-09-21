/****************************************

  Filename            : uart_byte_rx.v
  Description         : Uart Recv Module, Use multiple sampling
  Author              : wz
  Date                : 21-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module uart_byte_rx (
    input wire clk,               // 系统时钟50MHz
    input wire reset_n,           // 异步复位信号
    input wire [2: 0] baud_set,   // 波特率选择信号
    input wire uart_rx,           // 串行数据输入

    output reg [7: 0] data_byte,  // 并行数据输出
    output reg rx_done            // 接收结束信号
);
    assign reset = ~reset_n;
    reg [2: 0] START_BIT;
	reg [2: 0] STOP_BIT;
	reg [2: 0] data_byte_pre [7: 0];

    //====================== 双触发同步器 ======================//
    // 串口输入的信号相对于FPGA是异步信号, 为了避免亚稳态, 需要使用双触发同步器
    reg uart_rx_sync1;
    reg uart_rx_sync2;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            uart_rx_sync1 <= 1'b0;
            uart_rx_sync2 <= 1'b0;
        end else begin
            uart_rx_sync1 <= uart_rx;
            uart_rx_sync2 <= uart_rx_sync1;
        end
    end

    //====================== 边沿检测 ======================//
    // 使用两个D触发器和异或门实现边沿检测, 会带来两个时钟周期的检测延迟
    reg uart_rx_reg1;
    reg uart_rx_reg2;
    wire uart_rx_nedge;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            uart_rx_reg1 <= 1'b0;
            uart_rx_reg2 <= 1'b0;
        end else begin
            uart_rx_reg1 <= uart_rx_sync2;
            uart_rx_reg2 <= uart_rx_reg1;
        end
    end
    assign uart_rx_nedge = !uart_rx_reg1 & uart_rx_reg2;

    //====================== 采样时钟生成模块 ======================//
    // 实际的采样频率是波特率的16倍
    reg [15: 0] bps_DR;  // 波特率分频计数值
    reg [15: 0] div_cnt;  // 波特率时钟计数器
    reg uart_state;  // 串口信号传输状态
    reg bps_clk;  // 波特率生成时钟
    reg [7: 0] bps_cnt;  // 时钟计数器

    // 波特率查找表
    always @(posedge clk or posedge reset) begin
        if (reset)
            bps_DR <= 16'd324;
        else begin
            case (baud_set)
                3'b000: bps_DR <= 16'd324;
                3'b001: bps_DR <= 16'd162;
                3'b010: bps_DR <= 16'd80;
                3'b011: bps_DR <= 16'd53;
                3'b100: bps_DR <= 16'd26;
                default: bps_DR <= 16'd324;
            endcase
        end
    end

    // 波特率时钟计数器
    always @(posedge clk or posedge reset) begin
        if (reset)
            div_cnt <= 16'd0;
        else if (uart_state) begin
            if (div_cnt == bps_DR)
                div_cnt <= 16'd0;
            else
                div_cnt <= div_cnt + 1'b1;
        end else begin
            div_cnt <= 16'd0;
        end
    end

    // 波特率时钟生成, 脉冲信号
    always @(posedge clk or posedge reset) begin
        if (reset)
            bps_clk <= 1'b0;
        else if (div_cnt == 16'd1)
            bps_clk <= 1'b1;
        else
            bps_clk <= 1'b0;
    end

    // 采用时钟计数器, 计数器清零条件之一bps_cnt == 8'd159, 代表一个字节接收完毕
    always @(posedge clk or posedge reset) begin
        if (reset)
            bps_cnt <= 8'd0;
        else if (bps_cnt == 8'd159 || (bps_cnt == 8'd12 && (START_BIT > 2)))
            // 采样率是波特率的16倍, 每次传输10个bit, 故16 * 10 = 160, 0 ~ 159
            // 其中第12次是起始信号采样结束的时候, 判断起始信号是否有效(小于4)
            bps_cnt <= 8'd0;
        else if (bps_clk)
            bps_cnt <= bps_cnt + 1'b1;
        else
            bps_cnt <= bps_cnt;
    end

    always @(posedge clk or posedge reset) begin
        if (reset)
            rx_done <= 1'b0;
        else if (bps_cnt == 8'd159)
            rx_done <= 1'b1;
        else
            rx_done <= 1'b0;
    end

    //====================== 采样数据接收模块设计 ======================//
    // 位于中间的采样时间段对应的bps_cnt值分别为:6, 7, 8, 9, 10, 11
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            START_BIT <= 3'd0;
            data_byte_pre[0] <= 3'd0;
            data_byte_pre[1] <= 3'd0;
            data_byte_pre[2] <= 3'd0;
            data_byte_pre[3] <= 3'd0;
            data_byte_pre[4] <= 3'd0;
            data_byte_pre[5] <= 3'd0;
            data_byte_pre[6] <= 3'd0;
            data_byte_pre[7] <= 3'd0;
            STOP_BIT <= 3'd0;
        end else if (bps_clk) begin
            case(bps_cnt)
			0: begin
                START_BIT <= 3'd0;
                data_byte_pre[0] <= 3'd0;
                data_byte_pre[1] <= 3'd0;
                data_byte_pre[2] <= 3'd0;
                data_byte_pre[3] <= 3'd0;
                data_byte_pre[4] <= 3'd0;
                data_byte_pre[5] <= 3'd0;
                data_byte_pre[6] <= 3'd0;
                data_byte_pre[7] <= 3'd0;
                STOP_BIT <= 3'd0;
            end
			6 ,7 ,8 ,9 ,10,11: START_BIT <= START_BIT + uart_rx_sync2;
			22,23,24,25,26,27: data_byte_pre[0] <= data_byte_pre[0] + uart_rx_sync2;
			38,39,40,41,42,43: data_byte_pre[1] <= data_byte_pre[1] + uart_rx_sync2;
			54,55,56,57,58,59: data_byte_pre[2] <= data_byte_pre[2] + uart_rx_sync2;
			70,71,72,73,74,75: data_byte_pre[3] <= data_byte_pre[3] + uart_rx_sync2;
			86,87,88,89,90,91: data_byte_pre[4] <= data_byte_pre[4] + uart_rx_sync2;
			102,103,104,105,106,107: data_byte_pre[5] <= data_byte_pre[5] + uart_rx_sync2;
			118,119,120,121,122,123: data_byte_pre[6] <= data_byte_pre[6] + uart_rx_sync2;
			134,135,136,137,138,139: data_byte_pre[7] <= data_byte_pre[7] + uart_rx_sync2;
			150,151,152,153,154,155: STOP_BIT <= STOP_BIT + uart_rx_sync2;
			default: begin
                START_BIT <= START_BIT;
                data_byte_pre[0] <= data_byte_pre[0];
                data_byte_pre[1] <= data_byte_pre[1];
                data_byte_pre[2] <= data_byte_pre[2];
                data_byte_pre[3] <= data_byte_pre[3];
                data_byte_pre[4] <= data_byte_pre[4];
                data_byte_pre[5] <= data_byte_pre[5];
                data_byte_pre[6] <= data_byte_pre[6];
                data_byte_pre[7] <= data_byte_pre[7];
                STOP_BIT <= STOP_BIT;
            end
		    endcase
        end
    end

    //====================== 数据判断状态模块设计 ======================//
    always @(posedge clk or posedge reset) begin
        if(reset)
            data_byte <= 8'd0;
        else if (bps_cnt == 8'd159) begin
            data_byte[0] <= data_byte_pre[0][2];
            data_byte[1] <= data_byte_pre[1][2];
            data_byte[2] <= data_byte_pre[2][2];
            data_byte[3] <= data_byte_pre[3][2];
            data_byte[4] <= data_byte_pre[4][2];
            data_byte[5] <= data_byte_pre[5][2];
            data_byte[6] <= data_byte_pre[6][2];
            data_byte[7] <= data_byte_pre[7][2];
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset)
		    uart_state <= 1'b0;
        else if(uart_rx_nedge)
            uart_state <= 1'b1;
        else if(rx_done || (bps_cnt == 8'd12 && (START_BIT > 2)) || (bps_cnt == 8'd155 && (STOP_BIT < 3)))
            uart_state <= 1'b0;
        else
            uart_state <= uart_state;
    end

endmodule //uart_byte_rx
