/****************************************

  Filename            : uart_byte_tx.v
  Description         : Uart Send Byte Module.
  Author              : wz
  Date                : 21-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_byte_tx (
    input wire clk,                 // 全局时钟50MHz
    input wire reset_n,             // 全部复位信号
    input wire [7: 0] data_byte,    // 预发送的串行传输数据
    input wire send_en,             // 发送使能信号
    input wire [2: 0] baud_set,     // 波特率使能信号

    output reg uart_tx,             // 串口发送信号输出
    output reg tx_done,             // 发送结束信号, 一个时钟周期高电平
    output reg uart_state           // 发送状态, 处于发送状态时为1
);
    assign reset = ~reset_n;

    //====================== Baud Clock Generator ======================//
    reg [15: 0] bps_DR;  // 波特率分频计数值
    reg [16: 0] div_cnt;  // 波特率时钟计数器
    reg bps_clk;  // 波特率生成时钟
    // 波特率查找表
    always @(posedge clk or posedge reset) begin
        if (reset)
            bps_DR <= 16'd5207;
        else begin
            case (baud_set)
                3'b000: bps_DR <= 16'd5207;
                3'b001: bps_DR <= 16'd2603;
                3'b010: bps_DR <= 16'd1301;
                3'b011: bps_DR <= 16'd867;
                3'b100: bps_DR <= 16'd433;
                default: bps_DR <= 16'd5207;
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

    //====================== Data Send ======================//
    reg [3: 0] bps_cnt;  // 根据波特率时钟进行计数
    reg [7: 0] data_byte_reg;  // 对输入数据进行寄存
    localparam START_BIT = 1'b0;
	localparam STOP_BIT = 1'b1;

    // bps counter
    always @(posedge clk or posedge reset) begin
        if (reset)
            bps_cnt <= 4'd0;
        else if (bps_cnt == 4'd11)
            bps_cnt <= 4'd0;
        else if (bps_clk)
            bps_cnt <= bps_cnt + 1'b1;
        else
            bps_cnt <= bps_cnt;
    end

    // 发送一个字节的数据的结束信号
    always @(posedge clk or posedge reset) begin
        if (reset)
            tx_done <= 1'b0;
        else if (bps_cnt == 4'd11)
            tx_done <= 1'b1;
        else
            tx_done <= 1'b0;
    end

    // 产生数据传输状态信号
    always @(posedge clk or posedge reset) begin
        if (reset)
            uart_state <= 1'b0;
        else if (send_en)
            uart_state <= 1'b1;
        else if (bps_cnt == 4'd11)
            uart_state <= 1'b0;
        else
            uart_state <= uart_state;
    end

    // 对输入数据进行寄存
    always @(posedge clk  or posedge reset) begin
        if (reset)
            data_byte_reg <= 8'd0;
        else if (send_en)
            data_byte_reg <= data_byte;
        else
            data_byte_reg <= data_byte_reg;
    end

    // 数据传输状态控制寄存器
    always @(posedge clk or posedge reset) begin
        if (reset)
            uart_tx <= 1'b1;
        else begin
            case (bps_cnt)
            4'd0: uart_tx <= 1'b1;
            4'd1: uart_tx <= START_BIT;
            4'd2: uart_tx <= data_byte_reg[0];
            4'd3: uart_tx <= data_byte_reg[1];
            4'd4: uart_tx <= data_byte_reg[2];
            4'd5: uart_tx <= data_byte_reg[3];
            4'd6: uart_tx <= data_byte_reg[4];
            4'd7: uart_tx <= data_byte_reg[5];
            4'd8: uart_tx <= data_byte_reg[6];
            4'd9: uart_tx <= data_byte_reg[7];
            4'd10: uart_tx <= STOP_BIT;
            default: uart_tx <= 1'b1;
            endcase
        end
    end
endmodule //uart_byte_tx
