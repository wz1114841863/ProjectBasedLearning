/****************************************

  Filename            : uart_data_tx.v
  Description         : 多字节串口发送模块
  Author              : wz
  Date                : 23-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module uart_data_tx #(
    parameter DATA_WIDTH = 8,  // 数据宽度
    parameter MSB_FIRST = 1
)(
    clk,        // 系统时钟
    reset_n,    // 复位信号

    data,       // 输入数据
    send_en,    // 发送使能
    baud_set,   // 波特率选择

    uart_tx,    // 串口发送数据
    tx_done,    // 传输完成标志
    uart_state  // 串口单字节状态
);
    input wire clk;
    input wire reset_n;
    input [DATA_WIDTH - 1 : 0] data;
    input send_en;
    input [2: 0] baud_set;
    output reg uart_tx;
    output reg tx_done;
    output reg uart_state;

    assign reset = ~reset_n;

    //====================== uart tx module instance ======================//
    reg byte_send_en;
    reg [7: 0] data_byte;
    reg byte_tx_done;
    uart_byte_tx  uart_byte_tx_inst (
        .clk(clk),
        .reset_n(reset_n),
        .data_byte(data_byte),
        .send_en(byte_send_en),
        .baud_set(baud_set),
        .uart_tx(uart_tx),
        .tx_done(byte_tx_done),
        .uart_state(uart_state)
    );

    //====================== finial state machine ======================//
    localparam S0 = 0;  // 等待发送请求
    localparam S1 = 1;  // 发起单字节数据发送
    localparam S2 = 2;  // 等待单字节数据发送完成
    localparam S3 = 3;  // 检查所有数据是否发送完成
    reg [8: 0] cnt;     // 发送字节计数器
	reg [1: 0] state;   // 当前状态
    reg [DATA_WIDTH - 1 : 0] data_recv;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S0;
        end else begin
            case (state)
                S0: begin
                    if (send_en)
                        state <= S1;
                    else
                        state <= S0;
                end
                S1: begin
                    state <= S2;
                end
                S2: begin
                    if (byte_tx_done)
                        state <= S3;
                    else
                        state <= S2;
                end
                S3: begin
                    if (cnt >= DATA_WIDTH)
                        state <= S0;
                    else
                        state <= S1;
                end
                default state <= S0;
            endcase
        end
    end

    always @(posedge clk or posedge reset) begin
        if(reset) begin
            data_byte <= 0;
            byte_send_en <= 0;
            cnt <= 0;
        end else begin
            case (state)
                S0: begin
                    data_byte <= 0;
                    byte_send_en <= 0;
                    cnt <= 0;
                    if (send_en)
                        data_recv <= data;
                    else
                        data_recv <= data_recv;
                end
                S1: begin
                    byte_send_en <= 1;
                    if(MSB_FIRST == 1)begin
						data_byte <= data_recv[DATA_WIDTH-1: DATA_WIDTH - 8];
						data_recv <= data_recv << 8;
					end
					else begin
						data_byte <= data_recv[7: 0];
						data_recv <= data_recv >> 8;
					end
                end
                S2: begin
                    byte_send_en <= 0;
                    if (byte_tx_done == 1'b1)
                        cnt <= cnt + 9'd8;
                end
                S3: begin
                    if(cnt >= DATA_WIDTH)begin
                        cnt <= 0;
                        tx_done <= 1;
                    end
                    else begin
                        tx_done <= 1;
                    end
                end
                default state <= S0;
            endcase
        end
    end


endmodule //uart_data_tx
