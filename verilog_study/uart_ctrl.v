/****************************************

  Filename            : uart_ctrl.v
  Description         : 控制模块, 根据串口接收模块输出的rx_done信号, 实现每成功接收一个字节数据, RAM的写入地址进行加一.
  Author              : wz
  Date                : 01-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module uart_ctrl (
    input wire clk,         // 系统时钟,50MHz
    input wire reset_n,     // 复位信号
    input wire key_flag,    // 按键标志信号
    input wire key_state,   // 按键状态信号
    input wire rx_done,     // 串口一个字节数据接收完成标志
    input wire tx_done,     // 串口一个字节数据发送完成标志

    output reg send_en,     // 串口发送使能信号
    output wire wea,        // dpram写使能信号
    output reg [7: 0] addra,  // dpram写地址
    output reg [7: 0] addrb   // dpram读地址
);
    wire reset;
    assign reset = ~reset_n;

    assign wea = rx_done;

    //====================== RAM写地址自增控制 ======================//
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            addra <= 8'd0;
        end else if (rx_done) begin
            addra <= addra + 1'b1;
        end else
            addra <= addra;
    end

    //====================== 按键控制数据发送的状态 ======================//
    // 按键按下启动连续读, 再次按下暂停读数据
    reg send_state;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            send_state <= 1'b0;
        end else if (key_flag && !key_state) begin
            send_state <= ~send_state;
        end else begin
            send_state <= send_state;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            addrb <= 8'b0;
        end else if (tx_done && (send_state == 1'b1)) begin
            addrb <= addrb + 8'd1;
        end else begin
            addrb <= addrb;
        end
    end

    //====================== 发送状态 ======================//
    // 第一个数的发送使能由按键产生, 之后的信号根据串口发送的完成信号产生
    // 由于RAM在读数据时Latentcy = 3, 所以给uart发送模块的使能信号也需要打3拍
    reg send_1st_en;
	reg tx_done_dly1;
	reg tx_done_dly2;
	reg tx_done_dly3;
	wire send_en_pre;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            send_1st_en <= 1'b0;
        end else if (key_flag && !key_state && send_state == 1'b0) begin
            send_1st_en <= 1'b1;
        end else begin
            send_1st_en <= 1'b0;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            tx_done_dly1 <= 1'b0;
            tx_done_dly2 <= 1'b0;
            tx_done_dly3 <= 1'b0;
        end else begin
            tx_done_dly1 <= tx_done;
            tx_done_dly2 <= tx_done_dly1;
            tx_done_dly3 <= tx_done_dly2;
        end
    end

    assign send_en_pre = send_1st_en | (tx_done_dly3 == 1'b1 && send_state == 1'b1);

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            send_en <= 1'b0;
        end else begin
            send_en <= send_en_pre;
        end
    end
endmodule //uart_ctrl
