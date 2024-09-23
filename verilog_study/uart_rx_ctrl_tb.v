`timescale 1ns/1ps
// `define delay_time 8680
/****************************************

  Filename            : uart_rx_ctrl_tb.v
  Description         : TestBench for uart_rx_ctrl.v
  Author              : wz
  Date                : 22-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module uart_rx_ctrl_tb;
    reg  clk;
    reg  reset_n;
    reg  uart_rx;
    wire signal;
    wire [31: 0] delay_time;

    uart_rx_ctrl  uart_rx_ctrl_inst (
        .clk(clk),
        .reset_n(reset_n),
        .uart_rx(uart_rx),
        .signal(signal)
    );
    parameter baud_set = 3'd4;
    assign delay_time = (baud_set == 3'd0) ? 32'd104166 :
                        (baud_set == 3'd1) ? 32'd52083 :
                        (baud_set == 3'd2) ? 32'd26041 :
                        (baud_set == 3'd3) ? 32'd17361 :
                                             32'd8680;
    initial clk = 1;
    always #10 clk = ~clk;

    initial begin
        reset_n = 0;
        uart_rx = 1;
        #201;
        reset_n = 1;
        #200;

        uart_tx_byte(8'h55);
        #(delay_time*10);
        uart_tx_byte(8'ha5);
        #(delay_time*10);
        uart_tx_byte(8'h55);
        #(delay_time*10);
        uart_tx_byte(8'ha5);
        #(delay_time*10);
        uart_tx_byte(8'h00);
        #(delay_time*10);
        uart_tx_byte(8'h00);
        #(delay_time*10);
        uart_tx_byte(8'hc3);
        #(delay_time*10);
        uart_tx_byte(8'h50);
        #(delay_time*10);
        uart_tx_byte(8'haa);
        #(delay_time*10);
        uart_tx_byte(8'hf0);
        #(delay_time*10);

        uart_tx_byte(8'h55);
        #(delay_time*10);
        uart_tx_byte(8'ha5);
        #(delay_time*10);
        uart_tx_byte(8'h9a);
        #(delay_time*10);
        uart_tx_byte(8'h78);
        #(delay_time*10);
        uart_tx_byte(8'h56);
        #(delay_time*10);
        uart_tx_byte(8'h34);
        #(delay_time*10);
        uart_tx_byte(8'h12);
        #(delay_time*10);
        uart_tx_byte(8'hf1);
        #(delay_time*10);
        #15000000;
        $stop;
    end

    // 模拟UART串口数据发送
    task uart_tx_byte;
        input [7: 0] tx_data;
        begin
            uart_rx = 1;
            #20;
            uart_rx = 0;
            #delay_time;
            uart_rx = tx_data[0];
            #delay_time;
            uart_rx = tx_data[1];
            #delay_time;
            uart_rx = tx_data[2];
            #delay_time;
            uart_rx = tx_data[3];
            #delay_time;
            uart_rx = tx_data[4];
            #delay_time;
            uart_rx = tx_data[5];
            #delay_time;
            uart_rx = tx_data[6];
            #delay_time;
            uart_rx = tx_data[7];
            #delay_time;
            uart_rx = 1;
            #delay_time;
        end
    endtask
endmodule
