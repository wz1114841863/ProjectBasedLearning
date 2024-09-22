/****************************************

  Filename            : uart_rx_counter.v
  Description         : generator signal by Timer Word and Ctrl Word.
  Author              : wz
  Date                : 22-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module uart_rx_counter (
    input wire clk,
    input wire reset_n,
    input wire [7: 0] ctrl_set,
    input wire [31: 0] time_set,

    output reg signal
);
    assign reset = ~reset_n;

    reg [31: 0] counter;
    always @(posedge clk or posedge reset) begin
        if (reset)
            counter <= 0;
        else if(counter >= time_set - 1)
            counter <= 0;
        else
        counter <= counter + 1'b1;
    end

    reg [2: 0] counter2;
    always @(posedge clk or posedge reset) begin
        if(reset)
            counter2 <= 0;
        else if(counter >= time_set - 1)
            counter2 <= counter2 + 1'b1;
    end

    always@(posedge clk or posedge reset) begin
        if(reset)
            signal <= 0;
        else case(counter2)
            0: signal <= ctrl_set[0];
            1: signal <= ctrl_set[1];
            2: signal <= ctrl_set[2];
            3: signal <= ctrl_set[3];
            4: signal <= ctrl_set[4];
            5: signal <= ctrl_set[5];
            6: signal <= ctrl_set[6];
            7: signal <= ctrl_set[7];
            default: signal <= signal;
        endcase
    end

endmodule //uart_rx_counter
