/****************************************

  Filename            : bin_counter.v
  Description         : This is a counter.
  Author              : wz
  Date                : 20-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module bin_counter(
    input wire clk,
    input wire reset_n,
    output reg flag
);
    assign reset = ~reset_n;
    parameter MAX_CNT = 10'b11_1110_1000;
    reg [9: 0] counter;

    // 计数器计数模块
    always @(posedge clk or posedge reset) begin
        if (reset)
            counter <= 10'b0;
        else if (counter == MAX_CNT)
            counter <= 10'b0;
        else
            counter <= counter + 1'b1;
    end

    // 根据计数翻转输出
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            flag <= 1'b1;
        end else if (counter == MAX_CNT) begin
            flag <= ~flag;
        end else begin
            flag <= flag;
        end
    end
endmodule
