/****************************************

  Filename            : fsm_hello.v
  Description         : Finial state mechine for string "hello" detection.
  Author              : wz
  Date                : 23-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module fsm_hello (
    input wire clk,             // 系统时钟
    input wire reset_n,         // 复位信号
    input wire [7: 0] data_in,  // 字符数据
    input wire  data_in_valid,  // 字符数据输入有效标志位

    output reg check_ok         // 检测成功标志位
);
    assign reset = ~reset_n;

    localparam CHECK_h = 5'b0_0001;
    localparam CHECK_e = 5'b0_0010;
    localparam CHECK_l1 = 5'b0_0100;
    localparam CHECK_l2 = 5'b0_1000;
    localparam CHECK_o = 5'b1_0000;

    reg [4: 0] state = 5'b0_0001;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            check_ok <= 1'b0;
            state <= CHECK_h;
        end else begin
            case (state)
                CHECK_h: begin
                    check_ok <= 1'b0;
                    if (data_in_valid && data_in == "h")
                        state <= CHECK_e;
                    else
                        state <= CHECK_h;
                end
                CHECK_e: begin
                    check_ok <= 1'b0;
                    if (data_in_valid && data_in == "e")
                        state <= CHECK_l1;
                    else if (data_in_valid && data_in == "h")
                        state <= CHECK_e;
                    else if (data_in_valid)
                        state <= CHECK_h;
                    else
                        state <= CHECK_e;
                end
                CHECK_l1: begin
                    check_ok <= 1'b0;
                    if (data_in_valid && data_in == "l")
                        state <= CHECK_l2;
                    else if (data_in_valid && data_in == "h")
                        state <= CHECK_e;
                    else if (data_in_valid)
                        state <= CHECK_h;
                    else
                        state <= CHECK_l1;
                end
                CHECK_l2: begin
                    check_ok <= 1'b0;
                    if (data_in_valid && data_in == "l")
                        state <= CHECK_o;
                    else if (data_in_valid && data_in == "h")
                        state <= CHECK_e;
                    else if (data_in_valid)
                        state <= CHECK_h;
                    else
                        state <= CHECK_l2;
                end
                CHECK_o: begin
                    if (data_in_valid && data_in == "h")
                        state <= CHECK_e;
                    else if (data_in_valid) begin
                        state <= CHECK_h;
                        if (data_in == "o")
                            check_ok <= 1'b1;
                        else
                            check_ok <= 1'b0;
                    end else
                        state <= CHECK_o;
                end
                default state <= CHECK_e;
            endcase
        end
    end
endmodule //fsm_hello
