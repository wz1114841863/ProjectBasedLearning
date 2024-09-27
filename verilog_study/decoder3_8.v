/****************************************

  Filename            : decoder3_8.v
  Description         : This is a decoder 3-8.
  Author              : wz
  Date                : 19-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module decoder3_8(
    input wire a,  // 输入端口a
    input wire b,  // 输入端口b
    input wire c,  // 输入端口c
    output reg [7: 0] out  // 译码输出结果
);
    // 3-8译码器实现
    always @(*) begin
        case ({a, b, c})
            3'b000: out = 8'b0000_0001;
            3'b001: out = 8'b0000_0010;
            3'b010: out = 8'b0000_0100;
            3'b011: out = 8'b0000_1000;
            3'b100: out = 8'b0001_0000;
            3'b101: out = 8'b0010_0000;
            3'b110: out = 8'b0100_0000;
            3'b111: out = 8'b1000_0000;
            default: out = 8'b0000_0000;
        endcase
    end

endmodule
