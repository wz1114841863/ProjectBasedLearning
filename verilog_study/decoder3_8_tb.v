`timescale 1ns / 1ps
/****************************************

  Filename            : decoder3_8.v
  Description         : This is a decoder 3-8.
  Author              : wz
  Date                : 19-09-2023
  Version             : v1
  Version Description : First time edit.

****************************************/

module decoder3_8_tb;
    //Ports
    reg  a;
    reg  b;
    reg  c;
    wire [7: 0] out;

    decoder3_8  decoder3_8_inst (
        .a(a),
        .b(b),
        .c(c),
        .out(out)
    );

    initial begin
        a = 0; b = 0; c = 0;  // 在0时刻三个输入均为0
        #200;                 // 经过200ns的延时
        a = 0; b = 0; c = 1;
        #200;
        a = 0; b = 1; c = 0;
        #200;
        a = 1; b = 0; c = 0;
        #200;
        a = 0; b = 1; c = 1;
        #200;
        a = 1; b = 0; c = 1;
        #200;
        a = 1; b = 1; c = 0;
        #200;
        a = 1; b = 1; c = 1;

        $stop;                // 停止仿真
    end

endmodule
