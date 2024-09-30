/****************************************

  Filename            : led_ctrl.v
  Description         : 底层模块, 加减法/移位计数器
  Author              : wz
  Date                : 28-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module led_ctrl (
    input wire clk,                 // 系统时钟
    input wire reset_n,             // 复位信号
    input wire key_add,             // 自加按键
    input wire key_sub,             // 自减按键
    input wire key_shift_left,      // 左移按键
    input wire key_shift_right,     // 右移按键

    output reg [7: 0] led           // 显示信号
);
    assign reset = ~reset_n;
    always @(posedge clk or posedge reset) begin
        if (reset)
            led <= 8'b0000_0000;
        else if (key_add)
            led <= led + 1'b1;
        else if (key_sub)
            led <= led - 1'b1;
        else if (key_shift_left)
            led <= (led << 1);
        else if (key_shift_right)
            led <= (led >> 1);
        else
            led <= led;
    end


endmodule //key_led
