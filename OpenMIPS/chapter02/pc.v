`timescale 1ns / 1ps
/*
    给出取指令地址,同时每个时钟周期取指令地址递增
*/
module pc(
    input wire clk,
    input wire rst,
    output reg[5: 0] pc,
    output reg       ce
);
    always @(posedge clk) begin
        if (rst == 1'b1) begin
            ce <= 1'b0;
        end else begin
            ce <= 1'b1;
        end
    end

    always @(posedge clk) begin
        if (ce == 1'b0) begin
            pc <= 6'h00;
        end else begin
            pc <= pc + 1'b1;
        end
    end
endmodule
