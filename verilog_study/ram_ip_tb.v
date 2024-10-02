`timescale 1ns / 1ps
`define CLKA_PERIOD 20
`define CLKB_PERIOD 40
/****************************************

  Filename            : ram_ip_tb.v
  Description         : test bench for IP BRAM.
  Author              : wz
  Date                : 01-10-2024
  Version             : v1
  Version Description : First time edit.

****************************************/

module ram_ip_tb;
    reg clka;
    reg wea;
    reg [7: 0] addra;
    reg [7: 0] dina;

    reg clkb;
    reg [7: 0] addrb;
    wire[7: 0] doutb;

    integer i;

    blk_mem_ram_ip blk_mem_ram_ip_inst (
        .clka(clka),    // input wire clka
        .wea(wea),      // input wire [0: 0] wea
        .addra(addra),  // input wire [7: 0] addra
        .dina(dina),    // input wire [7: 0] dina

        .clkb(clkb),    // input wire clkb
        .addrb(addrb),  // input wire [7: 0] addrb
        .doutb(doutb)   // output wire [7: 0] doutb
    );

    initial clka = 1'b1;
    always #(`CLKA_PERIOD/2) clka = ~clka;

    initial clkb = 1'b1;
    always #(`CLKB_PERIOD/2) clkb = ~clkb;

    initial begin
        wea = 0;
        addra = 0;
        dina = 0;
        addrb = 255;
        #(`CLKA_PERIOD*10 + 1);

        wea=1;
        for (i = 0; i <= 15; i = i + 1) begin
            dina = 255 - i;
            addra = i;
            #`CLKA_PERIOD;
        end
        wea = 0;

        #1;
        for (i = 0; i <= 15; i = i + 1) begin
            addrb = i;
            #`CLKB_PERIOD;
        end

        #200;
        $stop;
    end
endmodule
