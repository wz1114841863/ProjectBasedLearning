/****************************************

  Filename            : key_filter.v
  Description         : 利用状态机实现按键消抖
  Author              : wz
  Date                : 27-09-2024
  Version             : v1
  Version Description : First time edit.

****************************************/
module key_filter (
    input wire clk,         // 系统时钟
    input wire reset_n,     // 复位信号
    input wire key_in,      // 按键输入信号

    output reg key_flag,    // 按键状态切换标志
    output reg key_state    // 按键状态标志
);
    assign reset = ~reset_n;

    //====================== Dual trigger ======================//
    reg key_in_sync1;
    reg key_in_sync2;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            key_in_sync1 <= 1'b0;
            key_in_sync2 <= 1'b0;
        end else begin
            key_in_sync1 <= key_in;
            key_in_sync2 <= key_in_sync1;
        end
    end

    //====================== Edge detection ======================//
    reg key_in_reg1;
    reg key_in_reg2;
    wire key_in_nedge;
    wire key_in_pedge;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            key_in_reg1 <= 1'b0;
            key_in_reg2 <= 1'b0;
        end else begin
            key_in_reg1 <= key_in_sync2;
            key_in_reg2 <= key_in_reg1;
        end
    end

    assign key_in_nedge = (!key_in_reg1) & key_in_reg2;
    assign key_in_pedge = key_in_reg1 & (!key_in_reg2);

    //====================== State Machine ======================//
    reg [19: 0] cnt;
    reg en_cnt;
    reg cnt_full;

    reg [3: 0] state;
    localparam IDLE = 4'b0001;
    localparam FILTER1 = 4'b0010;
    localparam DOWN = 4'b0100;
    localparam FILTER2 = 4'b1000;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            cnt <= 20'b0;
        end else if (en_cnt) begin
            cnt <= cnt + 1'b1;
        end else
            cnt <= 20'b0;
    end

    always @(posedge clk or posedge reset) begin
        if (reset)
            cnt_full <= 1'b0;
        else if (cnt == 20'd999_999)
            cnt_full <= 1'b1;
        else
            cnt_full <= 1'b0;
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            en_cnt <= 1'b0;
            key_flag <= 1'b0;
            key_state <= 1'b1;
        end else begin
            case (state)
                IDLE: begin
                    key_flag <= 1'b0;
                    if (key_in_nedge) begin
                        state <= FILTER1;
                        en_cnt <= 1'b1;
                    end else
                        state <= IDLE;
                end

                FILTER1: begin
                    if (cnt_full) begin
                        key_flag <= 1'b1;
                        key_state <= 1'b0;
                        en_cnt <= 1'b0;
                        state <= DOWN;
                    end else if (key_in_pedge) begin
                        en_cnt <= 1'b0;
                        state <= IDLE;
                    end else
                        state <= FILTER1;
                end

                DOWN: begin
                    key_flag <= 1'b0;
                    if (key_in_pedge) begin
                        en_cnt <= 1'b1;
                        state <= FILTER2;
                    end else
                        state <= DOWN;
                end

                FILTER2: begin
                    if (cnt_full) begin
                        key_flag <= 1'b1;
                        key_state <= 1'b1;
                        en_cnt <= 1'b0;
                        state <= IDLE;
                    end else if (key_in_nedge) begin
                        en_cnt <= 1'b0;
                        state <= DOWN;
                    end else
                        state <= FILTER2;
                end
                default: begin
                    state <= IDLE;
                    en_cnt <= 1'b0;
                    key_flag <= 1'b0;
                    key_state <= 1'b1;
                end
            endcase
        end
    end
endmodule //key_filter
