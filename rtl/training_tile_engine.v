module training_tile_engine #(
    parameter BUFFERED = 0,
    parameter BF16_INPUT = 0
) (
    input wire clk, rst, start,
    input wire transpose_a, transpose_b, accumulate,
    input wire job_bank, host_bank,
    input wire [2:0] write_enable,
    input wire [3:0] write_addr,
    input wire [31:0] write_data,
    input wire [3:0] read_addr,
    output wire [31:0] read_data,
    output wire write_ready,
    output reg busy, done,
    output reg [4:0] exception_flags
);
    localparam BANKS = BUFFERED ? 2 : 1;
    localparam ADDR_BITS = BUFFERED ? 5 : 4;
    localparam ROW_BITS = BUFFERED ? 3 : 2;
    localparam INPUT_WIDTH = BF16_INPUT ? 16 : 32;
    reg [INPUT_WIDTH-1:0] a_mem [0:16*BANKS-1];
    reg [INPUT_WIDTH-1:0] b_mem [0:16*BANKS-1];
    reg active_bank;
    wire host = BUFFERED && host_bank;
    wire selected = BUFFERED && job_bank;
    wire [4:0] host_addr = {host, write_addr};
    wire [4:0] host_second_addr = {host, write_addr[3:1], 1'b1};
    wire [INPUT_WIDTH-1:0] first_word = write_data[INPUT_WIDTH-1:0];
    wire [INPUT_WIDTH-1:0] second_word = {{(INPUT_WIDTH-16){1'b0}}, write_data[31:16]};
    assign write_ready = !rst && !(start && !busy) &&
                         (!busy || (BUFFERED && host != active_bank)) &&
                         (!BF16_INPUT || !(|write_enable[1:0]) ||
                          (!write_addr[0] && !write_enable[2]));
    reg trans_a, trans_b, issuing;
    reg [3:0] issue_index;
    wire [1:0] row = issue_index[1:0];
    wire [1:0] inner = issue_index[3:2];
    wire [4:0] a_addr = {active_bank, trans_a ? {inner, row} : {row, inner}};
    reg [1:0] row_s1, row_s2, row_s3;
    wire [2:0] c_read_addr = {host, read_addr[3:2]};
    wire [2:0] c_write_addr = {host, write_addr[3:2]};
    wire [2:0] c_issue_addr = {active_bank, row};
    wire [2:0] c_result_addr = {active_bank, row_s3};
    reg [2:0] last;
    wire [3:0] lane_valid;
    wire [4:0] flags [0:3];
    wire [31:0] read_words [0:3];
    assign read_data = read_words[read_addr[1:0]];

    genvar lane;
    generate for (lane = 0; lane < 4; lane = lane + 1) begin: lanes
        localparam [1:0] COL = lane;
        reg [31:0] c_bank [0:4*BANKS-1];
        wire [4:0] b_addr = {active_bank, trans_b ? {COL, inner} : {inner, COL}};
        wire [31:0] result;
        integer i;
        assign read_words[lane] = c_bank[c_read_addr[ROW_BITS-1:0]];
        fp32_accum_fma_pipe #(.BF16_INPUT(BF16_INPUT)) fma (
            .clk(clk), .rst(rst), .in_valid(busy && issuing),
            .a(a_mem[a_addr[ADDR_BITS-1:0]]), .b(b_mem[b_addr[ADDR_BITS-1:0]]),
            .c(c_bank[c_issue_addr[ROW_BITS-1:0]]),
            .out_valid(lane_valid[lane]), .result(result), .exception_flags(flags[lane])
        );
        always @(posedge clk) begin
            if (!rst) begin
                if (busy && lane_valid[lane]) c_bank[c_result_addr[ROW_BITS-1:0]] <= result;
                if (!busy && start && !accumulate)
                    for (i = 0; i < 4; i = i + 1) c_bank[4*selected+i] <= 32'b0;
                // The ready gate permits a host write only to a bank outside compute ownership.
                if (write_ready && write_enable[2] && write_addr[1:0] == COL)
                    c_bank[c_write_addr[ROW_BITS-1:0]] <= write_data;
            end
        end
    end endgenerate

    always @(posedge clk) begin
        if (write_ready) begin
            if (write_enable[0]) begin
                a_mem[host_addr[ADDR_BITS-1:0]] <= first_word;
                if (BF16_INPUT) a_mem[host_second_addr[ADDR_BITS-1:0]] <= second_word;
            end
            if (write_enable[1]) begin
                b_mem[host_addr[ADDR_BITS-1:0]] <= first_word;
                if (BF16_INPUT) b_mem[host_second_addr[ADDR_BITS-1:0]] <= second_word;
            end
        end
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
            issuing <= 1'b0;
            exception_flags <= 5'b0;
            issue_index <= 4'b0;
            trans_a <= 1'b0;
            trans_b <= 1'b0;
            active_bank <= 1'b0;
            row_s1 <= 2'b0;
            row_s2 <= 2'b0;
            row_s3 <= 2'b0;
            last <= 3'b0;
        end else begin
            done <= 1'b0;
            if (busy) begin
                // Four rows separate updates to an accumulator; writeback precedes its next read.
                row_s1 <= row;
                row_s2 <= row_s1;
                row_s3 <= row_s2;
                last <= {last[1:0], issuing && issue_index == 4'd15};
                if (issuing) begin
                    issue_index <= issue_index + 4'd1;
                    if (issue_index == 4'd15) issuing <= 1'b0;
                end
                if (lane_valid[0]) begin
                    exception_flags <= exception_flags | flags[0] | flags[1] | flags[2] | flags[3];
                    if (last[2]) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end
            end else if (start) begin
                busy <= 1'b1;
                issuing <= 1'b1;
                issue_index <= 4'b0;
                trans_a <= transpose_a;
                trans_b <= transpose_b;
                active_bank <= selected;
                exception_flags <= 5'b0;
                last <= 3'b0;
            end
        end
    end
endmodule
