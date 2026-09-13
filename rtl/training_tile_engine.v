module training_tile_engine #(
    parameter BUFFERED = 0,
    parameter BF16_INPUT = 0,
    parameter DIM_LOG2 = 2
) (
    input wire clk, rst, start,
    input wire transpose_a, transpose_b, accumulate,
    input wire job_bank, host_bank,
    input wire [2:0] write_enable,
    input wire [2*DIM_LOG2-1:0] write_addr,
    input wire [31:0] write_data,
    input wire [2*DIM_LOG2-1:0] read_addr,
    output wire [31:0] read_data,
    output wire write_ready,
    output reg busy, done,
    output reg [4:0] exception_flags
);
    localparam DIM = 1 << DIM_LOG2;
    localparam BANKS = BUFFERED ? 2 : 1;
    localparam ADDR_BITS = 2*DIM_LOG2 + (BUFFERED ? 1 : 0);
    localparam ROW_BITS = DIM_LOG2 + (BUFFERED ? 1 : 0);
    localparam INPUT_WIDTH = BF16_INPUT ? 16 : 32;
    reg [INPUT_WIDTH-1:0] a_mem [0:DIM*DIM*BANKS-1];
    reg [INPUT_WIDTH-1:0] b_mem [0:DIM*DIM*BANKS-1];
    reg active_bank;
    wire host = BUFFERED && host_bank;
    wire selected = BUFFERED && job_bank;
    wire [2*DIM_LOG2:0] host_addr = {host, write_addr};
    wire [2*DIM_LOG2:0] host_second_addr = {host, write_addr[2*DIM_LOG2-1:1], 1'b1};
    wire [INPUT_WIDTH-1:0] first_word = write_data[INPUT_WIDTH-1:0];
    wire [INPUT_WIDTH-1:0] second_word = {{(INPUT_WIDTH-16){1'b0}}, write_data[31:16]};
    assign write_ready = !rst && !(start && !busy) &&
                         (!busy || (BUFFERED && host != active_bank)) &&
                         (!BF16_INPUT || !(|write_enable[1:0]) ||
                          (!write_addr[0] && !write_enable[2]));
    reg trans_a, trans_b, issuing;
    reg [2*DIM_LOG2-1:0] issue_index;
    wire [DIM_LOG2-1:0] row = issue_index[DIM_LOG2-1:0];
    wire [DIM_LOG2-1:0] inner = issue_index[2*DIM_LOG2-1:DIM_LOG2];
    wire [2*DIM_LOG2:0] a_addr = {active_bank, trans_a ? {inner, row} : {row, inner}};
    reg [DIM_LOG2-1:0] row_s1, row_s2, row_s3;
    wire [DIM_LOG2:0] c_read_addr = {host, read_addr[2*DIM_LOG2-1:DIM_LOG2]};
    wire [DIM_LOG2:0] c_write_addr = {host, write_addr[2*DIM_LOG2-1:DIM_LOG2]};
    wire [DIM_LOG2:0] c_issue_addr = {active_bank, row};
    wire [DIM_LOG2:0] c_result_addr = {active_bank, row_s3};
    reg [2:0] last;
    wire [DIM-1:0] lane_valid;
    wire [4:0] flags [0:DIM-1];
    wire [31:0] read_words [0:DIM-1];
    reg [4:0] merged_flags;
    integer flag_lane;
    always @* begin
        merged_flags = 5'b0;
        for (flag_lane = 0; flag_lane < DIM; flag_lane = flag_lane + 1)
            merged_flags = merged_flags | flags[flag_lane];
    end
    assign read_data = read_words[read_addr[DIM_LOG2-1:0]];

    genvar lane;
    generate for (lane = 0; lane < DIM; lane = lane + 1) begin: lanes
        localparam [DIM_LOG2-1:0] COL = lane;
        reg [31:0] c_bank [0:DIM*BANKS-1];
        wire [2*DIM_LOG2:0] b_addr = {active_bank, trans_b ? {COL, inner} : {inner, COL}};
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
                    for (i = 0; i < DIM; i = i + 1) c_bank[DIM*selected+i] <= 32'b0;
                // The ready gate permits a host write only to a bank outside compute ownership.
                if (write_ready && write_enable[2] && write_addr[DIM_LOG2-1:0] == COL)
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
            issue_index <= {(2*DIM_LOG2){1'b0}};
            trans_a <= 1'b0;
            trans_b <= 1'b0;
            active_bank <= 1'b0;
            row_s1 <= {DIM_LOG2{1'b0}};
            row_s2 <= {DIM_LOG2{1'b0}};
            row_s3 <= {DIM_LOG2{1'b0}};
            last <= 3'b0;
        end else begin
            done <= 1'b0;
            if (busy) begin
                // At least four rows must separate updates to an accumulator; writeback precedes its next read.
                row_s1 <= row;
                row_s2 <= row_s1;
                row_s3 <= row_s2;
                last <= {last[1:0], issuing && (&issue_index)};
                if (issuing) begin
                    issue_index <= issue_index + 1'b1;
                    if (&issue_index) issuing <= 1'b0;
                end
                if (lane_valid[0]) begin
                    exception_flags <= exception_flags | merged_flags;
                    if (last[2]) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end
            end else if (start) begin
                busy <= 1'b1;
                issuing <= 1'b1;
                issue_index <= {(2*DIM_LOG2){1'b0}};
                trans_a <= transpose_a;
                trans_b <= transpose_b;
                active_bank <= selected;
                exception_flags <= 5'b0;
                last <= 3'b0;
            end
        end
    end
endmodule
