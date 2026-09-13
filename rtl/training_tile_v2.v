module training_tile_v2 (
    input wire clk,
    input wire rst,
    input wire start,
    input wire transpose_a,
    input wire transpose_b,
    input wire accumulate,
    input wire [2:0] write_enable,
    input wire [3:0] write_addr,
    input wire [31:0] write_data,
    input wire [3:0] read_addr,
    output wire [31:0] read_data,
    output reg busy,
    output reg done,
    output reg [4:0] exception_flags
);
    reg [31:0] a_mem [0:15];
    reg [31:0] b_mem [0:15];
    reg trans_a, trans_b, issuing;
    reg [3:0] issue_index;
    wire [1:0] row = issue_index[1:0];
    wire [1:0] inner = issue_index[3:2];
    wire [3:0] a_addr = trans_a ? {inner, row} : {row, inner};
    reg [1:0] row_s1, row_s2, row_s3;
    reg [2:0] last;
    wire [3:0] lane_valid;
    wire [4:0] flags [0:3];
    wire [31:0] read_words [0:3];
    assign read_data = read_words[read_addr[1:0]];

    genvar lane;
    generate for (lane = 0; lane < 4; lane = lane + 1) begin: lanes
        localparam [1:0] COL = lane;
        reg [31:0] c_bank [0:3];
        wire [3:0] b_addr = trans_b ? {COL, inner} : {inner, COL};
        wire [31:0] result;
        integer i;
        assign read_words[lane] = c_bank[read_addr[3:2]];
        fp32_fma_pipe fma (
            .clk(clk), .rst(rst), .in_valid(busy && issuing),
            .a(a_mem[a_addr]), .b(b_mem[b_addr]), .c(c_bank[row]),
            .out_valid(lane_valid[lane]), .result(result), .exception_flags(flags[lane])
        );
        always @(posedge clk) begin
            if (!rst) begin
                if (busy) begin
                    if (lane_valid[lane]) c_bank[row_s3] <= result;
                end else if (start) begin
                    if (!accumulate)
                        for (i = 0; i < 4; i = i + 1) c_bank[i] <= 32'b0;
                end else if (write_enable[2] && write_addr[1:0] == COL) begin
                    c_bank[write_addr[3:2]] <= write_data;
                end
            end
        end
    end endgenerate

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
            issuing <= 1'b0;
            exception_flags <= 5'b0;
            issue_index <= 4'b0;
            trans_a <= 1'b0;
            trans_b <= 1'b0;
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
                exception_flags <= 5'b0;
                last <= 3'b0;
            end else begin
                if (write_enable[0]) a_mem[write_addr] <= write_data;
                if (write_enable[1]) b_mem[write_addr] <= write_data;
            end
        end
    end
endmodule
