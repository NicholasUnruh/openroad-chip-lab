module training_tile (
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
    reg [31:0] c_mem [0:15];
    reg [1:0] row, col, inner;
    reg trans_a, trans_b;
    wire [3:0] a_addr = trans_a ? {inner, row} : {row, inner};
    wire [3:0] b_addr = trans_b ? {col, inner} : {inner, col};
    wire [3:0] c_addr = {row, col};
    wire [32:0] a_rec, b_rec, c_rec, result_rec;
    wire [31:0] result;
    wire [4:0] flags;
    integer i;

    assign read_data = c_mem[read_addr];
    fNToRecFN #(8, 24) convert_a (.in(a_mem[a_addr]), .out(a_rec));
    fNToRecFN #(8, 24) convert_b (.in(b_mem[b_addr]), .out(b_rec));
    fNToRecFN #(8, 24) convert_c (.in(c_mem[c_addr]), .out(c_rec));
    // Round once to nearest-even; detect tininess after rounding.
    mulAddRecFN #(8, 24) fma (
        .control(1'b1), .op(2'b00), .roundingMode(3'b000),
        .a(a_rec), .b(b_rec), .c(c_rec),
        .out(result_rec), .exceptionFlags(flags)
    );
    recFNToFN #(8, 24) convert_result (.in(result_rec), .out(result));

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
            exception_flags <= 5'b0;
            row <= 2'b0;
            col <= 2'b0;
            inner <= 2'b0;
            trans_a <= 1'b0;
            trans_b <= 1'b0;
        end else begin
            done <= 1'b0;
            if (busy) begin
                c_mem[c_addr] <= result;
                exception_flags <= exception_flags | flags;
                inner <= inner + 2'd1;
                if (inner == 2'd3) begin
                    col <= col + 2'd1;
                    if (col == 2'd3) begin
                        row <= row + 2'd1;
                        if (row == 2'd3) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                        end
                    end
                end
            end else if (start) begin
                busy <= 1'b1;
                row <= 2'b0;
                col <= 2'b0;
                inner <= 2'b0;
                trans_a <= transpose_a;
                trans_b <= transpose_b;
                exception_flags <= 5'b0;
                if (!accumulate)
                    for (i = 0; i < 16; i = i + 1)
                        c_mem[i] <= 32'b0;
            end else begin
                if (write_enable[0]) a_mem[write_addr] <= write_data;
                if (write_enable[1]) b_mem[write_addr] <= write_data;
                if (write_enable[2]) c_mem[write_addr] <= write_data;
            end
        end
    end
endmodule
