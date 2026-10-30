// PPA harness: registered inputs and outputs around one BF16 FMA lane (1 MAC per clock).
module ppa_fma_bf16 (
    input wire clk, rst, in_valid,
    input wire [15:0] a, b,
    input wire [31:0] c,
    output reg out_valid,
    output reg [31:0] result,
    output reg [4:0] exception_flags
);
    reg valid_r;
    reg [15:0] a_r, b_r;
    reg [31:0] c_r;
    wire unit_valid;
    wire [31:0] unit_result;
    wire [4:0] unit_flags;
    fp32_accum_fma_pipe #(.BF16_INPUT(1)) unit (
        .clk(clk), .rst(rst), .in_valid(valid_r), .a(a_r), .b(b_r), .c(c_r),
        .out_valid(unit_valid), .result(unit_result), .exception_flags(unit_flags)
    );
    always @(posedge clk) begin
        valid_r <= in_valid;
        a_r <= a;
        b_r <= b;
        c_r <= c;
        out_valid <= unit_valid;
        result <= unit_result;
        exception_flags <= unit_flags;
    end
endmodule
