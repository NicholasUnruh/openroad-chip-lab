// PPA harness: registered inputs around one NVFP4 block dot-product unit (32 MACs per clock: two
// 16-element blocks, each with E4M3 block scales); the unit registers its own outputs. The FP32
// per-tensor scale is applied once per output outside the unit.
module ppa_dot32_nvfp4 (
    input wire clk,
    input wire [127:0] a, b,
    input wire [15:0] scale_a, scale_b,
    input wire [31:0] c,
    output wire [31:0] result,
    output wire [4:0] exception_flags
);
    reg [127:0] a_r, b_r;
    reg [15:0] scale_a_r, scale_b_r;
    reg [31:0] c_r;
    mx_dot32_fp32 #(.ELEM(5)) unit (
        .clk(clk), .a(a_r), .b(b_r), .scale_a(scale_a_r), .scale_b(scale_b_r), .c(c_r),
        .result(result), .exception_flags(exception_flags)
    );
    always @(posedge clk) begin
        a_r <= a;
        b_r <= b;
        scale_a_r <= scale_a;
        scale_b_r <= scale_b;
        c_r <= c;
    end
endmodule
