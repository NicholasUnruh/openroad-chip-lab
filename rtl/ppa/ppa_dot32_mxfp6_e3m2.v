// PPA harness: registered inputs around one OCP MXFP6 (E3M2) block dot-product unit
// (32 MACs per clock); the unit registers its own outputs.
module ppa_dot32_mxfp6_e3m2 (
    input wire clk,
    input wire [191:0] a, b,
    input wire [7:0] scale_a, scale_b,
    input wire [31:0] c,
    output wire [31:0] result,
    output wire [4:0] exception_flags
);
    reg [191:0] a_r, b_r;
    reg [7:0] scale_a_r, scale_b_r;
    reg [31:0] c_r;
    mx_dot32_fp32 #(.ELEM(3)) unit (
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
