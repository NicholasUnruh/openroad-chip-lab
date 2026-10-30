// PPA harness: registered inputs around one fused OCP FP8 E4M3 dot-product unit (8 MACs per clock);
// the unit registers its own outputs.
module ppa_dot8_e4m3 (
    input wire clk,
    input wire [63:0] a, b,
    input wire [31:0] c,
    output wire [31:0] result,
    output wire [4:0] exception_flags
);
    reg [63:0] a_r, b_r;
    reg [31:0] c_r;
    fp_dot8_fp32 #(.FORMAT(1)) unit (
        .clk(clk), .a(a_r), .b(b_r), .c(c_r), .result(result), .exception_flags(exception_flags)
    );
    always @(posedge clk) begin
        a_r <= a;
        b_r <= b;
        c_r <= c;
    end
endmodule
