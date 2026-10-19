// PPA harness: registered inputs around one INT8 dot-product unit (8 MACs per clock);
// the unit registers its own output.
module ppa_dot8_int8 (
    input wire clk,
    input wire [63:0] a, b,
    input wire [31:0] c,
    output wire [31:0] result
);
    reg [63:0] a_r, b_r;
    reg [31:0] c_r;
    int8_dot8_int32 unit (.clk(clk), .a(a_r), .b(b_r), .c(c_r), .result(result));
    always @(posedge clk) begin
        a_r <= a;
        b_r <= b;
        c_r <= c;
    end
endmodule
