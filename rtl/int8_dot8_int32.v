// Inference reference: c + sum of eight signed INT8 products, wrapping to INT32; two-clock latency.
module int8_dot8_int32 (
    input wire clk,
    input wire [63:0] a, b,
    input wire [31:0] c,
    output reg [31:0] result
);
    reg [127:0] p_s1;
    reg [31:0] c_s1;
    reg signed [31:0] sum;
    integer k;
    always @(posedge clk) begin
        for (k = 0; k < 8; k = k + 1) p_s1[16*k +: 16] <= $signed(a[8*k +: 8]) * $signed(b[8*k +: 8]);
        c_s1 <= c;
        result <= sum;
    end
    always @* begin
        sum = $signed(c_s1);
        for (k = 0; k < 8; k = k + 1) sum = sum + {{16{p_s1[16*k+15]}}, p_s1[16*k +: 16]};
    end
endmodule
