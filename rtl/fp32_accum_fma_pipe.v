module fp32_accum_fma_pipe #(
    parameter BF16_INPUT = 0
) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [(BF16_INPUT ? 16 : 32)-1:0] a, b,
    input wire [31:0] c,
    output wire out_valid,
    output wire [31:0] result,
    output wire [4:0] exception_flags
);
    wire [32:0] ar, br, cr, rr;
    wire [23:0] mul_a, mul_b;
    wire [47:0] mul_c;
    wire [5:0] state;
    wire signed [9:0] exponent;
    wire [4:0] align_dist;
    wire [25:0] aligned_c;
    localparam SIG_WIDTH = BF16_INPUT ? 8 : 24;
    reg [SIG_WIDTH-1:0] a_s1, b_s1;
    wire [2*SIG_WIDTH-1:0] significand_product = a_s1 * b_s1;
    wire [31:0] a_fp32, b_fp32;
    reg [47:0] c_s1;
    reg [5:0] state_s1, state_s2;
    reg signed [9:0] exponent_s1, exponent_s2;
    reg [4:0] align_s1, align_s2;
    reg [25:0] aligned_s1, aligned_s2;
    reg [48:0] product_s2;
    wire invalid, is_nan, is_inf, is_zero, sign;
    wire signed [9:0] raw_exp;
    wire [26:0] raw_sig;
    reg invalid_s3, nan_s3, inf_s3, zero_s3, sign_s3;
    reg signed [9:0] exp_s3;
    reg [26:0] sig_s3;
    reg [2:0] valid;

    generate if (BF16_INPUT) begin: bf16
        assign a_fp32 = {a, 16'b0};
        assign b_fp32 = {b, 16'b0};
    end else begin: fp32
        assign a_fp32 = a;
        assign b_fp32 = b;
    end endgenerate
    fNToRecFN #(8, 24) convert_a (.in(a_fp32), .out(ar));
    fNToRecFN #(8, 24) convert_b (.in(b_fp32), .out(br));
    fNToRecFN #(8, 24) convert_c (.in(c), .out(cr));
    mulAddRecFNToRaw_preMul #(8, 24) pre (
        .control(1'b1), .op(2'b00), .roundingMode(3'b000),
        .a(ar), .b(br), .c(cr), .mulAddA(mul_a), .mulAddB(mul_b), .mulAddC(mul_c),
        .intermed_compactState(state), .intermed_sExp(exponent),
        .intermed_CDom_CAlignDist(align_dist), .intermed_highAlignedSigC(aligned_c)
    );
    mulAddRecFNToRaw_postMul #(8, 24) post (
        .intermed_compactState(state_s2), .intermed_sExp(exponent_s2),
        .intermed_CDom_CAlignDist(align_s2), .intermed_highAlignedSigC(aligned_s2),
        .mulAddResult(product_s2), .roundingMode(3'b000), .invalidExc(invalid),
        .out_isNaN(is_nan), .out_isInf(is_inf), .out_isZero(is_zero),
        .out_sign(sign), .out_sExp(raw_exp), .out_sig(raw_sig)
    );
    roundRawFNToRecFN #(8, 24, 0) round_result (
        .control(1'b1), .invalidExc(invalid_s3), .infiniteExc(1'b0),
        .in_isNaN(nan_s3), .in_isInf(inf_s3), .in_isZero(zero_s3),
        .in_sign(sign_s3), .in_sExp(exp_s3), .in_sig(sig_s3),
        .roundingMode(3'b000), .out(rr), .exceptionFlags(exception_flags)
    );
    recFNToFN #(8, 24) convert_result (.in(rr), .out(result));
    assign out_valid = valid[2];

    // Carry the full unrounded significand and exception state across every stage.
    always @(posedge clk) begin
        if (rst) begin
            valid <= 3'b0;
        end else begin
            valid <= {valid[1:0], in_valid};
            if (in_valid) begin
                // BF16 normalized significands have sixteen zero low bits, including subnormals.
                a_s1 <= mul_a[23:24-SIG_WIDTH];
                b_s1 <= mul_b[23:24-SIG_WIDTH];
                c_s1 <= mul_c;
                state_s1 <= state;
                exponent_s1 <= exponent;
                align_s1 <= align_dist;
                aligned_s1 <= aligned_c;
            end
            if (valid[0]) begin
                product_s2 <= {1'b0, significand_product, {(48-2*SIG_WIDTH){1'b0}}} + {1'b0, c_s1};
                state_s2 <= state_s1;
                exponent_s2 <= exponent_s1;
                align_s2 <= align_s1;
                aligned_s2 <= aligned_s1;
            end
            if (valid[1]) begin
                invalid_s3 <= invalid;
                nan_s3 <= is_nan;
                inf_s3 <= is_inf;
                zero_s3 <= is_zero;
                sign_s3 <= sign;
                exp_s3 <= raw_exp;
                sig_s3 <= raw_sig;
            end
        end
    end
endmodule
