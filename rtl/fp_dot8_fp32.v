// c + sum of eight products of a small float format, rounded once to FP32 (docs/training-v6.md);
// four-clock latency. FORMAT 0 = BF16, 1 = OCP FP8 E4M3, 2 = OCP FP8 E5M2.
// Term tops use t = unbiased top exponent + 252 so every finite term has 2 <= t <= 508.
module fp_dot8_fp32 #(
    parameter FORMAT = 0
) (
    input wire clk,
    input wire [8*(FORMAT == 0 ? 16 : 8)-1:0] a, b,
    input wire [31:0] c,
    output reg [31:0] result,
    output reg [4:0] exception_flags
);
    localparam EXP = FORMAT == 0 ? 8 : FORMAT == 1 ? 4 : 5;
    localparam MAN = FORMAT == 0 ? 7 : FORMAT == 1 ? 3 : 2;
    localparam BIAS = (1 << (EXP - 1)) - 1;
    localparam W = 1 + EXP + MAN;
    localparam PW = 2 * (MAN + 1);
    // E4M3 has no infinities and a single NaN mantissa; BF16 and E5M2 follow IEEE 754.
    localparam IEEE_SPECIALS = FORMAT != 1;
    // Product top t = ea + eb + T_OFFSET with subnormal exponent fields read as 1.
    localparam [8:0] T_OFFSET = 252 - 2 * BIAS - 2 * MAN + PW;

    // Stage 1: decode, exact significand products and special-value summary.
    reg [8*PW-1:0] p_s1;
    reg [71:0] t_s1;
    reg [7:0] sign_s1;
    reg [23:0] c_sig_s1;
    reg [8:0] c_t_s1;
    reg c_sign_s1, nan_s1, invalid_s1, pos_inf_s1, neg_inf_s1, all_zero_s1, all_negative_s1;

    reg [EXP-1:0] ea, eb;
    reg [MAN-1:0] ma, mb;
    reg a_zero, b_zero, a_inf, b_inf, a_nan, b_nan, prod_sign;
    reg nan, invalid, pos_inf, neg_inf, all_zero, all_negative;
    integer k, x;
    wire [7:0] ec = c[30:23];
    wire c_zero = ec == 8'd0 && c[22:0] == 23'd0;
    wire c_nan = &ec && c[22:0] != 23'd0;
    always @* begin
        nan = c_nan;
        invalid = c_nan && !c[22];
        pos_inf = &ec && c[22:0] == 23'd0 && !c[31];
        neg_inf = &ec && c[22:0] == 23'd0 && c[31];
        all_zero = c_zero;
        all_negative = c[31];
        for (k = 0; k < 8; k = k + 1) begin
            ea = a[W*k+MAN +: EXP];
            eb = b[W*k+MAN +: EXP];
            ma = a[W*k +: MAN];
            mb = b[W*k +: MAN];
            a_zero = ea == 0 && ma == 0;
            b_zero = eb == 0 && mb == 0;
            a_inf = IEEE_SPECIALS && &ea && ma == 0;
            b_inf = IEEE_SPECIALS && &eb && mb == 0;
            a_nan = &ea && (IEEE_SPECIALS ? ma != 0 : &ma);
            b_nan = &eb && (IEEE_SPECIALS ? mb != 0 : &mb);
            prod_sign = a[W*k+W-1] ^ b[W*k+W-1];
            nan = nan | a_nan | b_nan;
            invalid = invalid | (IEEE_SPECIALS && a_nan && !ma[MAN-1]) | (IEEE_SPECIALS && b_nan && !mb[MAN-1]) |
                      (a_inf && b_zero) | (b_inf && a_zero);
            if ((a_inf || b_inf) && !a_nan && !b_nan && !a_zero && !b_zero) begin
                pos_inf = pos_inf | !prod_sign;
                neg_inf = neg_inf | prod_sign;
            end
            all_zero = all_zero & (a_zero | b_zero);
            all_negative = all_negative & prod_sign;
        end
    end

    always @(posedge clk) begin
        for (k = 0; k < 8; k = k + 1) begin
            p_s1[PW*k +: PW] <= {a[W*k+MAN +: EXP] != 0, a[W*k +: MAN]} * {b[W*k+MAN +: EXP] != 0, b[W*k +: MAN]};
            t_s1[9*k +: 9] <= (a[W*k+MAN +: EXP] == 0 ? 9'd1 : {{(9-EXP){1'b0}}, a[W*k+MAN +: EXP]}) +
                              (b[W*k+MAN +: EXP] == 0 ? 9'd1 : {{(9-EXP){1'b0}}, b[W*k+MAN +: EXP]}) + T_OFFSET;
            sign_s1[k] <= a[W*k+W-1] ^ b[W*k+W-1];
        end
        c_sig_s1 <= {ec != 8'd0, c[22:0]};
        c_t_s1 <= {1'b0, ec == 8'd0 ? 8'd1 : ec} + 9'd126;
        c_sign_s1 <= c[31];
        nan_s1 <= nan;
        invalid_s1 <= invalid | (pos_inf & neg_inf);
        pos_inf_s1 <= pos_inf;
        neg_inf_s1 <= neg_inf;
        all_zero_s1 <= all_zero;
        all_negative_s1 <= all_negative;
    end

    // Stage 2: largest top, then left-justify each term in the 32-bit frame and truncate.
    reg [8:0] top;
    reg [9:0] distance;
    reg [31:0] justified, low_mask;
    reg [287:0] aligned;
    reg [8:0] term_sign;
    reg truncated;
    always @* begin
        top = c_sig_s1 != 24'd0 ? c_t_s1 : 9'd0;
        for (k = 0; k < 8; k = k + 1)
            if (p_s1[PW*k +: PW] != 0 && t_s1[9*k +: 9] > top) top = t_s1[9*k +: 9];
        truncated = 1'b0;
        for (k = 0; k < 9; k = k + 1) begin
            justified = k == 8 ? {c_sig_s1, 8'd0} : {p_s1[PW*k +: PW], {(32-PW){1'b0}}};
            distance = {1'b0, top} - {1'b0, k == 8 ? c_t_s1 : t_s1[9*k +: 9]};
            aligned[32*k +: 32] = distance >= 10'd32 ? 32'd0 : justified >> distance[4:0];
            // Bits below the shift distance are the ones truncated away.
            for (x = 0; x < 32; x = x + 1) low_mask[x] = {22'd0, distance} > x;
            truncated = truncated | (justified & low_mask) != 32'd0;
            term_sign[k] = k == 8 ? c_sign_s1 : sign_s1[k];
        end
    end

    reg [287:0] aligned_s2;
    reg [8:0] sign_s2;
    reg [8:0] top_s2;
    reg truncated_s2, nan_s2, invalid_s2, pos_inf_s2, neg_inf_s2, all_zero_s2, all_negative_s2;
    always @(posedge clk) begin
        aligned_s2 <= aligned;
        sign_s2 <= term_sign;
        top_s2 <= top;
        truncated_s2 <= truncated;
        nan_s2 <= nan_s1;
        invalid_s2 <= invalid_s1;
        pos_inf_s2 <= pos_inf_s1;
        neg_inf_s2 <= neg_inf_s1;
        all_zero_s2 <= all_zero_s1;
        all_negative_s2 <= all_negative_s1;
    end

    // Stage 3: exact signed frame sum; nine 32-bit terms need 37 bits.
    reg signed [36:0] sum;
    always @* begin
        sum = 37'sd0;
        for (k = 0; k < 9; k = k + 1)
            sum = sign_s2[k] ? sum - $signed({5'b0, aligned_s2[32*k +: 32]}) :
                                 sum + $signed({5'b0, aligned_s2[32*k +: 32]});
    end

    reg signed [36:0] sum_s3;
    reg [8:0] top_s3;
    reg truncated_s3, nan_s3, invalid_s3, pos_inf_s3, neg_inf_s3, all_zero_s3, all_negative_s3;
    always @(posedge clk) begin
        sum_s3 <= sum;
        top_s3 <= top_s2;
        truncated_s3 <= truncated_s2;
        nan_s3 <= nan_s2;
        invalid_s3 <= invalid_s2;
        pos_inf_s3 <= pos_inf_s2;
        neg_inf_s3 <= neg_inf_s2;
        all_zero_s3 <= all_zero_s2;
        all_negative_s3 <= all_negative_s2;
    end

    // Stage 4: normalize and round once. The frame LSB has exponent top - 284.
    reg [35:0] mag;
    reg [5:0] lead, trailing;
    reg signed [11:0] shift, exp_field;
    reg [71:0] shifted;
    reg [8:0] amount;
    reg round_bit, sticky, inexact;
    reg [24:0] sig;
    reg [31:0] rounded;
    reg overflow;
    always @* begin
        mag = sum_s3[36] ? -sum_s3[35:0] : sum_s3[35:0];
        lead = 6'd0;
        trailing = 6'd36;
        for (k = 0; k < 36; k = k + 1)
            if (mag[k]) lead = k[5:0];
        for (k = 35; k >= 0; k = k - 1)
            if (mag[k]) trailing = k[5:0];
        // Rounding LSB: 24 significant bits, but never below the 2^-149 subnormal grid.
        shift = $signed({6'b0, lead}) - 12'sd23;
        if ($signed(12'sd135) - $signed({3'b0, top_s3}) > shift)
            shift = $signed(12'sd135) - $signed({3'b0, top_s3});
        // One right shift of mag * 2^36 covers both directions (shift is -23..133); the
        // round bit is mag[shift-1] and sticky is any lower set bit.
        amount = shift[8:0] + 9'd36;
        shifted = {mag, 36'd0} >> amount;
        round_bit = shift > 0 && shift <= 12'sd36 && mag[shift[5:0] - 6'd1];
        sticky = shift > 0 && $signed({6'b0, trailing}) < shift - 12'sd1;
        inexact = round_bit | sticky | truncated_s3;
        sig = shifted[24:0] + {24'd0, round_bit && (sticky || shifted[0])};
        exp_field = shift + $signed({3'b0, top_s3}) - 12'sd134;
        if (sig[24]) begin
            sig = sig >> 1;
            exp_field = exp_field + 12'sd1;
        end
        if (!sig[23]) exp_field = 12'sd0;
        overflow = exp_field >= 12'sd255;
        rounded = overflow ? {sum_s3[36], 8'hff, 23'd0} : {sum_s3[36], exp_field[7:0], sig[22:0]};
    end

    always @(posedge clk) begin
        if (nan_s3 || invalid_s3) begin
            result <= 32'h7fc00000;
            exception_flags <= {invalid_s3, 4'b0};
        end else if (pos_inf_s3 || neg_inf_s3) begin
            result <= {neg_inf_s3, 8'hff, 23'd0};
            exception_flags <= 5'b0;
        end else if (all_zero_s3) begin
            result <= {all_negative_s3, 31'd0};
            exception_flags <= 5'b0;
        end else if (sum_s3 == 37'sd0) begin
            result <= 32'd0;
            exception_flags <= {4'b0, truncated_s3};
        end else begin
            result <= rounded;
            exception_flags <= {2'b0, overflow, $signed({6'b0, lead}) + $signed({3'b0, top_s3}) < 12'sd158 && inexact,
                                inexact | overflow};
        end
    end
endmodule
