// c + sum of eight products of a small float format, rounded once to FP32 (docs/training-v6.md);
// six-clock latency. FORMAT 0 = BF16, 1 = OCP FP8 E4M3, 2 = OCP FP8 E5M2. FRAME (24 to 40) is
// the alignment width below the largest term's top; bits below it are truncated.
// Term tops use t = unbiased top exponent + 252 so every finite term has 2 <= t <= 508.
module fp_dot8_fp32 #(
    parameter FORMAT = 0,
    parameter FRAME = 32
) (
    input wire clk,
    input wire [8*(FORMAT == 0 ? 16 : 8)-1:0] a, b,
    input wire [31:0] c,
    output wire [31:0] result,
    output wire [4:0] exception_flags
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

    // Stage 1: decode, exact significand products, term tops and specials.
    function [8:0] max9;
        input [8:0] x, y;
        max9 = x > y ? x : y;
    endfunction

    reg [8*PW-1:0] p_s1;
    reg [80:0] t_s1;
    reg [7:0] sign_s1;
    reg [23:0] c_sig_s1;
    reg c_sign_s1, nan_s1, invalid_s1, pos_inf_s1, neg_inf_s1, all_zero_s1, all_negative_s1;

    reg [EXP-1:0] ea, eb;
    reg [MAN-1:0] ma, mb;
    reg a_zero, b_zero, a_inf, b_inf, a_nan, b_nan, prod_sign;
    reg nan, invalid, pos_inf, neg_inf, all_zero, all_negative;
    reg [80:0] t;          // nine 9-bit term tops; zero terms read as 0, below every finite top
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
            t[9*k +: 9] = a_zero || b_zero ? 9'd0 :
                          (ea == 0 ? 9'd1 : {{(9-EXP){1'b0}}, ea}) + (eb == 0 ? 9'd1 : {{(9-EXP){1'b0}}, eb}) + T_OFFSET;
        end
        t[72 +: 9] = c_zero ? 9'd0 : {1'b0, ec == 8'd0 ? 8'd1 : ec} + 9'd126;
    end

    always @(posedge clk) begin
        for (k = 0; k < 8; k = k + 1) begin
            p_s1[PW*k +: PW] <= {a[W*k+MAN +: EXP] != 0, a[W*k +: MAN]} * {b[W*k+MAN +: EXP] != 0, b[W*k +: MAN]};
            sign_s1[k] <= a[W*k+W-1] ^ b[W*k+W-1];
        end
        t_s1 <= t;
        c_sig_s1 <= {ec != 8'd0, c[22:0]};
        c_sign_s1 <= c[31];
        nan_s1 <= nan;
        invalid_s1 <= invalid | (pos_inf & neg_inf);
        pos_inf_s1 <= pos_inf;
        neg_inf_s1 <= neg_inf;
        all_zero_s1 <= all_zero;
        all_negative_s1 <= all_negative;
    end

    // Stage 2: the largest term top (balanced max tree) and each term's distance below it.
    reg [8:0] top;
    reg [9:0] distance;
    localparam [31:0] FRAME32 = FRAME;
    localparam [5:0] FRAME6 = FRAME32[5:0];
    reg [53:0] term_dist;  // nine 6-bit distances, saturated at FRAME
    always @* begin
        top = max9(max9(max9(max9(t_s1[0 +: 9], t_s1[9 +: 9]), max9(t_s1[18 +: 9], t_s1[27 +: 9])),
                        max9(max9(t_s1[36 +: 9], t_s1[45 +: 9]), max9(t_s1[54 +: 9], t_s1[63 +: 9]))), t_s1[72 +: 9]);
        for (k = 0; k < 9; k = k + 1) begin
            distance = {1'b0, top} - {1'b0, t_s1[9*k +: 9]};
            term_dist[6*k +: 6] = distance >= FRAME32[9:0] ? FRAME6 : distance[5:0];
        end
    end
    reg [8*PW-1:0] p_s2;
    reg [53:0] term_dist_s2;
    reg [7:0] sign_s2;
    reg [23:0] c_sig_s2;
    reg [8:0] top_s2;
    reg c_sign_s2, nan_s2, invalid_s2, pos_inf_s2, neg_inf_s2, all_zero_s2, all_negative_s2;
    always @(posedge clk) begin
        p_s2 <= p_s1;
        term_dist_s2 <= term_dist;
        sign_s2 <= sign_s1;
        c_sig_s2 <= c_sig_s1;
        top_s2 <= top;
        c_sign_s2 <= c_sign_s1;
        nan_s2 <= nan_s1;
        invalid_s2 <= invalid_s1;
        pos_inf_s2 <= pos_inf_s1;
        neg_inf_s2 <= neg_inf_s1;
        all_zero_s2 <= all_zero_s1;
        all_negative_s2 <= all_negative_s1;
    end

    // Stage 3: left-justify each term in the frame, shift by its distance, truncate.
    reg [FRAME-1:0] justified, low_mask, c_frame, p_frame;
    reg [5:0] d;
    reg [9*FRAME-1:0] aligned;
    reg [8:0] term_sign;
    reg truncated;
    always @* begin
        truncated = 1'b0;
        for (k = 0; k < 9; k = k + 1) begin
            c_frame = {FRAME{1'b0}};
            c_frame[23:0] = c_sig_s2;
            p_frame = {FRAME{1'b0}};
            p_frame[PW-1:0] = p_s2[PW*k +: PW];
            justified = k == 8 ? c_frame << (FRAME - 24) : p_frame << (FRAME - PW);
            d = term_dist_s2[6*k +: 6];
            // d saturates at FRAME, where the shift already yields zero.
            aligned[FRAME*k +: FRAME] = justified >> d;
            // Bits below the shift distance are the ones truncated away.
            for (x = 0; x < FRAME; x = x + 1) low_mask[x] = {26'd0, d} > x;
            truncated = truncated | (justified & low_mask) != {FRAME{1'b0}};
            term_sign[k] = k == 8 ? c_sign_s2 : sign_s2[k];
        end
    end

    reg [9*FRAME-1:0] aligned_s3;
    reg [8:0] sign_s3;
    reg [8:0] top_s3;
    reg truncated_s3, nan_s3, invalid_s3, pos_inf_s3, neg_inf_s3, all_zero_s3, all_negative_s3;
    always @(posedge clk) begin
        aligned_s3 <= aligned;
        sign_s3 <= term_sign;
        top_s3 <= top_s2;
        truncated_s3 <= truncated;
        nan_s3 <= nan_s2;
        invalid_s3 <= invalid_s2;
        pos_inf_s3 <= pos_inf_s2;
        neg_inf_s3 <= neg_inf_s2;
        all_zero_s3 <= all_zero_s2;
        all_negative_s3 <= all_negative_s2;
    end

    // Stage 4: exact signed frame sum as a balanced tree; nine FRAME-bit terms need FRAME+5 bits.
    localparam SW = FRAME + 5;
    localparam [31:0] LSB_OFFSET32 = 252 + FRAME;  // frame LSB = top - FRAME
    localparam signed [11:0] LSB_OFFSET = LSB_OFFSET32[11:0];
    reg [9*SW-1:0] term;  // nine signed SW-bit terms
    reg signed [SW-1:0] sum;
    always @* begin
        for (k = 0; k < 9; k = k + 1)
            term[SW*k +: SW] = sign_s3[k] ? -{5'b0, aligned_s3[FRAME*k +: FRAME]} : {5'b0, aligned_s3[FRAME*k +: FRAME]};
        sum = (((term[0 +: SW] + term[SW +: SW]) + (term[2*SW +: SW] + term[3*SW +: SW])) +
               ((term[4*SW +: SW] + term[5*SW +: SW]) + (term[6*SW +: SW] + term[7*SW +: SW]))) + term[8*SW +: SW];
    end

    reg signed [SW-1:0] sum_s4;
    reg [8:0] top_s4;
    reg truncated_s4, nan_s4, invalid_s4, pos_inf_s4, neg_inf_s4, all_zero_s4, all_negative_s4;
    always @(posedge clk) begin
        sum_s4 <= sum;
        top_s4 <= top_s3;
        truncated_s4 <= truncated_s3;
        nan_s4 <= nan_s3;
        invalid_s4 <= invalid_s3;
        pos_inf_s4 <= pos_inf_s3;
        neg_inf_s4 <= neg_inf_s3;
        all_zero_s4 <= all_zero_s3;
        all_negative_s4 <= all_negative_s3;
    end

    // Stages 5-6: round once in fp32_round, or pass NaN, infinity or a zero sum.
    wire [SW-2:0] mag = sum_s4[SW-1] ? -sum_s4[SW-2:0] : sum_s4[SW-2:0];
    wire nan_out = nan_s4 || invalid_s4;
    wire inf_out = pos_inf_s4 || neg_inf_s4;
    wire special = nan_out || inf_out || all_zero_s4 || sum_s4 == 0;
    wire [31:0] special_result = nan_out ? 32'h7fc00000 : inf_out ? {neg_inf_s4, 8'hff, 23'd0} :
                                 all_zero_s4 ? {all_negative_s4, 31'd0} : 32'd0;
    wire [4:0] special_flags = nan_out ? {invalid_s4, 4'b0} :
                               inf_out || all_zero_s4 ? 5'b0 : {4'b0, truncated_s4};
    fp32_round #(.MW(SW-1)) round_sum (
        .clk(clk), .sign(sum_s4[SW-1]), .mag(mag), .lsb_exp($signed({3'b0, top_s4}) - LSB_OFFSET),
        .sticky_in(truncated_s4), .special(special), .special_result(special_result),
        .special_flags(special_flags), .result(result), .flags(exception_flags)
    );
endmodule
