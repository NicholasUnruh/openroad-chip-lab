// OCP MX block dot product: correctly rounded c + 2^(sa + sb - 254) * sum of 32 element products
// (docs/numerics-study.md); seven-clock latency. ELEM selects the element type:
// 0 = FP8 E4M3, 1 = FP4 E2M1, 2 = FP8 E5M2, 3 = FP6 E3M2, 4 = FP6 E2M3, each with one E8M0 scale
// per operand; 5 = NVFP4: E2M1 elements in two 16-element blocks, each block with its own E4M3
// scale per operand (scale_a[7:0] for elements 0-15, scale_a[15:8] for 16-31).
// Element exponents are small, so the 32 products add exactly in fixed point; only the final
// addition of the block sum to c aligns, keeping dropped bits as a sticky LSB.
module mx_dot32_fp32 #(
    parameter ELEM = 0
) (
    input wire clk,
    input wire [32*(ELEM == 0 || ELEM == 2 ? 8 : ELEM == 1 || ELEM == 5 ? 4 : 6)-1:0] a, b,
    input wire [(ELEM == 5 ? 16 : 8)-1:0] scale_a, scale_b,
    input wire [31:0] c,
    output wire [31:0] result,
    output wire [4:0] exception_flags
);
    localparam NV = ELEM == 5;
    localparam EXP = ELEM == 0 ? 4 : ELEM == 2 ? 5 : ELEM == 3 ? 3 : 2;
    localparam MAN = ELEM == 0 || ELEM == 4 ? 3 : ELEM == 1 || NV ? 1 : 2;
    // E5M2 has IEEE infinities and NaNs (signaling when the top mantissa bit is clear); E4M3 has
    // one NaN mantissa (S.1111.111) and no infinities; the FP6 and FP4 types have no specials.
    localparam IEEE = ELEM == 2;
    localparam BIAS = (1 << (EXP - 1)) - 1;
    localparam W = 1 + EXP + MAN;
    localparam PW = 2 * (MAN + 1);
    localparam SHW = EXP + 1;                        // product shift width: 0 .. 2*(2^EXP-1)-2
    localparam FIXW = PW + 2 * ((1 << EXP) - 1) - 2; // exact fixed-point product width
    localparam SUMW = FIXW + 6;                      // 32 terms plus sign
    localparam FIX_LSB = 2 - 2 * BIAS - 2 * MAN;     // exponent of the fixed-point LSB
    // NVFP4 multiplies each 16-element block sum by its 8-bit scale significand product and
    // aligns the two by their scale exponent difference (at most 28).
    localparam BW = NV ? SUMW + 9 + 28 : SUMW;      // block sum width
    localparam XW = BW - 1;                          // block magnitude width
    localparam LW = $clog2(XW);
    // Frame: holds the normalized block and keeps three bits below FP32 precision, so dropped
    // bits only ever reach the sticky LSB.
    localparam WF = XW > 27 ? XW : 27;

    // Stage 1: decode, exact element products, specials.
    reg [32*PW-1:0] p_s1;
    reg [32*SHW-1:0] shift_s1;
    reg [31:0] sign_s1;
    reg [31:0] c_s1;
    reg signed [11:0] scale_s1;
    wire [15:0] nv_sa, nv_sb;  // NVFP4 block scales; zero-extended for the MX types
    generate if (NV) begin: nv_scales
        assign nv_sa = scale_a;
        assign nv_sb = scale_b;
    end else begin: mx_scales
        assign nv_sa = {8'd0, scale_a};
        assign nv_sb = {8'd0, scale_b};
    end endgenerate
    reg [15:0] scale_sig_s1;
    reg [11:0] scale_lsb_s1;  // two signed 6-bit exponents, -18 .. 10
    reg nan_s1, invalid_s1, pos_inf_s1, neg_inf_s1, all_zero_s1, all_negative_s1;
    reg [EXP-1:0] ea, eb;
    reg [MAN-1:0] ma, mb;
    reg nan, invalid, pos_inf, neg_inf, all_zero, all_negative;
    reg a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    reg [31:0] term_inf;
    integer k, j;
    always @* begin
        // The E8M0 NaN is 0xFF; the E4M3 NaN is S.1111.111, and E4M3 scale signs are ignored.
        nan = NV ? &nv_sa[6:0] || &nv_sa[14:8] || &nv_sb[6:0] || &nv_sb[14:8]
                 : &scale_a || &scale_b;
        invalid = 1'b0;
        pos_inf = 1'b0;
        neg_inf = 1'b0;
        all_zero = 1'b1;
        all_negative = 1'b1;
        for (k = 0; k < 32; k = k + 1) begin
            ea = a[W*k+MAN +: EXP];
            eb = b[W*k+MAN +: EXP];
            ma = a[W*k +: MAN];
            mb = b[W*k +: MAN];
            a_zero = ea == 0 && ma == 0;
            b_zero = eb == 0 && mb == 0;
            a_inf = IEEE && &ea && ma == 0;
            b_inf = IEEE && &eb && mb == 0;
            a_nan = IEEE ? &ea && ma != 0 : ELEM == 0 && &ea && &ma;
            b_nan = IEEE ? &eb && mb != 0 : ELEM == 0 && &eb && &mb;
            term_inf[k] = a_inf || b_inf;
            nan = nan | a_nan | b_nan;
            // Signaling NaNs and infinity times zero are invalid; an infinite product of finite
            // nonzero values gives a signed infinity.
            invalid = invalid | (IEEE && a_nan && !ma[MAN-1]) | (IEEE && b_nan && !mb[MAN-1]) |
                      (a_inf && b_zero) | (b_inf && a_zero);
            if (term_inf[k] && !a_nan && !b_nan && !a_zero && !b_zero) begin
                if (a[W*k+W-1] ^ b[W*k+W-1]) neg_inf = 1'b1;
                else pos_inf = 1'b1;
            end
            // An NVFP4 lane is also a zero product when either of its block scales is zero.
            all_zero = all_zero & (a_zero | b_zero |
                       (NV && (k < 16 ? nv_sa[6:0] == 0 || nv_sb[6:0] == 0
                                      : nv_sa[14:8] == 0 || nv_sb[14:8] == 0)));
            all_negative = all_negative & (a[W*k+W-1] ^ b[W*k+W-1]);
        end
    end
    always @(posedge clk) begin
        for (j = 0; j < 32; j = j + 1) begin
            // Infinite terms decide the result as specials; their products stay out of the sum.
            p_s1[PW*j +: PW] <= term_inf[j] ? {PW{1'b0}} :
                                {a[W*j+MAN +: EXP] != 0, a[W*j +: MAN]} * {b[W*j+MAN +: EXP] != 0, b[W*j +: MAN]};
            shift_s1[SHW*j +: SHW] <= (a[W*j+MAN +: EXP] == 0 ? {{EXP{1'b0}}, 1'b1} : {1'b0, a[W*j+MAN +: EXP]}) +
                                      (b[W*j+MAN +: EXP] == 0 ? {{EXP{1'b0}}, 1'b1} : {1'b0, b[W*j+MAN +: EXP]}) -
                                      {{(SHW-2){1'b0}}, 2'd2};
            sign_s1[j] <= a[W*j+W-1] ^ b[W*j+W-1];
        end
        scale_s1 <= NV ? FIX_LSB : $signed({4'b0, scale_a[7:0]}) + $signed({4'b0, scale_b[7:0]}) -
                                   12'sd254 + FIX_LSB;
        // NVFP4 block scales: significand product and LSB exponent of sa * sb per block.
        for (j = 0; j < 2; j = j + 1) begin
            scale_sig_s1[8*j +: 8] <= {nv_sa[8*j+3 +: 4] != 0, nv_sa[8*j +: 3]} *
                                      {nv_sb[8*j+3 +: 4] != 0, nv_sb[8*j +: 3]};
            scale_lsb_s1[6*j +: 6] <= (nv_sa[8*j+3 +: 4] == 0 ? 6'd1 : {2'b0, nv_sa[8*j+3 +: 4]}) +
                                      (nv_sb[8*j+3 +: 4] == 0 ? 6'd1 : {2'b0, nv_sb[8*j+3 +: 4]}) -
                                      6'd20;
        end
        c_s1 <= c;
        nan_s1 <= nan;
        invalid_s1 <= invalid;
        pos_inf_s1 <= pos_inf;
        neg_inf_s1 <= neg_inf;
        all_zero_s1 <= all_zero;
        all_negative_s1 <= all_negative;
    end

    // Stages 2-3: exact block sum in fixed point as a balanced adder tree, split after the
    // third level into four registered partial sums.
    reg [32*SUMW-1:0] level;
    integer m, n;
    always @* begin
        for (m = 0; m < 32; m = m + 1)
            level[SUMW*m +: SUMW] = sign_s1[m] ? -({{(SUMW-PW){1'b0}}, p_s1[PW*m +: PW]} << shift_s1[SHW*m +: SHW])
                                               : {{(SUMW-PW){1'b0}}, p_s1[PW*m +: PW]} << shift_s1[SHW*m +: SHW];
        for (n = 16; n > 2; n = n >> 1)
            for (m = 0; m < n; m = m + 1)
                level[SUMW*m +: SUMW] = level[SUMW*2*m +: SUMW] + level[SUMW*(2*m+1) +: SUMW];
    end
    reg [4*SUMW-1:0] partial_s2;
    reg [15:0] scale_sig_s2;
    reg [11:0] scale_lsb_s2;
    reg signed [11:0] block_lsb_s2;
    reg [31:0] c_s2;
    reg nan_s2, invalid_s2, pos_inf_s2, neg_inf_s2, all_zero_s2, all_negative_s2;
    always @(posedge clk) begin
        // NVFP4 keeps the two 16-element block sums; MX keeps four 8-element partial sums.
        partial_s2 <= NV ? {{(2*SUMW){1'b0}}, level[2*SUMW +: SUMW] + level[3*SUMW +: SUMW],
                            level[0 +: SUMW] + level[SUMW +: SUMW]} : level[4*SUMW-1:0];
        scale_sig_s2 <= scale_sig_s1;
        scale_lsb_s2 <= scale_lsb_s1;
        block_lsb_s2 <= scale_s1;
        c_s2 <= c_s1;
        nan_s2 <= nan_s1;
        invalid_s2 <= invalid_s1;
        pos_inf_s2 <= pos_inf_s1;
        neg_inf_s2 <= neg_inf_s1;
        all_zero_s2 <= all_zero_s1;
        all_negative_s2 <= all_negative_s1;
    end
    wire signed [BW-1:0] block_sum;
    wire signed [11:0] block_lsb;
    generate if (NV) begin: nvfp4
        wire signed [SUMW+8:0] scaled0 = $signed(partial_s2[0 +: SUMW]) * $signed({1'b0, scale_sig_s2[7:0]});
        wire signed [SUMW+8:0] scaled1 = $signed(partial_s2[SUMW +: SUMW]) * $signed({1'b0, scale_sig_s2[15:8]});
        wire signed [5:0] lsb0 = scale_lsb_s2[5:0], lsb1 = scale_lsb_s2[11:6];
        wire signed [6:0] gap = lsb0 - lsb1;
        wire [4:0] shift0 = gap > 0 ? gap[4:0] : 5'd0;
        wire [4:0] shift1 = gap < 0 ? -gap[4:0] : 5'd0;
        // The block with the larger scale exponent shifts up onto the other's LSB grid.
        assign block_sum = ({{(BW-SUMW-9){scaled0[SUMW+8]}}, scaled0} <<< shift0) +
                           ({{(BW-SUMW-9){scaled1[SUMW+8]}}, scaled1} <<< shift1);
        assign block_lsb = block_lsb_s2 + (gap > 0 ? {{6{lsb1[5]}}, lsb1} : {{6{lsb0[5]}}, lsb0});
    end else begin: mx
        assign block_sum = (partial_s2[0 +: SUMW] + partial_s2[SUMW +: SUMW]) +
                           (partial_s2[2*SUMW +: SUMW] + partial_s2[3*SUMW +: SUMW]);
        assign block_lsb = block_lsb_s2;
    end endgenerate
    reg signed [BW-1:0] block_s3;
    reg signed [11:0] block_lsb_s3;
    reg [31:0] c_s3;
    reg nan_s3, invalid_s3, pos_inf_s3, neg_inf_s3, all_zero_s3, all_negative_s3;
    always @(posedge clk) begin
        block_s3 <= block_sum;
        block_lsb_s3 <= block_lsb;
        c_s3 <= c_s2;
        nan_s3 <= nan_s2;
        invalid_s3 <= invalid_s2;
        pos_inf_s3 <= pos_inf_s2;
        neg_inf_s3 <= neg_inf_s2;
        all_zero_s3 <= all_zero_s2;
        all_negative_s3 <= all_negative_s2;
    end

    // Stage 4: normalize the block; find the larger top and each operand's distance below it.
    wire [XW-1:0] x_mag = block_s3[BW-1] ? -block_s3[XW-1:0] : block_s3[XW-1:0];
    wire [7:0] ec = c_s3[30:23];
    wire [23:0] c_sig = {ec != 8'd0, c_s3[22:0]};
    wire c_nan = &ec && c_s3[22:0] != 23'd0;
    wire c_inf = &ec && c_s3[22:0] == 23'd0;
    wire any_pos_inf = pos_inf_s3 || (c_inf && !c_s3[31]);
    wire any_neg_inf = neg_inf_s3 || (c_inf && c_s3[31]);
    wire any_invalid = invalid_s3 || (c_nan && !c_s3[22]) || (any_pos_inf && any_neg_inf);
    reg [LW-1:0] x_lead;
    reg [XW-1:0] x_norm;
    reg signed [11:0] x_top, c_top, top, x_dist, c_dist;
    integer i;
    always @* begin
        x_lead = {LW{1'b0}};
        for (i = 0; i < XW; i = i + 1)
            if (x_mag[i]) x_lead = i[LW-1:0];
        x_norm = x_mag << (XW - 1 - x_lead);
        x_top = block_lsb_s3 + $signed({{(12-LW){1'b0}}, x_lead}) + 12'sd1;
        c_top = (ec == 8'd0 ? -12'sd149 : $signed({4'b0, ec}) - 12'sd150) + 12'sd24;
        top = x_mag == 0 ? c_top : c_sig == 24'd0 || x_top > c_top ? x_top : c_top;
        x_dist = top - x_top;
        c_dist = top - c_top;
    end
    reg [XW-1:0] x_norm_s4;
    reg [23:0] c_sig_s4;
    reg [6:0] x_dist_s4, c_dist_s4;  // saturated at WF
    reg signed [11:0] lsb_s4;
    reg x_sign_s4, c_sign_s4, nan_s4, invalid_s4, inf_s4, inf_sign_s4, all_zero_s4, all_negative_s4;
    always @(posedge clk) begin
        x_norm_s4 <= x_norm;
        c_sig_s4 <= c_sig;
        x_dist_s4 <= x_dist >= WF ? WF[6:0] : x_dist[6:0];
        c_dist_s4 <= c_dist >= WF ? WF[6:0] : c_dist[6:0];
        lsb_s4 <= top - WF;
        x_sign_s4 <= block_s3[BW-1];
        c_sign_s4 <= c_s3[31];
        nan_s4 <= nan_s3 || c_nan || any_invalid;
        invalid_s4 <= any_invalid;
        inf_s4 <= any_pos_inf || any_neg_inf;
        inf_sign_s4 <= any_neg_inf;
        all_zero_s4 <= all_zero_s3 && c_sig == 24'd0;
        all_negative_s4 <= all_negative_s3 && c_s3[31];
    end

    // Stage 5: align both operands below the top, keeping dropped bits as a sticky LSB; add once.
    wire [WF-1:0] x_just = {x_norm_s4, {(WF-XW){1'b0}}};
    wire [WF-1:0] c_just = {c_sig_s4, {(WF-24){1'b0}}};
    reg [WF-1:0] x_al, c_al, low_mask;
    reg signed [WF+1:0] frame;
    integer x;
    always @* begin
        for (x = 0; x < WF; x = x + 1) low_mask[x] = x < {25'd0, x_dist_s4};
        x_al = x_dist_s4 >= WF ? {{(WF-1){1'b0}}, x_just != 0}
                               : (x_just >> x_dist_s4) | {{(WF-1){1'b0}}, (x_just & low_mask) != 0};
        for (x = 0; x < WF; x = x + 1) low_mask[x] = x < {25'd0, c_dist_s4};
        c_al = c_dist_s4 >= WF ? {{(WF-1){1'b0}}, c_just != 0}
                               : (c_just >> c_dist_s4) | {{(WF-1){1'b0}}, (c_just & low_mask) != 0};
        frame = (x_sign_s4 ? -$signed({2'b0, x_al}) : $signed({2'b0, x_al})) +
                (c_sign_s4 ? -$signed({2'b0, c_al}) : $signed({2'b0, c_al}));
    end
    reg signed [WF+1:0] frame_s5;
    reg signed [11:0] lsb_s5;
    reg nan_s5, invalid_s5, inf_s5, inf_sign_s5, all_zero_s5, all_negative_s5;
    always @(posedge clk) begin
        frame_s5 <= frame;
        lsb_s5 <= lsb_s4;
        nan_s5 <= nan_s4;
        invalid_s5 <= invalid_s4;
        inf_s5 <= inf_s4;
        inf_sign_s5 <= inf_sign_s4;
        all_zero_s5 <= all_zero_s4;
        all_negative_s5 <= all_negative_s4;
    end

    // Stages 6-7: round once in fp32_round, or pass NaN, infinity or a zero sum.
    wire [WF:0] mag = frame_s5[WF+1] ? -frame_s5[WF:0] : frame_s5[WF:0];
    wire special = nan_s5 || inf_s5 || all_zero_s5 || frame_s5 == 0;
    wire [31:0] special_result = nan_s5 ? 32'h7fc00000 : inf_s5 ? {inf_sign_s5, 8'hff, 23'd0} :
                                 all_zero_s5 ? {all_negative_s5, 31'd0} : 32'd0;
    fp32_round #(.MW(WF+1)) round_sum (
        .clk(clk), .sign(frame_s5[WF+1]), .mag(mag), .lsb_exp(lsb_s5), .sticky_in(1'b0),
        .special(special), .special_result(special_result), .special_flags({nan_s5 && invalid_s5, 4'b0}),
        .result(result), .flags(exception_flags)
    );
endmodule
