// OCP MX block dot product: correctly rounded c + 2^(sa + sb - 254) * sum of 32 element products
// (docs/numerics-study.md); four-clock latency. ELEM 0 = FP8 E4M3, 1 = FP4 E2M1.
// Element exponents are small, so the 32 products add exactly in fixed point; only the final
// addition of the block sum to c aligns, keeping dropped bits as a sticky LSB.
module mx_dot32_fp32 #(
    parameter ELEM = 0
) (
    input wire clk,
    input wire [32*(ELEM == 0 ? 8 : 4)-1:0] a, b,
    input wire [7:0] scale_a, scale_b,
    input wire [31:0] c,
    output reg [31:0] result,
    output reg [4:0] exception_flags
);
    localparam EXP = ELEM == 0 ? 4 : 2;
    localparam MAN = ELEM == 0 ? 3 : 1;
    localparam BIAS = (1 << (EXP - 1)) - 1;
    localparam W = 1 + EXP + MAN;
    localparam PW = 2 * (MAN + 1);
    localparam SHW = EXP + 1;                        // product shift width: 0 .. 2*(2^EXP-1)-2
    localparam FIXW = PW + 2 * ((1 << EXP) - 1) - 2; // exact fixed-point product width
    localparam SUMW = FIXW + 6;                      // 32 terms plus sign
    localparam FIX_LSB = 2 - 2 * BIAS - 2 * MAN;     // exponent of the fixed-point LSB
    localparam XW = SUMW - 1;                        // block magnitude width
    // Frame: holds the normalized block and keeps three bits below FP32 precision, so dropped
    // bits only ever reach the sticky LSB.
    localparam WF = XW > 27 ? XW : 27;

    // Stage 1: decode, exact element products, specials.
    reg [32*PW-1:0] p_s1;
    reg [32*SHW-1:0] shift_s1;
    reg [31:0] sign_s1;
    reg [31:0] c_s1;
    reg signed [11:0] scale_s1;
    reg nan_s1, all_zero_s1, all_negative_s1;
    reg [EXP-1:0] ea, eb;
    reg [MAN-1:0] ma, mb;
    reg nan, all_zero, all_negative;
    integer k;
    always @* begin
        nan = &scale_a || &scale_b;
        all_zero = 1'b1;
        all_negative = 1'b1;
        for (k = 0; k < 32; k = k + 1) begin
            ea = a[W*k+MAN +: EXP];
            eb = b[W*k+MAN +: EXP];
            ma = a[W*k +: MAN];
            mb = b[W*k +: MAN];
            // Only E4M3 has a NaN encoding (S.1111.111); E2M1 has no special values.
            if (ELEM == 0) nan = nan | (&ea && &ma) | (&eb && &mb);
            all_zero = all_zero & ((ea == 0 && ma == 0) | (eb == 0 && mb == 0));
            all_negative = all_negative & (a[W*k+W-1] ^ b[W*k+W-1]);
        end
    end
    always @(posedge clk) begin
        for (k = 0; k < 32; k = k + 1) begin
            p_s1[PW*k +: PW] <= {a[W*k+MAN +: EXP] != 0, a[W*k +: MAN]} * {b[W*k+MAN +: EXP] != 0, b[W*k +: MAN]};
            shift_s1[SHW*k +: SHW] <= (a[W*k+MAN +: EXP] == 0 ? {{EXP{1'b0}}, 1'b1} : {1'b0, a[W*k+MAN +: EXP]}) +
                                      (b[W*k+MAN +: EXP] == 0 ? {{EXP{1'b0}}, 1'b1} : {1'b0, b[W*k+MAN +: EXP]}) -
                                      {{(SHW-2){1'b0}}, 2'd2};
            sign_s1[k] <= a[W*k+W-1] ^ b[W*k+W-1];
        end
        scale_s1 <= $signed({4'b0, scale_a}) + $signed({4'b0, scale_b}) - 12'sd254 + FIX_LSB;
        c_s1 <= c;
        nan_s1 <= nan;
        all_zero_s1 <= all_zero;
        all_negative_s1 <= all_negative;
    end

    // Stage 2: exact block sum in fixed point.
    reg signed [SUMW-1:0] block;
    always @* begin
        block = 0;
        for (k = 0; k < 32; k = k + 1)
            block = sign_s1[k] ? block - $signed({{(SUMW-PW){1'b0}}, p_s1[PW*k +: PW]} << shift_s1[SHW*k +: SHW])
                               : block + $signed({{(SUMW-PW){1'b0}}, p_s1[PW*k +: PW]} << shift_s1[SHW*k +: SHW]);
    end
    reg signed [SUMW-1:0] block_s2;
    reg signed [11:0] block_lsb_s2;
    reg [31:0] c_s2;
    reg nan_s2, all_zero_s2, all_negative_s2;
    always @(posedge clk) begin
        block_s2 <= block;
        block_lsb_s2 <= scale_s1;
        c_s2 <= c_s1;
        nan_s2 <= nan_s1;
        all_zero_s2 <= all_zero_s1;
        all_negative_s2 <= all_negative_s1;
    end

    // Stage 3: normalize the block, align it with c below the larger top, add once.
    wire [XW-1:0] x_mag = block_s2[SUMW-1] ? -block_s2[XW-1:0] : block_s2[XW-1:0];
    wire [7:0] ec = c_s2[30:23];
    wire [23:0] c_sig = {ec != 8'd0, c_s2[22:0]};
    wire c_nan = &ec && c_s2[22:0] != 23'd0;
    wire c_inf = &ec && c_s2[22:0] == 23'd0;
    reg [5:0] x_lead;
    reg [XW-1:0] x_norm;
    reg signed [11:0] x_top, c_top, top, x_dist, c_dist;
    reg [WF-1:0] x_just, c_just, x_al, c_al, low_mask;
    reg signed [WF+1:0] frame;
    integer x;
    always @* begin
        x_lead = 6'd0;
        for (k = 0; k < XW; k = k + 1)
            if (x_mag[k]) x_lead = k[5:0];
        x_norm = x_mag << (XW - 1 - x_lead);
        x_top = block_lsb_s2 + $signed({6'b0, x_lead}) + 12'sd1;
        c_top = (ec == 8'd0 ? -12'sd149 : $signed({4'b0, ec}) - 12'sd150) + 12'sd24;
        top = x_mag == 0 ? c_top : c_sig == 24'd0 || x_top > c_top ? x_top : c_top;
        x_just = {x_norm, {(WF-XW){1'b0}}};
        c_just = {c_sig, {(WF-24){1'b0}}};
        x_dist = top - x_top;
        c_dist = top - c_top;
        for (x = 0; x < WF; x = x + 1) low_mask[x] = x < {20'd0, x_dist};
        x_al = x_dist >= WF ? {{(WF-1){1'b0}}, x_just != 0} : (x_just >> x_dist) | {{(WF-1){1'b0}}, (x_just & low_mask) != 0};
        for (x = 0; x < WF; x = x + 1) low_mask[x] = x < {20'd0, c_dist};
        c_al = c_dist >= WF ? {{(WF-1){1'b0}}, c_just != 0} : (c_just >> c_dist) | {{(WF-1){1'b0}}, (c_just & low_mask) != 0};
        frame = (block_s2[SUMW-1] ? -$signed({2'b0, x_al}) : $signed({2'b0, x_al})) +
                (c_s2[31] ? -$signed({2'b0, c_al}) : $signed({2'b0, c_al}));
    end
    reg signed [WF+1:0] frame_s3;
    reg signed [11:0] lsb_s3;
    reg nan_s3, invalid_s3, inf_s3, inf_sign_s3, all_zero_s3, all_negative_s3;
    always @(posedge clk) begin
        frame_s3 <= frame;
        lsb_s3 <= top - WF;
        nan_s3 <= nan_s2 | c_nan;
        invalid_s3 <= c_nan && !c_s2[22];
        inf_s3 <= c_inf;
        inf_sign_s3 <= c_s2[31];
        all_zero_s3 <= all_zero_s2 && c_sig == 24'd0;
        all_negative_s3 <= all_negative_s2 && c_s2[31];
    end

    // Stage 4: round once.
    wire [WF:0] mag = frame_s3[WF+1] ? -frame_s3[WF:0] : frame_s3[WF:0];
    wire [31:0] rounded;
    wire [4:0] round_flags;
    fp32_round #(.MW(WF+1)) round_sum (
        .sign(frame_s3[WF+1]), .mag(mag), .lsb_exp(lsb_s3), .sticky_in(1'b0),
        .result(rounded), .flags(round_flags)
    );
    always @(posedge clk) begin
        if (nan_s3) begin
            result <= 32'h7fc00000;
            exception_flags <= {invalid_s3, 4'b0};
        end else if (inf_s3) begin
            result <= {inf_sign_s3, 8'hff, 23'd0};
            exception_flags <= 5'b0;
        end else if (all_zero_s3) begin
            result <= {all_negative_s3, 31'd0};
            exception_flags <= 5'b0;
        end else if (frame_s3 == 0) begin
            result <= 32'd0;
            exception_flags <= 5'b0;
        end else begin
            result <= rounded;
            exception_flags <= round_flags;
        end
    end
endmodule
