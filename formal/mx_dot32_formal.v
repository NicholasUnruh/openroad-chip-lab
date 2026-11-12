// Formal checks of rtl/mx_dot32_fp32.v against a direct specification (formal/run.sh). The
// specification decodes each code from its format definition, forms the exact sum as one wide
// integer and rounds it to nearest even; it shares no logic with the unit. Inputs are held for the
// unit's seven-clock latency, then its result and flags are compared.
//
// MODE 0, products: lanes I and J free, every other lane a signed zero, c = +-0, block scales
//   anywhere in 2^-20 .. 2^20 (MX) or any non-NaN E4M3 pair (NVFP4). Checks element decode
//   (normals, subnormals, signed zeros), the exact products, the block scale, normalization,
//   rounding, the inexact flag and the sign of exact zeros.
// MODE 1, alignment: lanes I and J free, unit block scales, c any normal number with exponent
//   field CEXP. formal/run.sh sweeps CEXP across every offset where c and the block sum interact
//   (and beyond, where one is only sticky), covering alignment, cancellation and sticky bits.
// MODE 2, specials: all 32 lanes, both scales and c free. Whenever the specification says the
//   result is a NaN or an infinity, the unit must return it with the right flags.
// Solvers cannot handle a free alignment offset and a free c together (no result after 15 min),
// hence the per-offset sweep in MODE 1.
module mx_dot32_formal #(
    parameter ELEM = 0,
    parameter MODE = 0,
    parameter CEXP = 127,
    parameter I = 0,
    parameter J = 31
) (
    input wire clk,
    input wire [7:0] a_i, b_i, a_j, b_j,  // MODES 0-1: the two free lanes' codes (low W bits)
    input wire [31:0] zero_sign_a, zero_sign_b,  // MODES 0-1: sign bits of the zero lanes
    input wire [255:0] a_all, b_all,  // MODE 2: every lane
    input wire [15:0] scale_a, scale_b,  // MX formats use the low byte
    input wire [31:0] c
);
    localparam NV = ELEM == 5;
    localparam EXP = ELEM == 0 ? 4 : ELEM == 2 ? 5 : ELEM == 3 ? 3 : 2;
    localparam MAN = ELEM == 0 || ELEM == 4 ? 3 : ELEM == 1 || NV ? 1 : 2;
    localparam W = 1 + EXP + MAN;
    localparam BIAS = (1 << (EXP - 1)) - 1;
    localparam IEEE = ELEM == 2;
    localparam LAT = 7;
    localparam SW = NV ? 16 : 8;

    // The unit's inputs.
    reg [32*W-1:0] a, b;
    integer k;
    always @* begin
        if (MODE == 2) begin
            a = a_all[32*W-1:0];
            b = b_all[32*W-1:0];
        end else begin
            for (k = 0; k < 32; k = k + 1) begin
                a[W*k +: W] = {zero_sign_a[k], {(W-1){1'b0}}};
                b[W*k +: W] = {zero_sign_b[k], {(W-1){1'b0}}};
            end
            a[W*I +: W] = a_i[W-1:0];
            b[W*I +: W] = b_i[W-1:0];
            a[W*J +: W] = a_j[W-1:0];
            b[W*J +: W] = b_j[W-1:0];
        end
    end
    wire [31:0] result;
    wire [4:0] flags;
    mx_dot32_fp32 #(.ELEM(ELEM)) dut (
        .clk(clk), .a(a), .b(b), .scale_a(scale_a[SW-1:0]), .scale_b(scale_b[SW-1:0]), .c(c),
        .result(result), .exception_flags(flags)
    );

    // Finite values: sig * 2^lsb. Element sig = hidden.mantissa, lsb = max(e, 1) - BIAS - MAN.
    function automatic [3:0] elem_sig;
        input [7:0] code;
        elem_sig = (code[MAN +: EXP] != 0) << MAN | code[MAN-1:0];
    endfunction
    function automatic integer elem_lsb;
        input [7:0] code;
        elem_lsb = (code[MAN +: EXP] == 0 ? 1 : code[MAN +: EXP]) - BIAS - MAN;
    endfunction
    // E4M3 block scale (NVFP4): sign ignored.
    function automatic [3:0] e4m3_sig;
        input [7:0] code;
        e4m3_sig = (code[6:3] != 0) << 3 | code[2:0];
    endfunction
    function automatic integer e4m3_lsb;
        input [7:0] code;
        e4m3_lsb = (code[6:3] == 0 ? 1 : code[6:3]) - 10;
    endfunction

    // MODES 0-1: exact value of c + a_I b_I + a_J b_J in a frame whose LSB is 2^FL, rounded.
    localparam FL = -96, FW = 208;
    reg signed [FW:0] exact, term;
    reg [FW:0] mag;
    reg [15:0] block_sig_i, block_sig_j;
    reg [23:0] prod_i, prod_j;
    reg [7:0] spec_exponent;
    integer block_lsb_i, block_lsb_j, p, x;
    reg [31:0] spec_result;
    reg spec_inexact, all_negative, any_nonzero, guard, sticky, round_up;
    reg [24:0] sig;
    always @* begin
        if (NV) begin
            block_sig_i = e4m3_sig(scale_a[8*(I/16) +: 8]) * e4m3_sig(scale_b[8*(I/16) +: 8]);
            block_lsb_i = e4m3_lsb(scale_a[8*(I/16) +: 8]) + e4m3_lsb(scale_b[8*(I/16) +: 8]);
            block_sig_j = e4m3_sig(scale_a[8*(J/16) +: 8]) * e4m3_sig(scale_b[8*(J/16) +: 8]);
            block_lsb_j = e4m3_lsb(scale_a[8*(J/16) +: 8]) + e4m3_lsb(scale_b[8*(J/16) +: 8]);
        end else begin
            block_sig_i = 1;
            block_sig_j = 1;
            block_lsb_i = scale_a[7:0] + scale_b[7:0] - 254;
            block_lsb_j = block_lsb_i;
        end
        // c is a normal number or zero here (the window excludes subnormal c).
        exact = c[30:23] == 0 ? 0 : $signed({1'b0, 1'b1, c[22:0]}) <<< (c[30:23] - 150 - FL);
        if (c[31]) exact = -exact;
        prod_i = elem_sig(a_i) * elem_sig(b_i) * block_sig_i;
        prod_j = elem_sig(a_j) * elem_sig(b_j) * block_sig_j;
        term = $signed({1'b0, prod_i}) <<< (elem_lsb(a_i) + elem_lsb(b_i) + block_lsb_i - FL);
        exact = a_i[W-1] ^ b_i[W-1] ? exact - term : exact + term;
        term = $signed({1'b0, prod_j}) <<< (elem_lsb(a_j) + elem_lsb(b_j) + block_lsb_j - FL);
        exact = a_j[W-1] ^ b_j[W-1] ? exact - term : exact + term;
        // An exact zero is -0 only when c and every product are negative zeros.
        all_negative = c[31];
        any_nonzero = c[30:0] != 0 || prod_i != 0 || prod_j != 0;
        for (x = 0; x < 32; x = x + 1)
            all_negative = all_negative & (a[W*x+W-1] ^ b[W*x+W-1]);
        // Round to nearest even: 24 significant bits (fewer after cancellation), guard, sticky.
        mag = exact < 0 ? -exact : exact;
        p = 0;
        for (x = 0; x < FW; x = x + 1)
            if (mag[x]) p = x;
        sig = p >= 23 ? mag >> (p - 23) : mag << (23 - p);
        guard = p >= 24 ? mag[p - 24] : 1'b0;
        sticky = 0;
        for (x = 0; x < FW; x = x + 1)
            if (x < p - 24 && mag[x]) sticky = 1;
        round_up = guard && (sticky || sig[0]);
        sig = sig + round_up;
        if (sig[24]) begin
            sig = sig >> 1;
            p = p + 1;
        end
        spec_inexact = guard || sticky;
        spec_exponent = p + FL + 127;
        if (mag == 0)
            spec_result = {!any_nonzero && all_negative, 31'b0};
        else
            spec_result = {exact < 0, spec_exponent, sig[22:0]};
    end

    // MODE 2: specials over all lanes (OCP MX: E8M0 NaN 0xFF, E4M3 NaN S.1111.111, E5M2 IEEE
    // infinities and NaNs, signaling when the top mantissa bit is clear; NVFP4: E4M3 scale NaN).
    reg nan, invalid, pos_inf, neg_inf, x_nan, y_nan, x_inf, y_inf, x_zero, y_zero;
    reg [7:0] xc, yc;
    integer s;
    always @* begin
        nan = NV ? scale_a[6:0] == 7'h7f || scale_a[14:8] == 7'h7f ||
                   scale_b[6:0] == 7'h7f || scale_b[14:8] == 7'h7f
                 : scale_a[7:0] == 8'hff || scale_b[7:0] == 8'hff;
        invalid = 0;
        pos_inf = 0;
        neg_inf = 0;
        for (s = 0; s < 32; s = s + 1) begin
            xc = a[W*s +: W];
            yc = b[W*s +: W];
            x_nan = IEEE ? &xc[MAN +: EXP] && xc[MAN-1:0] != 0 : ELEM == 0 && &xc[6:0];
            y_nan = IEEE ? &yc[MAN +: EXP] && yc[MAN-1:0] != 0 : ELEM == 0 && &yc[6:0];
            x_inf = IEEE && &xc[MAN +: EXP] && xc[MAN-1:0] == 0;
            y_inf = IEEE && &yc[MAN +: EXP] && yc[MAN-1:0] == 0;
            x_zero = xc[W-2:0] == 0;
            y_zero = yc[W-2:0] == 0;
            nan = nan || x_nan || y_nan;
            invalid = invalid || (x_nan && !xc[MAN-1]) || (y_nan && !yc[MAN-1]) ||
                      (x_inf && y_zero) || (y_inf && x_zero);
            if ((x_inf || y_inf) && !x_nan && !y_nan && !x_zero && !y_zero) begin
                if (xc[W-1] ^ yc[W-1]) neg_inf = 1;
                else pos_inf = 1;
            end
        end
        if (c[30:23] == 8'hff && c[22:0] == 0) begin
            if (c[31]) neg_inf = 1;
            else pos_inf = 1;
        end
        invalid = invalid || (c[30:23] == 8'hff && c[22:0] != 0 && !c[22]) || (pos_inf && neg_inf);
        nan = nan || (c[30:23] == 8'hff && c[22:0] != 0);
    end

    function automatic special_code;
        input [7:0] code;
        special_code = IEEE ? &code[MAN +: EXP] : ELEM == 0 && &code[6:0];
    endfunction
    reg in_window;
    always @*
        if (MODE == 2)
            in_window = 1;
        else
            in_window = !special_code(a_i) && !special_code(b_i) && !special_code(a_j) &&
                        !special_code(b_j) &&
                        (MODE == 0 ? c[30:0] == 0 : c[30:23] == CEXP) &&
                        (MODE == 1 ? (NV ? scale_a == 16'h3838 && scale_b == 16'h3838
                                         : scale_a[7:0] == 127 && scale_b[7:0] == 127)
                         : NV ? scale_a[6:0] != 7'h7f && scale_a[14:8] != 7'h7f &&
                                scale_b[6:0] != 7'h7f && scale_b[14:8] != 7'h7f
                              : scale_a[7:0] != 8'hff && scale_b[7:0] != 8'hff &&
                                scale_a[7:0] + scale_b[7:0] >= 254 - 20 &&
                                scale_a[7:0] + scale_b[7:0] <= 254 + 20);

    // Hold every input for the latency, then compare.
    localparam IW = 4 * 8 + 2 * 32 + 2 * 256 + 2 * 16 + 32;
    wire [IW-1:0] inputs = {a_i, b_i, a_j, b_j, zero_sign_a, zero_sign_b, a_all, b_all, scale_a,
                            scale_b, c};
    reg [IW-1:0] held;
    reg [3:0] cycle = 0;
    always @(posedge clk) begin
        if (cycle != 4'd15) cycle <= cycle + 1;
        held <= inputs;
    end
    always @* begin
        assume (in_window);
        if (cycle != 0) assume (inputs == held);
        if (cycle >= LAT) begin
            if (MODE == 2) begin
                if (nan || invalid) assert (result == 32'h7fc00000 && flags == {invalid, 4'b0});
                else if (pos_inf || neg_inf)
                    assert (result == {neg_inf, 8'hff, 23'b0} && flags == 5'b0);
            end else begin
                assert (result == spec_result);
                assert (flags == {4'b0, spec_inexact});
            end
        end
    end
endmodule
