// Final two pipeline stages of the dot-product units: round sign * mag * 2^lsb_exp once to FP32,
// nearest-even, with subnormals on the 2^-149 grid, or pass a precomputed special result.
// Stage A finds the leading/trailing ones and the rounding position; stage B shifts, rounds,
// packs and registers. sticky_in marks nonzero bits already dropped below mag. Underflow uses
// tininess before rounding. Latency: two clocks from the inputs to result.
module fp32_round #(
    parameter MW = 36  // magnitude width, at least 24
) (
    input wire clk,
    input wire sign,
    input wire [MW-1:0] mag,          // nonzero unless special
    input wire signed [11:0] lsb_exp,
    input wire sticky_in,
    input wire special,
    input wire [31:0] special_result,
    input wire [4:0] special_flags,
    output reg [31:0] result,
    output reg [4:0] flags
);
    localparam LW = $clog2(MW + 1);
    localparam signed [11:0] MW12 = MW[11:0];

    // Stage A: rounding position. shift is the LSB position kept: 24 significant bits, but
    // never below the 2^-149 subnormal grid.
    reg [LW-1:0] lead, trailing;
    reg signed [11:0] shift;
    integer k;
    always @* begin
        lead = {LW{1'b0}};
        trailing = MW[LW-1:0];
        for (k = 0; k < MW; k = k + 1)
            if (mag[k]) lead = k[LW-1:0];
        for (k = MW - 1; k >= 0; k = k - 1)
            if (mag[k]) trailing = k[LW-1:0];
        shift = $signed({{(12-LW){1'b0}}, lead}) - 12'sd23;
        if (-12'sd149 - lsb_exp > shift) shift = -12'sd149 - lsb_exp;
    end
    reg [MW-1:0] mag_a;
    reg [11:0] amount_a;
    reg signed [11:0] exp_a;
    reg sign_a, round_a, sticky_a, dropped_a, tiny_a, special_a;
    reg [31:0] special_result_a;
    reg [4:0] special_flags_a;
    always @(posedge clk) begin
        mag_a <= mag;
        amount_a <= shift + MW12;
        exp_a <= shift + lsb_exp + 12'sd150;
        sign_a <= sign;
        // The round bit is mag[shift-1] and sticky is any lower set bit. Bits dropped before
        // the frame only make the result inexact; they never change the rounding direction.
        round_a <= shift > 0 && shift <= MW12 && mag[shift[LW-1:0] - 1'b1];
        sticky_a <= shift > 0 && $signed({{(12-LW){1'b0}}, trailing}) < shift - 12'sd1;
        dropped_a <= sticky_in;
        tiny_a <= $signed({{(12-LW){1'b0}}, lead}) + lsb_exp < -12'sd126;
        special_a <= special;
        special_result_a <= special_result;
        special_flags_a <= special_flags;
    end

    // Stage B: one right shift of mag * 2^MW covers both directions; round, carry, pack.
    reg [2*MW-1:0] shifted;
    reg [24:0] sig;
    reg signed [11:0] exp_field;
    reg inexact, overflow;
    always @* begin
        shifted = {mag_a, {MW{1'b0}}} >> amount_a;
        inexact = round_a | sticky_a | dropped_a;
        sig = shifted[24:0] + {24'd0, round_a && (sticky_a || shifted[0])};
        exp_field = exp_a;
        if (sig[24]) begin
            sig = sig >> 1;
            exp_field = exp_field + 12'sd1;
        end
        if (!sig[23]) exp_field = 12'sd0;
        overflow = exp_field >= 12'sd255;
    end
    always @(posedge clk) begin
        result <= special_a ? special_result_a :
                  overflow ? {sign_a, 8'hff, 23'd0} : {sign_a, exp_field[7:0], sig[22:0]};
        flags <= special_a ? special_flags_a : {2'b0, overflow, tiny_a && inexact, inexact | overflow};
    end
endmodule
