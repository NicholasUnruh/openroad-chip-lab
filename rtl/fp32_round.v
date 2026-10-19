// Rounds sign * mag * 2^lsb_exp once to FP32, nearest-even, with subnormals on the 2^-149 grid.
// Combinational; mag must be nonzero. sticky_in marks nonzero bits already dropped below mag.
// Underflow uses tininess before rounding.
module fp32_round #(
    parameter MW = 36  // magnitude width, at least 24
) (
    input wire sign,
    input wire [MW-1:0] mag,
    input wire signed [11:0] lsb_exp,
    input wire sticky_in,
    output reg [31:0] result,
    output reg [4:0] flags
);
    localparam LW = $clog2(MW + 1);
    localparam signed [11:0] MW12 = MW[11:0];
    reg [LW-1:0] lead, trailing;
    reg signed [11:0] shift, exp_field;
    reg [2*MW-1:0] shifted;
    reg [11:0] amount;
    reg round_bit, sticky, inexact, overflow;
    reg [24:0] sig;
    integer k;
    always @* begin
        lead = {LW{1'b0}};
        trailing = MW[LW-1:0];
        for (k = 0; k < MW; k = k + 1)
            if (mag[k]) lead = k[LW-1:0];
        for (k = MW - 1; k >= 0; k = k - 1)
            if (mag[k]) trailing = k[LW-1:0];
        // Rounding LSB: 24 significant bits, but never below the 2^-149 subnormal grid.
        shift = $signed({{(12-LW){1'b0}}, lead}) - 12'sd23;
        if (-12'sd149 - lsb_exp > shift) shift = -12'sd149 - lsb_exp;
        // One right shift of mag * 2^MW covers both directions; the round bit is mag[shift-1]
        // and sticky is any lower set bit.
        amount = shift + MW12;
        shifted = {mag, {MW{1'b0}}} >> amount;
        round_bit = shift > 0 && shift <= MW12 && mag[shift[LW-1:0] - 1'b1];
        sticky = shift > 0 && $signed({{(12-LW){1'b0}}, trailing}) < shift - 12'sd1;
        inexact = round_bit | sticky | sticky_in;
        sig = shifted[24:0] + {24'd0, round_bit && (sticky || shifted[0])};
        exp_field = shift + lsb_exp + 12'sd150;
        if (sig[24]) begin
            sig = sig >> 1;
            exp_field = exp_field + 12'sd1;
        end
        if (!sig[23]) exp_field = 12'sd0;
        overflow = exp_field >= 12'sd255;
        result = overflow ? {sign, 8'hff, 23'd0} : {sign, exp_field[7:0], sig[22:0]};
        flags = {2'b0, overflow,
                 $signed({{(12-LW){1'b0}}, lead}) + lsb_exp < -12'sd126 && inexact, inexact | overflow};
    end
endmodule
