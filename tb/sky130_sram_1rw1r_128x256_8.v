// Cycle model of the ORFS sky130ram macro: inputs register on the rising edge, read data is
// usable for the following clock only. Unselected or colliding reads return poison.
module sky130_sram_1rw1r_128x256_8 (
    input wire clk0, csb0, web0,
    input wire [15:0] wmask0,
    input wire [7:0] addr0,
    input wire [127:0] din0,
    output wire [127:0] dout0,
    input wire clk1, csb1,
    input wire [7:0] addr1,
    output reg [127:0] dout1
);
    localparam [127:0] POISON = {4{32'h7fa5a5a5}};
    reg [127:0] mem [0:255];
    integer byte_index;
    assign dout0 = POISON;
    always @(posedge clk0)
        if (!csb0 && !web0)
            for (byte_index = 0; byte_index < 16; byte_index = byte_index + 1)
                if (wmask0[byte_index]) mem[addr0][8*byte_index +: 8] <= din0[8*byte_index +: 8];
    always @(posedge clk1)
        dout1 <= csb1 || (!csb0 && !web0 && addr0 == addr1) ? POISON : mem[addr1];
endmodule
