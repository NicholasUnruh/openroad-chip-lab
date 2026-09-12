module mac8 (
    input wire clk,
    input wire rst,
    input wire en,
    input wire [7:0] a,
    input wire [7:0] b,
    output reg [23:0] acc
);
    always @(posedge clk) begin
        if (rst)
            acc <= 24'd0;
        else if (en)
            acc <= acc + (a * b);
    end
endmodule
