module training_tile_v2 (
    input wire clk,
    input wire rst,
    input wire start,
    input wire transpose_a,
    input wire transpose_b,
    input wire accumulate,
    input wire [2:0] write_enable,
    input wire [3:0] write_addr,
    input wire [31:0] write_data,
    input wire [3:0] read_addr,
    output wire [31:0] read_data,
    output wire busy,
    output wire done,
    output wire [4:0] exception_flags
);
    training_tile_engine engine (
        .clk(clk), .rst(rst), .start(start),
        .transpose_a(transpose_a), .transpose_b(transpose_b), .accumulate(accumulate),
        .job_bank(1'b0), .host_bank(1'b0),
        .write_enable(write_enable), .write_addr(write_addr), .write_data(write_data),
        .read_addr(read_addr), .read_data(read_data), .write_ready(),
        .busy(busy), .done(done), .exception_flags(exception_flags)
    );
endmodule
