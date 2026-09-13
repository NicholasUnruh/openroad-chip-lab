module training_tile_v3 (
    input wire clk, rst, start,
    input wire transpose_a, transpose_b, accumulate,
    input wire job_bank, host_bank,
    input wire [2:0] write_enable,
    input wire [3:0] write_addr,
    input wire [31:0] write_data,
    input wire [3:0] read_addr,
    output wire [31:0] read_data,
    output wire write_ready,
    output wire busy, done,
    output wire [4:0] exception_flags
);
    training_tile_engine #(.BUFFERED(1), .BF16_INPUT(1)) engine (
        .clk(clk), .rst(rst), .start(start),
        .transpose_a(transpose_a), .transpose_b(transpose_b), .accumulate(accumulate),
        .job_bank(job_bank), .host_bank(host_bank),
        .write_enable(write_enable), .write_addr(write_addr), .write_data(write_data),
        .read_addr(read_addr), .read_data(read_data), .write_ready(write_ready),
        .busy(busy), .done(done), .exception_flags(exception_flags)
    );
endmodule
